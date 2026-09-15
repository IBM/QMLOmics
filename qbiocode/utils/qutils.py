from qiskit_ibm_runtime.qiskit_runtime_service import QiskitRuntimeService


import logging
import math
import os
import re
from functools import reduce

import numpy as np
import pandas as pd
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.library import (
    EfficientSU2,
    PauliFeatureMap,
    RealAmplitudes,
    TwoLocal,
    XGate,
    YGate,
    ZFeatureMap,
    ZZFeatureMap,
)
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, NFT, SPSA, GradientDescent, spsa
from qiskit_ibm_runtime import EstimatorOptions
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerOptions
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import Session
from qiskit_ibm_transpiler.transpiler_service import TranspilerService
from qiskit_aer import AerSimulator

from qbiocode.utils.ibm_account import instantiate_runtime_service


def get_backend_session(args: dict, primitive: str, num_qubits: int):
    """
    This function to get the backend and session for the specified primitive.

    Args:
        args (dict): Dictionary containing backend and other parameters.
        primitive (str): The type of primitive to instantiate ('sampler' or 'estimator').
        num_qubits (int): Number of qubits for the backend.

    Returns:
        backend: The backend instance.
        session: The session instance.
        prim: The instantiated primitive (Sampler or Estimator).
    """
    backend = None
    session = None
    prim = None

    # Checked here rather than left to a bare ``KeyError: 'seed'`` several frames
    # into a notebook: these keys are the caller's contract, and the message has
    # to name which one is missing and what it is for.
    required = ["backend"]
    if args.get("backend") == "simulator":
        required.append("seed")
        if primitive != "estimator":
            required.append("shots")
    missing = [key for key in required if key not in args]
    if missing:
        raise ValueError(
            f"args is missing the required key(s) {missing} for the "
            f"{args.get('backend')!r} backend with the {primitive!r} primitive. "
            f"'backend' selects where circuits run ('simulator', 'simulator_aer', "
            f"or an 'ibm_*' device), 'seed' seeds the statevector primitive so "
            f"runs are reproducible, and 'shots' sets the sampler's shot count. "
            f"Got keys: {sorted(args)}."
        )

    if args["backend"] == "simulator":

        if primitive == "estimator":
            # Estimator primitive
            prim = StatevectorEstimator(seed=args["seed"])
        else:
            prim = StatevectorSampler(seed=args["seed"], default_shots=args["shots"])
    elif ("ibm" in args["backend"]) or (args["backend"] == "simulator_aer"):
        service: QiskitRuntimeService = instantiate_runtime_service(args)
        if args["backend"] == "simulator_aer":
            backend = AerSimulator.from_backend(method = args['sim_method'])
        elif "noisy" in args["backend"]:
            noisy_backend_name = re.sub("noisy_", "", args["backend"])
            backend = AerSimulator.from_backend(service.backend(name=noisy_backend_name), method = args['sim_method'])
        else:
            if args["backend"] == "ibm_least":
                backend = service.least_busy(
                    simulator=False, operational=True, min_num_qubits=num_qubits
                )
            else:
                backend = service.backend(name=args["backend"])

        session = Session(backend=backend)

        if primitive == "sampler":
            prim = get_sampler(mode=session, shots=args["shots"])
        else:
            prim = get_estimator(mode=session, shots=args["shots"], resil_level=args["resil_level"])
    return backend, session, prim


def transpile_circuit(circuit, opt_level, backend, initial_layout, PT=False, dd_sequence="XpXm"):
    """
    This function transpiles the given quantum circuit based on the optimization level and backend.

    Args:
        circuit (QuantumCircuit): The quantum circuit to be transpiled.
        opt_level (int or str): Optimization level for transpilation.
        backend (Backend): The backend to which the circuit will be transpiled.
        initial_layout (Layout): Initial layout for the transpilation.
        PT (bool): Whether to apply pulse twirling. Defaults to False.
        dd_sequence (str): Sequence for dynamical decoupling. Defaults to 'XpXm'.

    Returns:
        t_qc (QuantumCircuit): The transpiled quantum circuit.
    """
    if str(opt_level) == "AI":
        pm = TranspilerService(
            backend_name=backend,
            ai="true",
            optimization_level=3,
        )
    else:
        pm = generate_preset_pass_manager(
            optimization_level=opt_level,
            backend=backend,
            seed_transpiler=42,
            initial_layout=initial_layout,
        )
    t_qc = pm.run(circuit)

    return t_qc


def get_observable(circuit, backend):
    observable = SparsePauliOp.from_list([("Z" * circuit.num_qubits, 1)])
    # observable = SparsePauliOp.from_list([("Z" + "I" * (int(circuit.num_qubits) - 1), 0.5)])
    if "ibm" in backend.name:
        observable = observable.apply_layout(circuit.layout)  # , num_qubits=backend.num_qubits)
    return observable


def get_sampler(
    mode=None,
    shots=1024,
    dd=True,
    dd_seq="XpXm",
    PT=True,
):
    """
    This function creates a Sampler instance with specified options.

    Args:
        mode (Session): The session mode for the sampler.
        shots (int): Number of shots for sampling.
        dd (bool): Whether to enable dynamical decoupling.
        dd_seq (str): Sequence type for dynamical decoupling.
        PT (bool): Whether to enable pulse twirling.

    Returns:
        Sampler: An instance of the Sampler with the specified options.
    """

    sampler_options = SamplerOptions()

    ## ERROR SUPPRESSION TESTING ###
    sampler_options.default_shots = shots
    if dd:
        sampler_options.dynamical_decoupling.enable = dd
        sampler_options.dynamical_decoupling.sequence_type = dd_seq
        sampler_options.dynamical_decoupling.extra_slack_distribution = "middle"
        sampler_options.dynamical_decoupling.scheduling_method = "alap"
    if PT:
        sampler_options.twirling.enable_gates = True
        sampler_options.twirling.enable_measure = False
        sampler_options.twirling.num_randomizations = "auto"
        sampler_options.twirling.shots_per_randomization = "auto"
        sampler_options.twirling.strategy = "active-accum"  ### TRY VARYING THIS ###

    sampler = Sampler(mode=mode, options=sampler_options)

    return sampler


def get_estimator(
    mode=None,
    shots=1024,
    resil_level=2,
    dd=True,
    dd_seq="XpXm",
    PT=True,
):
    """
    This function creates an Estimator instance with specified options.

    Args:
        mode (Session): The session mode for the estimator.
        shots (int): Number of shots for estimation.
        resil_level (int): Resilience level for error suppression.
        dd (bool): Whether to enable dynamical decoupling.
        dd_seq (str): Sequence type for dynamical decoupling.
        PT (bool): Whether to enable pulse twirling.
    Returns:
        Estimator: An instance of the Estimator with the specified options.
    """

    experimental_opts = {}
    # experimental_opts["execution_path"] = "gen3-turbo"

    estimator_options = EstimatorOptions(experimental=experimental_opts)

    ## ERROR SUPPRESSION TESTING ###
    estimator_options.default_shots = shots
    estimator_options.resilience_level = resil_level
    if dd:
        estimator_options.dynamical_decoupling.enable = dd
        estimator_options.dynamical_decoupling.sequence_type = dd_seq
        estimator_options.dynamical_decoupling.extra_slack_distribution = "middle"
        estimator_options.dynamical_decoupling.scheduling_method = "alap"
    if PT:
        estimator_options.twirling.enable_gates = True
        estimator_options.twirling.enable_measure = False
        estimator_options.twirling.num_randomizations = "auto"
        estimator_options.twirling.shots_per_randomization = "auto"
        estimator_options.twirling.strategy = "active-accum"  ### TRY VARYING THIS ###

    estimator = Estimator(mode=mode, options=estimator_options)
    return estimator


def get_ansatz(ansatz_type, feat_dimension, reps=1, entanglement="linear"):
    """
    This function returns an ansatz based on the specified type and parameters.
    It supports 'esu2', 'amp', and 'twolocal' ansatz types, constructing it using the specified feature dimension,
    number of repetitions, and entanglement type.

    Args:
        ansatz_type (str): Type of the ansatz ('esu2', 'amp', or 'twolocal').
        feat_dimension (int): Number of qubits for the ansatz.
        reps (int): Number of repetitions for the ansatz.
        entanglement (str): Type of entanglement for the ansatz.
    Returns:
        ansatz: An instance of the specified ansatz type.
    """
    if ansatz_type == "esu2":
        ansatz = EfficientSU2(feat_dimension, ["ry", "rz"], entanglement, reps=reps)
    elif ansatz_type == "amp":
        ansatz = RealAmplitudes(num_qubits=feat_dimension, reps=reps)
    elif ansatz_type == "twolocal":
        ansatz = TwoLocal(feat_dimension, ["ry", "rz"], "cz", entanglement, reps=reps)
    return ansatz


#: Feature-map names accepted by :func:`get_feature_map`, in the spelling the
#: config files use. Exported so callers can validate at their own boundary --
#: e.g. :func:`qbiocode.learning.compute_pqk.compute_pqk` rejects a mistyped
#: ``encoding`` before it creates directories or reads a projection cache.
SUPPORTED_FEATURE_MAPS = ("Z", "ZZ", "P")

#: Entanglement patterns accepted by qiskit's ZZ/Pauli feature maps. Validated
#: because qiskit itself reports an unknown pattern as
#: "Something went wrong in Rust space", which names neither the parameter nor
#: the value. A callable or an explicit index list is also valid and is passed
#: through to qiskit unchecked.
SUPPORTED_ENTANGLEMENTS = ("full", "linear", "reverse_linear", "pairwise", "circular", "sca")

#: Optimizer names accepted by :func:`get_optimizer`.
SUPPORTED_OPTIMIZERS = ("SPSA", "COBYLA", "GradientDescent", "L_BFGS_B")


def unit_coefficient_data_map(x):
    """
    Map a row of features to the rotation angle of a feature-map gate.

    This is the ``data_map_func`` used by the projected-quantum-kernel paths. It
    divides by two at every step so that every multiplicative factor of a data
    feature inside a single-qubit gate is 1.0, rather than qiskit's default
    ``prod(pi - x_i)``.

    Args:
        x: one row of features -- either numeric, or a symbolic
            ``ParameterVector``. Qiskit calls a data map with *both*: with
            symbolic parameters when it builds the feature-map circuit
            (``PauliFeatureMap.pauli_block`` passes a ``ParameterVector``), and
            with numeric values only if a caller evaluates the map directly.

    Returns:
        The mapped angle: a ``float`` for numeric input, or the unevaluated
        ``ParameterExpression`` for symbolic input.

    Notes:
        Narrowing the symbolic case with ``float()`` raises
        ``TypeError: Parameter expression with unbound parameters ... is not
        numeric`` and makes every ``data_map=True`` feature map unbuildable, so
        the symbolic expression is returned untouched for qiskit to bind later.
    """
    coeff = x[0] / 2 if len(x) == 1 else reduce(lambda m, n: (m * n) / 2, x)
    try:
        return float(coeff)
    except (TypeError, ValueError):
        return coeff


def get_feature_map(feature_map, feat_dimension, reps=1, entanglement="linear", data_map_func=None):
    """
    This function returns a feature map based on the specified type and parameters.
    It supports 'Z', 'ZZ', and 'P' feature maps, constructing it using the specified feature dimension,
    number of repetitions, entanglement type, and data mapping function.
    Args:
        feature_map (str): Type of the feature map ('Z', 'ZZ', or 'P').
        feat_dimension (int): Number of qubits for the feature map.
        reps (int): Number of repetitions for the feature map.
        entanglement (str): Type of entanglement for the feature map.
        data_map_func (callable, optional): Function to map data to the feature map parameters.
    Returns:
        feature_map: An instance of the specified feature map type.
        feat_dimension (int): The number of qubits in the feature map.

    Raises:
        ValueError: if ``feature_map`` is not one of ``'Z'``, ``'ZZ'``, ``'P'``,
            or if ``feat_dimension``/``reps`` is not a positive integer.
    """
    # Validated here rather than left to the if/elif chain: an unrecognized name
    # used to fall through every branch and return the *input string* as the
    # feature map, which then failed several frames later with
    # "'str' object has no attribute 'num_qubits'" -- a message that says nothing
    # about the actual mistake, a mistyped encoding.
    if feature_map not in SUPPORTED_FEATURE_MAPS:
        raise ValueError(
            f"Unsupported feature_map {feature_map!r}. Expected one of "
            f"{SUPPORTED_FEATURE_MAPS} (case-sensitive): 'Z' for ZFeatureMap, "
            f"'ZZ' for ZZFeatureMap, 'P' for PauliFeatureMap."
        )
    if isinstance(entanglement, str) and entanglement not in SUPPORTED_ENTANGLEMENTS:
        raise ValueError(
            f"Unsupported entanglement {entanglement!r}. Expected one of "
            f"{SUPPORTED_ENTANGLEMENTS}, or a callable/index list passed straight "
            f"through to qiskit."
        )
    if not isinstance(feat_dimension, (int, np.integer)) or feat_dimension < 1:
        raise ValueError(
            f"feat_dimension is the number of qubits and must be a positive "
            f"integer; got {feat_dimension!r}."
        )
    if not isinstance(reps, (int, np.integer)) or reps < 1:
        raise ValueError(
            f"reps is the number of feature-map repetitions and must be a "
            f"positive integer; got {reps!r}."
        )

    # Get Feature Map
    if feature_map == "Z":
        feature_map = ZFeatureMap(
            feat_dimension, reps=reps, parameter_prefix="a", data_map_func=data_map_func
        )
    elif feature_map == "ZZ":
        feature_map = ZZFeatureMap(
            feature_dimension=feat_dimension,
            reps=reps,
            entanglement=entanglement,
            parameter_prefix="a",
            data_map_func=data_map_func,
        )
    elif feature_map == "P":
        feature_map = PauliFeatureMap(
            feature_dimension=feat_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
        )

    # print("The number of qubits is:", feature_map.num_qubits)
    # print("The number of parameters is:", feature_map.num_parameters)

    return feature_map, feat_dimension


def get_optimizer(
    type="COBYLA", max_iter=100, learning_rate_a=None, perturbation_gamma=None, prior_iter=0
):
    """
    This function returns an optimizer based on the specified type and parameters.
    It supports 'SPSA', 'COBYLA', 'GradientDescent', and 'L_BFGS_B' optimizer types,
    constructing it using the specified maximum iterations, learning rate, perturbation gamma, and prior iterations.

    Args:
        type (str): Type of the optimizer ('SPSA', 'COBYLA', 'GradientDescent', or 'L_BFGS_B').
        max_iter (int): Maximum number of iterations for the optimizer.
        learning_rate_a (float, optional): Initial learning rate for SPSA.
        perturbation_gamma (float, optional): Perturbation gamma for SPSA.
        prior_iter (int): Number of prior iterations to consider.

    Returns:
        optimizer: An instance of the specified optimizer type.

    Raises:
        ValueError: if ``type`` is not one of ``'SPSA'``, ``'COBYLA'``,
            ``'GradientDescent'``, ``'L_BFGS_B'``, or if ``max_iter`` is not a
            positive integer.
    """
    # Validated up front: an unrecognized name previously left `optimizer`
    # unbound and surfaced as "cannot access local variable 'optimizer'", which
    # points at this function's internals rather than at the caller's typo.
    if type not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer {type!r}. Expected one of "
            f"{SUPPORTED_OPTIMIZERS} (case-sensitive)."
        )
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(
            f"max_iter must be a positive integer number of optimizer "
            f"iterations; got {max_iter!r}."
        )

    if type == "SPSA":
        if (learning_rate_a != None) & (perturbation_gamma != None):
            # set up the power series
            def learning_rate():
                return spsa.powerseries(learning_rate_a, 0.602, 0)

            gen = learning_rate()
            learning_rates = np.array([next(gen) for _ in range(max_iter + prior_iter)])
            learning_rates = learning_rates[prior_iter : (max_iter + prior_iter)]

            def perturbation():
                return spsa.powerseries(0.2, perturbation_gamma)

            gen = perturbation()
            perturbations = np.array([next(gen) for _ in range(max_iter + prior_iter)])
            perturbations = perturbations[prior_iter : (max_iter + prior_iter)]

            optimizer = SPSA(
                maxiter=max_iter, learning_rate=learning_rates, perturbation=perturbations
            )
        else:
            optimizer = SPSA(maxiter=max_iter)
    elif type == "COBYLA":
        optimizer = COBYLA(maxiter=max_iter)
    elif type == "GradientDescent":
        optimizer = GradientDescent(maxiter=max_iter)
    elif type == "L_BFGS_B":
        # `=`, not `==`. This was a comparison, so the branch built an optimizer,
        # discarded it, and left `optimizer` unbound -- every request for
        # L_BFGS_B (an option both compute_vqc and compute_qnn advertise in their
        # Literal type hints) raised UnboundLocalError instead of running.
        optimizer = L_BFGS_B(maxiter=max_iter)

    return optimizer



def retrieve_probabilities(counts: dict) -> list:
    """
    Extract probability predictions from measurement counts.
    
    Converts raw measurement counts from quantum circuit execution into
    probability predictions for binary classification. Handles edge cases
    where only one outcome is observed.
    
    Parameters
    ----------
    counts : dict
        Measurement counts with keys '0' and/or '1'
        Example: {'0': 4123, '1': 4069}
    
    Returns
    -------
    list of float
        [p0, p1] where p0 is probability of class 0 and p1 is probability
        of class 1. Always sums to 1.0.
    
    Notes
    -----
    - Handles missing keys gracefully (assigns probability 0 or 1)
    - If only '0' observed: returns [1.0, 0.0]
    - If only '1' observed: returns [0.0, 1.0]
    - If both observed: returns normalized probabilities
    
    Examples
    --------
    >>> counts = {'0': 6000, '1': 2000}
    >>> retrieve_probabilities(counts)
    [0.75, 0.25]
    
    >>> counts = {'0': 8192}  # Only one outcome
    >>> retrieve_probabilities(counts)
    [1.0, 0.0]
    """
    state_zero = '0'
    state_one = '1'
    
    try:
        p0 = counts[state_zero] / (counts[state_zero] + counts[state_one])
        p1 = 1 - p0
    except KeyError:
        if list(counts.keys())[0] == state_zero:
            p0, p1 = 1.0, 0.0
        else:
            p0, p1 = 0.0, 1.0
    
    return [p0, p1]


def execute_circuit(qc, n_shots: int = 8192, device: str = 'CPU', seed: int = None):
    """
    Execute quantum circuit on Aer simulator.
    
    General-purpose function for executing quantum circuits using the Qiskit
    Aer simulator with statevector method. Useful for custom quantum algorithms
    that need direct circuit execution without the full runtime service setup.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit to execute
    n_shots : int, optional
        Number of measurement shots (default: 8192)
    device : str, optional
        Device type: 'CPU' or 'GPU' (default: 'CPU')
        Note: GPU requires qiskit-aer-gpu installation
    
    Returns
    -------
    dict
        Measurement counts dictionary with bitstring keys and count values
        Example: {'0': 4123, '1': 4069}
    
    Notes
    -----
    - Uses statevector simulation method for exact state evolution
    - Automatically transpiles circuit with optimization level 3
    - Parallel threshold set to 50 qubits for statevector parallelization
    - For hardware execution, use get_backend_session() instead
    
    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> from qbiocode.utils import execute_circuit
    >>> qc = QuantumCircuit(2, 2)
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> qc.measure([0, 1], [0, 1])
    >>> counts = execute_circuit(qc, n_shots=1024)
    >>> print(counts)
    {'00': 512, '11': 512}
    """
    from qiskit.compiler import transpile
    from qiskit_aer import AerSimulator
    
    backend = AerSimulator(method='statevector', device=device, 
                          statevector_parallel_threshold=50)
    tqc = transpile(qc, backend, optimization_level=3)
    # seed_simulator, without which the shot sampling is drawn from OS entropy. This call
    # used to omit it, so `compute_qensemble`'s documented `seed` parameter reached the
    # training-set selection but never the measurement -- every metric it reported drifted
    # between runs at an identical seed, and nothing said why.
    result = backend.run([tqc], shots=n_shots, seed_simulator=seed).result()
    return result.get_counts(tqc)
