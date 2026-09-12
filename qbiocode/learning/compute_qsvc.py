import time
from typing import Literal

import numpy as np
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# from qiskit.primitives import Sampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from sklearn.model_selection import GridSearchCV

import qbiocode.utils.qutils as qutils

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import modeleval
from qbiocode.learning._tuning import (
    build_search_space,
    record_tuned_params,
    run_function_study,
)
from qbiocode.learning._grid import warn_ignored_hyperparameter

# ====== Scikit-learn imports ======


# ====== Qiskit imports ======


def compute_qsvc(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    model="QSVC",
    data_key="",
    C=1,
    gamma="scale",
    pegasos=False,
    encoding: Literal["ZZ", "Z", "P"] = "ZZ",
    entanglement="linear",
    primitive="sampler",
    reps=2,
    verbose=False,
    local_optimizer="",
):
    """
    This function computes a quantum support vector classifier (QSVC) using the Qiskit Machine Learning library.
    It takes training and testing datasets, along with various parameters to configure the QSVC model.
    It initializes the quantum feature map, sets up the backend and session, and fits the QSVC model to the training data.
    It then predicts the labels for the test data and evaluates the model's performance.
    The function returns the performance results, including accuracy, F1-score, AUC, runtime, as well as model parameters, and other relevant metrics.

    Args:
        X_train (np.ndarray): Training feature set.
        X_test (np.ndarray): Testing feature set.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
        args (dict): Dictionary containing arguments for the quantum backend and other settings.
        model (str): Model type, default is 'QSVC'.
        data_key (str): Key for the dataset, default is an empty string.
        C (float): Regularization parameter for the SVM, default is 1.
        gamma (str or float): Kernel coefficient, default is 'scale'.
        pegasos (bool): Whether to use Pegasos QSVC, default is False.
        encoding (str): Feature map encoding type, options are 'ZZ', 'Z', or 'P', default is 'ZZ'.
        entanglement (str): Entanglement strategy for the feature map, default is 'linear'.
        primitive (str): Primitive type to use, default is 'sampler'.
        reps (int): Number of repetitions for the feature map, default is 2.
        verbose (bool): Whether to print additional information, default is False.

    Returns:
        modeleval (dict): A dictionary containing the evaluation results, including accuracy, runtime, model parameters, and other relevant metrics.
    """
    beg_time = time.time()

    # choose a method for mapping your features onto the circuit
    feature_map, _ = qutils.get_feature_map(
        feature_map=encoding, feat_dimension=X_train.shape[1], reps=reps, entanglement=entanglement
    )

    #  Generate the backend, session and primitive
    backend, session, prim = qutils.get_backend_session(
        args, primitive, num_qubits=feature_map.num_qubits
    )

    print(f"Currently running a quantum support vector classifier (QSVC) on this dataset.")
    print(f"The number of qubits in your circuit is: {feature_map.num_qubits}")
    print(f"The number of parameters in your circuit is: {feature_map.num_parameters}")

    if "simulator" == args["backend"]:
        fidelity = ComputeUncompute(sampler=prim)
    else:
        # Need to instatiate a basic pass manager to store the chosen hardware backend
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        fidelity = ComputeUncompute(
            sampler=prim, pass_manager=pm
        )  # , num_virtual_qubits = feature_map.num_qubits )

    Qkernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    if pegasos == True:
        qsvc = PegasosQSVC(C=C, quantum_kernel=Qkernel)
    else:
        qsvc = QSVC(C=C, gamma=gamma, quantum_kernel=Qkernel)

    model_fit = qsvc.fit(X_train, y_train)
    # model_params = model_fit.get_params()
    hyperparameters = {
        "feature_map": feature_map.__class__.__name__,
        "quantum_kernel": Qkernel.__class__.__name__,
        "C": C,
        "gamma": gamma,
    }
    model_params = hyperparameters
    y_predicted = qsvc.predict(X_test)

    if not isinstance(session, type(None)):
        session.close()

    return modeleval(
        y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose
    )


def compute_qsvc_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="QSVC",
    data_key="",
    C=None,
    gamma=None,
    pegasos=None,
    encoding=None,
    entanglement=None,
    primitive=None,
    reps=None,
    local_optimizer=None,
    *,
    n_trials=10,
    validation_split=0.25,
):
    """Tune QSVC's hyperparameters with Optuna, then run it at the best ones found.

    The quantum counterpart of the classical ``compute_*_opt`` functions, and driven by
    the same ``gridsearch_qsvc_args`` config block -- a list is a choice, a
    ``{low, high}`` mapping is a range. It differs in how a candidate is scored: a
    quantum fit builds an n-by-n fidelity kernel by circuit simulation, so scoring by
    k-fold cross-validation would multiply an already expensive search by k. Each trial
    is scored once, on a stratified holdout carved out of ``X_train``; the caller's test
    set is never touched by the search.

    Only reachable when the config sets both ``grid_search: True`` and
    ``tune_quantum: True``. Tuning against a real device is refused unless
    ``allow_hardware_tuning: True`` -- every trial would be a queued job.

    Args:
        X_train (array-like): Training data features. Split again internally to score
            candidates; the final model is refitted on all of it.
        X_test (array-like): Test data features, used only for the final evaluation.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Run configuration. ``backend``, ``shots`` and ``seed`` are read
            from it by the underlying quantum function.
        verbose (bool): If True, prints additional information during execution.
        model (str): Name of the model being used, default is 'QSVC'.
        data_key (str): Key for identifying the dataset.
        C (list or dict): Regularization strength values to search. None leaves it at the default.
        gamma (list or dict): Kernel coefficient values to search. None leaves it at the default.
        pegasos (list or dict): Whether to use the Pegasos QSVC solver. None leaves it at the default.
        encoding (list or dict): Feature-map values to search ('Z', 'ZZ', 'P'). None leaves it at the default.
        entanglement (list or dict): Entanglement patterns to search ('linear', 'full', ...). None leaves it at the default.
        primitive (list or dict): Qiskit primitives to search ('sampler', 'estimator'). None leaves it at the default.
        reps (list or dict): Feature-map repetition counts to search. None leaves it at the default.
        local_optimizer (list or dict): Accepted so a shared config block can name it,
            and warned about -- compute_qsvc takes the parameter and never reads it.
        n_trials (int): Trial budget, default 10 -- an order of magnitude below the
            classical default because each trial is a quantum fit. Lowered
            automatically when the configured values describe fewer combinations.
        validation_split (float): Fraction of the training data held out to score
            candidates on, default 0.25.

    Returns:
        modeleval (dict): The evaluation of the model at the best hyperparameters found,
        with the tuned values recorded in the results frame and the reported time
        covering the whole search rather than only the final fit.
    """
    beg_time = time.time()
    # Accepted by compute_qsvc and never read, so a grid over it would
    # multiply the trials while every one returned the same model.
    if local_optimizer:
        warn_ignored_hyperparameter("qsvc", "local_optimizer", "QSVC does not read -- the kernel is fitted by libsvm, not an optimizer.")

    candidates = {
        "C": C,
        "gamma": gamma,
        "pegasos": pegasos,
        "encoding": encoding,
        "entanglement": entanglement,
        "primitive": primitive,
        "reps": reps,
    }

    best_params = run_function_study(
        compute_qsvc,
        build_search_space("qsvc", candidates),
        X_train,
        y_train,
        args,
        model="qsvc",
        n_trials=n_trials,
        seed=args.get("seed") if isinstance(args, dict) else None,
        validation_split=validation_split,
    )

    frame = compute_qsvc(
        X_train,
        X_test,
        y_train,
        y_test,
        args,
        model=model,
        data_key=data_key,
        verbose=verbose,
        **best_params,
    )
    return record_tuned_params(frame, best_params, beg_time)
