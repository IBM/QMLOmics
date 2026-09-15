"""
Quantum Ensemble Learning Module
=================================

This module implements quantum ensemble learning algorithms using controlled
swap operations and quantum superposition to create ensembles of training data
arrangements. The ensemble leverages quantum superposition to evaluate multiple
training set configurations simultaneously.

Supports two ensemble construction methods:
- Fixed swap patterns: Deterministic controlled-SWAP operations
- Random unitaries: Haar-random unitary transformations

References
----------
- Macaluso et al., "A variational algorithm for quantum ensemble learning"
  IET Quantum Communication (2023)
- Rhrissorrakrai et al., "Quantum Ensembling Methods for Healthcare and Life Science"
  Briefings in Bioinformatics (2026)
  https://doi.org/10.1093/bib/bbag280
"""

import time
import numpy as np
import scipy.stats
from typing import List, Literal

# Qiskit imports
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import UnitaryGate

# Local imports
from qbiocode.evaluation.model_evaluation import modeleval
from qbiocode.utils import (
    normalize_data,
    label_to_array,
    prepare_training_set,
    retrieve_probabilities,
    execute_circuit,
)


def build_cosine_classifier(train: np.ndarray, test: np.ndarray, 
                            label_train: np.ndarray) -> QuantumCircuit:
    """
    Build quantum cosine similarity classifier circuit.
    
    Implements a quantum cosine similarity classifier using controlled swap
    operations and Hadamard gates to measure similarity between training
    and test data points.
    
    Parameters
    ----------
    train : np.ndarray
        Training data point as normalized vector (length must be power of 2)
    test : np.ndarray
        Test data point as normalized vector
    label_train : np.ndarray
        Training label as normalized probability vector [p0, p1]
    
    Returns
    -------
    QuantumCircuit
        Quantum circuit implementing the cosine classifier
    
    Notes
    -----
    The circuit computes ``P(0) = 1/2 + 1/2 * |<train|test>|^2``
    """
    qubits_per = int(np.log2(len(train)))
    
    c = ClassicalRegister(1, 'c')
    x_train = QuantumRegister(qubits_per, 'x_train')
    x_test = QuantumRegister(qubits_per, 'x_test')
    y_train = QuantumRegister(1, 'y_train')
    y_test = QuantumRegister(1, 'y_test')
    
    qc = QuantumCircuit(x_train, x_test, y_train, y_test, c)
    
    # Initialize states - convert to list for Qiskit compatibility
    qc.initialize(train.tolist() if isinstance(train, np.ndarray) else train, [x_train])
    qc.initialize(test.tolist() if isinstance(test, np.ndarray) else test, [x_test])
    qc.initialize(label_train.tolist() if isinstance(label_train, np.ndarray) else label_train, [y_train])
    qc.barrier()
    
    # SWAP test
    qc.h(y_test)
    qc.cswap(y_test, x_train, x_test)
    qc.h(y_test)
    qc.barrier()
    
    # Label integration
    qc.cx(y_train, y_test)
    qc.measure(y_test, c)
    
    return qc


def build_ensemble_circuit(X_data: np.ndarray, Y_data: np.ndarray, 
                          x_test: List[complex], n_swap: int = 1, 
                          d: int = 2, mode: str = "balanced",
                          ensemble_method: Literal["swap", "random_unitary"] = "swap",
                          barriers: bool = False) -> QuantumCircuit:
    """
    Build quantum ensemble classifier circuit.
    
    Creates a quantum ensemble learning circuit using either fixed swap
    operations or random unitary transformations.
    
    Parameters
    ----------
    X_data : np.ndarray, shape (n_samples, n_features)
        Training data points (normalized, n_features must be power of 2)
    Y_data : np.ndarray, shape (n_samples, 2)
        Training labels as one-hot encoded vectors
    x_test : List[complex]
        Test data point to classify (normalized)
    n_swap : int, optional
        Number of swap/unitary operations per control qubit (default: 1)
    d : int, optional
        Number of control qubits, creates 2^d ensemble members (default: 2)
    mode : str, optional
        Sampling strategy: "balanced", "unbalanced", or "pair_sample" (default: "balanced")
    ensemble_method : {"swap", "random_unitary"}, optional
        Method for ensemble construction:
        - "swap": Fixed controlled-SWAP operations (faster, deterministic)
        - "random_unitary": Haar-random unitaries (more general, slower)
        (default: "swap")
    barriers : bool, optional
        Add barrier gates for visualization (default: False)
    
    Returns
    -------
    QuantumCircuit
        Quantum circuit implementing the ensemble classifier
    
    Notes
    -----
    Total qubits: d + 2*n_samples*log2(n_features) + n_samples + 1 (+ 1 for random_unitary)
    """
    
    def cswap_obs(c, a, b):
        """Controlled swap of observation qubits between indices a and b."""
        qubit_indices_a = train_qubit_map[a]
        qubit_indices_b = train_qubit_map[b]
        for (i, j) in zip(qubit_indices_a, qubit_indices_b):
            qc.cswap(control[c], data[i], data[j])
    
    def cswap_cosine(a):
        """Controlled swap for cosine similarity measurement."""
        qubit_indices_a = train_qubit_map[a]
        for (j, i) in enumerate(qubit_indices_a):
            qc.cswap(label_test, data[i], data_test[j])
    
    def cswap_labels(c, a, b):
        """Controlled swap of label qubits."""
        qc.cswap(c, labels[a], labels[b])
    
    n_obs = len(X_data)
    qubits_per = int(np.log2(X_data.shape[1]))
    n_obs_qubits = qubits_per * n_obs
    n_test_qubits = qubits_per
    
    control = QuantumRegister(d, 'control')
    data = QuantumRegister(n_obs_qubits, 'x')
    labels = QuantumRegister(n_obs, 'y')
    data_test = QuantumRegister(n_test_qubits, 'test_data')
    label_test = QuantumRegister(1, 'test_label')
    c = ClassicalRegister(1)
    
    # For random unitary method, add prediction qubit
    if ensemble_method == "random_unitary":
        prediction = QuantumRegister(1, 'pred_qubit')
        qc = QuantumCircuit(control, data, labels, data_test, prediction, label_test, c)
    else:
        qc = QuantumCircuit(control, data, labels, data_test, label_test, c)
    
    # Initialize test data
    qc.initialize(x_test, data_test)
    
    # Initialize training data and labels
    train_qubit_map = {}
    for index in range(n_obs):
        indices = list(range(index * qubits_per, (index + 1) * qubits_per))
        x_init = X_data[index].tolist() if isinstance(X_data[index], np.ndarray) else X_data[index]
        y_init = Y_data[index].tolist() if isinstance(Y_data[index], np.ndarray) else Y_data[index]
        qc.initialize(x_init, [data[indices]])
        qc.initialize(y_init, [labels[index]])
        train_qubit_map[index] = indices
    
    # Create superposition on control qubits
    for i in range(d):
        qc.h(control[i])
    
    if barriers:
        qc.barrier()
    
    # Apply ensemble operations based on method
    if ensemble_method == "random_unitary":
        # Sample random unitaries from Haar measure
        unitary_sampler = scipy.stats.unitary_group(2**(n_obs_qubits + n_obs))
        
        if mode == 'balanced':
            for i in range(d - 1):
                for _ in range(n_swap):
                    U = unitary_sampler.rvs()
                    U1 = UnitaryGate(U)
                    CU1 = U1.control(1)
                    qc.append(CU1, [control[i]] + [x for x in data] + [x for x in labels])
                    
                    if barriers:
                        qc.barrier()
                    
                    qc.x(control[i])
                    
                    if barriers:
                        qc.barrier()
                    
                    U = unitary_sampler.rvs()
                    U2 = UnitaryGate(U)
                    CU2 = U2.control(1)
                    qc.append(CU2, [control[i]] + [x for x in data] + [x for x in labels])
                    
                    if barriers:
                        qc.barrier()
            
            # Final swap for balanced mode
            U = np.random.choice(range(int(n_obs / 2)), 1, replace=False)
            U = np.insert(U, 1, n_obs - 1)
            d1, d2 = U[0], U[1]
            cswap_obs(d - 1, d1, d2)
            cswap_labels(d - 1, d1, d2)
            qc.x(control[d - 1])
    
    else:  # Fixed swap method
        if mode == 'balanced':
            for i in range(d - 1):
                for j in range(n_swap):
                    # Swap within first class
                    U = np.random.choice(range(int(n_obs / 2)), 2, replace=False)
                    U_b = np.random.choice(range(len(train_qubit_map[U[0]])), 1)[0]
                    d1 = train_qubit_map[U[0]][U_b]
                    d2 = train_qubit_map[U[1]][U_b]
                    qc.cswap(control[i], data[d1], data[d2])
                    qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])
                    
                    # Swap within second class
                    U = np.random.choice(range(int(n_obs / 2), n_obs), 2, replace=False)
                    U_b = np.random.choice(range(len(train_qubit_map[U[0]])), 1)[0]
                    d1 = train_qubit_map[U[0]][U_b]
                    d2 = train_qubit_map[U[1]][U_b]
                    qc.cswap(control[i], data[d1], data[d2])
                    qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])
                
                qc.x(control[i])
                
                for j in range(n_swap):
                    U = np.random.choice(range(int(n_obs / 2)), 2, replace=False)
                    U_b = np.random.choice(range(len(train_qubit_map[U[0]])), 1)[0]
                    d1 = train_qubit_map[U[0]][U_b]
                    d2 = train_qubit_map[U[1]][U_b]
                    qc.cswap(control[i], data[d1], data[d2])
                    qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])
                    
                    U = np.random.choice(range(int(n_obs / 2), n_obs), 2, replace=False)
                    U_b = np.random.choice(range(len(train_qubit_map[U[0]])), 1)[0]
                    d1 = train_qubit_map[U[0]][U_b]
                    d2 = train_qubit_map[U[1]][U_b]
                    qc.cswap(control[i], data[d1], data[d2])
                    qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])
                    
                    if barriers:
                        qc.barrier()
            
            # Final swap for balanced mode
            qc.x(control[d - 1])
            U = np.random.choice(range(int(n_obs / 2)), 1, replace=False)
            U = np.insert(U, 1, n_obs - 1)
            U_b = np.random.choice(range(len(train_qubit_map[U[0]])), 1)[0]
            d1 = train_qubit_map[U[0]][U_b]
            d2 = train_qubit_map[U[1]][U_b]
            qc.cswap(control[d - 1], data[d1], data[d2])
            qc.cswap(control[d - 1], labels[int(U[0])], labels[int(U[1])])
        
        elif mode == "unbalanced":
            for i in range(d):
                for j in range(n_swap):
                    U = np.random.choice(range(n_obs), 2, replace=False)
                    U_b = np.random.choice(range(len(train_qubit_map[U[0]])), 1)[0]
                    d1 = train_qubit_map[U[0]][U_b]
                    d2 = train_qubit_map[U[1]][U_b]
                    qc.cswap(control[i], data[d1], data[d2])
                    qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])
                
                qc.x(control[i])
                
                for j in range(n_swap):
                    U = np.random.choice(range(n_obs), 2, replace=False)
                    U_b = np.random.choice(range(len(train_qubit_map[U[0]])), 1)[0]
                    d1 = train_qubit_map[U[0]][U_b]
                    d2 = train_qubit_map[U[1]][U_b]
                    qc.cswap(control[i], data[d1], data[d2])
                    qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])
        
        elif mode == "pair_sample":
            for i in range(d):
                for j in range(n_swap):
                    pairs = np.random.choice(range(n_obs), n_obs, replace=False)
                    for U in pairs.reshape(int(len(pairs) / 2), 2):
                        U_b = np.random.choice(range(len(train_qubit_map[U[0]])), 1)[0]
                        d1 = train_qubit_map[U[0]][U_b]
                        d2 = train_qubit_map[U[1]][U_b]
                        qc.cswap(control[i], data[d1], data[d2])
                        qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])
                
                qc.x(control[i])
                if barriers:
                    qc.barrier()
                
                for j in range(n_swap):
                    pairs = np.random.choice(range(n_obs), n_obs, replace=False)
                    for U in pairs.reshape(int(len(pairs) / 2), 2):
                        U_b = np.random.choice(range(len(train_qubit_map[U[0]])), 1)[0]
                        d1 = train_qubit_map[U[0]][U_b]
                        d2 = train_qubit_map[U[1]][U_b]
                        qc.cswap(control[i], data[d1], data[d2])
                        qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])
                
                if barriers:
                    qc.barrier()
    
    if barriers:
        qc.barrier()
    
    # Final classification step
    ix_cls = n_obs - 1
    qc.h(label_test)
    if barriers:
        qc.barrier()
    cswap_cosine(ix_cls)
    if barriers:
        qc.barrier()
    qc.h(label_test)
    qc.cx(labels[ix_cls], label_test)
    qc.measure(label_test, c)
    
    return qc


def compute_qensemble(X_train: np.ndarray, X_test: np.ndarray, 
                     y_train: np.ndarray, y_test: np.ndarray,
                     args: dict, model: str = 'QEnsemble',
                     data_key: str = '', n_train: int = 4,
                     n_swap: int = 1, d: int = 2,
                     mode: str = "balanced",
                     ensemble_method: Literal["swap", "random_unitary"] = "swap",
                     n_shots: int = 8192,
                     seed: int = 123, device: str = 'CPU',
                     verbose: bool = False):
    """
    Compute quantum ensemble classifier predictions.
    
    This function implements a quantum ensemble learning algorithm using
    either fixed swap operations or random unitary transformations to create
    superpositions of different training data arrangements.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training feature set
    X_test : np.ndarray
        Testing feature set
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Testing labels
    args : dict
        Dictionary containing arguments for backend and settings
    model : str, optional
        Model type (default: 'QEnsemble')
    data_key : str, optional
        Key for the dataset (default: '')
    n_train : int, optional
        Number of training samples to use (must be even, default: 4)
    n_swap : int, optional
        Number of swap/unitary operations per control qubit (default: 1)
    d : int, optional
        Number of control qubits, creates 2^d ensemble members (default: 2)
    mode : str, optional
        Sampling strategy: "balanced", "unbalanced", or "pair_sample" (default: "balanced")
    ensemble_method : {"swap", "random_unitary"}, optional
        Method for ensemble construction:
        - "swap": Fixed controlled-SWAP operations (faster, deterministic)
        - "random_unitary": Haar-random unitaries (more general, slower)
        (default: "swap")
    n_shots : int, optional
        Number of measurement shots (default: 8192)
    seed : int, optional
        Random seed for reproducibility (default: 123)
    device : str, optional
        Device type: 'CPU' or 'GPU' (default: 'CPU')
    verbose : bool, optional
        Print additional information (default: False)
    
    Returns
    -------
    dict
        Dictionary containing evaluation results including accuracy, runtime,
        model parameters, and other relevant metrics
    
    Examples
    --------
    >>> from qbiocode.learning import compute_qensemble
    >>> # Fixed swap ensemble (standard)
    >>> results = compute_qensemble(X_train, X_test, y_train, y_test, args,
    ...                             n_train=4, d=2, ensemble_method="swap")
    >>> # Random unitary ensemble (advanced)
    >>> results = compute_qensemble(X_train, X_test, y_train, y_test, args,
    ...                             n_train=4, d=2, ensemble_method="random_unitary")
    """
    beg_time = time.time()
    
    if verbose:
        method_name = "Random Unitary" if ensemble_method == "random_unitary" else "Fixed Swap"
        print(f"Running Quantum Ensemble Classifier ({method_name})")
        print(f"Training samples: {n_train}, Ensemble depth (d): {d}, Operations: {n_swap}")
        print(f"Mode: {mode}, Shots: {n_shots}")
    
    # Prepare training subset
    X_data, Y_data = prepare_training_set(X_train, y_train, n=n_train, seed=seed)
    
    # Make predictions for each test sample
    predictions = []
    qc = None  # Initialize to avoid unbound variable
    for x_test in X_test:
        x_test_norm = normalize_data(x_test)
        
        # Build and execute circuit
        qc = build_ensemble_circuit(X_data, Y_data, x_test_norm, 
                                    n_swap=n_swap, d=d, mode=mode,
                                    ensemble_method=ensemble_method)
        
        if qc.num_qubits > 36:
            raise ValueError(f"Circuit has {qc.num_qubits} qubits, exceeds simulation limit of 36")
        
        counts = execute_circuit(qc, n_shots=n_shots, device=device, seed=seed)
        probs = retrieve_probabilities(counts)
        predictions.append(probs)
    
    # Convert probabilities to class predictions
    y_predicted = np.array([1 if p[1] > p[0] else 0 for p in predictions])

    # `auc` is computed from these scores alone, never from y_predicted. This is the one
    # learner in the package with no fitted estimator to interrogate -- there is nothing
    # for extract_binary_scores to be called on -- but it needs no help: the ensemble
    # measures a probability per test sample directly, and `retrieve_probabilities`
    # returns it as [p0, p1] normalised to sum to 1. p1 is the positive-class score, and
    # it is the same quantity the argmax above throws away. Note that with few shots it
    # takes only `n_shots + 1` distinct values, so the ranking is coarse; that is a
    # property of the measurement, not of the metric.
    y_score = np.array([p[1] for p in predictions], dtype=float)
    
    # Model parameters
    model_params = {
        'n_train': n_train,
        'n_swap': n_swap,
        'd': d,
        'mode': mode,
        'ensemble_method': ensemble_method,
        'n_shots': n_shots,
        'seed': seed,
        'n_qubits': qc.num_qubits if qc is not None else 0,
        'circuit_depth': qc.depth() if qc is not None else 0
    }
    
    return modeleval(y_test, y_predicted, beg_time, model_params, args, 
                    model=model, verbose=verbose, y_score=y_score)
