# ====== Base class imports ======
import time
from typing import Literal

# from qiskit.primitives import Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ====== Qiskit imports ======
from qiskit_machine_learning.algorithms.classifiers import VQC

import qbiocode.utils.qutils as qutils

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import modeleval
from qbiocode.learning._tuning import (
    build_search_space,
    record_tuned_params,
    run_function_study,
)


def compute_vqc(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="VQC",
    data_key="",
    local_optimizer: Literal["COBYLA", "L_BFGS_B", "GradientDescent"] = "COBYLA",
    maxiter=100,
    encoding="Z",
    entanglement="linear",
    reps=2,
    primitive="sampler",
    ansatz_type="amp",
):
    """
    This function computes a Variational Quantum Classifier (VQC) using the Qiskit Machine Learning library.
    It takes training and testing datasets, along with various parameters to configure the VQC model.
    It initializes the quantum feature map, sets up the backend and session, and fits the VQC model to the training data.
    It then predicts the labels for the test data and evaluates the model's performance.
    The function returns the performance results, including accuracy, F1-score, AUC, runtime, as well as model parameters, and other relevant metrics.

    Args:
        X_train (array-like): Training feature set.
        X_test (array-like): Testing feature set.
        y_train (array-like): Training labels.
        y_test (array-like): Testing labels.
        args (dict): Dictionary containing configuration parameters for the VQC.
        verbose (bool, optional): If True, prints additional information. Defaults to False.
        model (str, optional): Model type. Defaults to 'VQC'.
        data_key (str, optional): Key for the dataset. Defaults to ''.
        local_optimizer (str, optional): Local optimizer to use. Defaults to 'COBYLA'.
        maxiter (int, optional): Maximum number of iterations for the optimizer. Defaults to 100.
        encoding (str, optional): Feature map encoding type. Defaults to 'Z'.
        entanglement (str, optional): Entanglement strategy. Defaults to 'linear'.
        reps (int, optional): Number of repetitions for the feature map and ansatz. Defaults to 2.
        primitive (str, optional): Primitive type ('sampler' or 'estimator'). Defaults to 'sampler'.
        ansatz_type (str, optional): Type of ansatz to use. Defaults to 'amp'.
    Returns:
        dict: Evaluation results including accuracy, time taken, and model parameters.
    """
    beg_time = time.time()
    # choose a method for mapping your features onto the circuit
    feature_map, _ = qutils.get_feature_map(
        feature_map=encoding, feat_dimension=X_train.shape[1], reps=reps, entanglement=entanglement
    )

    # get ansatz
    ansatz = qutils.get_ansatz(
        ansatz_type=ansatz_type,
        feat_dimension=feature_map.num_qubits,
        reps=reps,
        entanglement=entanglement,
    )

    #  Generate the backend, session and primitive
    backend, session, prim = qutils.get_backend_session(
        args, primitive, num_qubits=feature_map.num_qubits
    )

    # Get Optimizer
    optimizer = qutils.get_optimizer(local_optimizer, max_iter=maxiter)

    # instantiate the primitive
    if "simulator" == args["backend"]:
        vqc = VQC(sampler=prim, feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
    else:
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        vqc = VQC(
            sampler=prim,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            pass_manager=pm,
        )

    print(f"Currently running a variational quantum classifer (VQC) on this dataset.")
    print(f"The number of qubits in your circuit is: {feature_map.num_qubits}")
    print(f"The number of parameters in your circuit is: {feature_map.num_parameters}")

    # fit classifier to data
    model_fit = vqc.fit(X_train, y_train)
    hyperparameters = {
        "feature_map": feature_map.__class__.__name__,
        "ansatz": ansatz.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "optimizer_params": optimizer.settings,
        # Add other hyperparameters as needed
    }
    model_params = hyperparameters
    y_predicted = vqc.predict(X_test)

    if not isinstance(session, type(None)):
        session.close()

    return modeleval(
        y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose
    )


def compute_vqc_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="VQC",
    data_key="",
    local_optimizer=None,
    maxiter=None,
    encoding=None,
    entanglement=None,
    reps=None,
    primitive=None,
    ansatz_type=None,
    *,
    n_trials=10,
    validation_split=0.25,
):
    """Tune VQC's hyperparameters with Optuna, then run it at the best ones found.

    The quantum counterpart of the classical ``compute_*_opt`` functions, and driven by
    the same ``gridsearch_vqc_args`` config block -- a list is a choice, a
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
        model (str): Name of the model being used, default is 'VQC'.
        data_key (str): Key for identifying the dataset.
        local_optimizer (list or dict): Optimizers to search ('COBYLA', 'L_BFGS_B', 'GradientDescent'). None leaves it at the default.
        maxiter (list or dict): Optimizer iteration budgets to search. None leaves it at the default.
        encoding (list or dict): Feature-map values to search ('Z', 'ZZ', 'P'). None leaves it at the default.
        entanglement (list or dict): Entanglement patterns to search ('linear', 'full', ...). None leaves it at the default.
        reps (list or dict): Feature-map repetition counts to search. None leaves it at the default.
        primitive (list or dict): Qiskit primitives to search ('sampler', 'estimator'). None leaves it at the default.
        ansatz_type (list or dict): Ansatz types to search ('amp', ...). None leaves it at the default.
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

    candidates = {
        "local_optimizer": local_optimizer,
        "maxiter": maxiter,
        "encoding": encoding,
        "entanglement": entanglement,
        "reps": reps,
        "primitive": primitive,
        "ansatz_type": ansatz_type,
    }

    best_params = run_function_study(
        compute_vqc,
        build_search_space("vqc", candidates),
        X_train,
        y_train,
        args,
        model="vqc",
        n_trials=n_trials,
        seed=args.get("seed") if isinstance(args, dict) else None,
        validation_split=validation_split,
    )

    frame = compute_vqc(
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
