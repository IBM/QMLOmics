# ====== Base class imports ======
import hashlib
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Deliberately broad: a missing xgboost raises ImportError, but an xgboost whose
# native library cannot load raises OSError (no libomp on macOS) or
# xgboost.core.XGBoostError, which subclasses ValueError -- narrowing to
# ImportError here would let those escape as an unhandled error at import time.
# The reason is kept so the messages below can quote the actual failure instead
# of guessing at it.
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
    _XGBOOST_ERROR = None
except Exception as exc:
    XGBOOST_AVAILABLE = False
    _XGBOOST_ERROR = str(exc)
    XGBClassifier = None  # type: ignore

# Same broad guard, same reason: catboost's failure mode when its native extension
# cannot load is an OSError rather than an ImportError.
try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
    _CATBOOST_ERROR = None
except Exception as exc:  # noqa: BLE001 -- see above
    CATBOOST_AVAILABLE = False
    _CATBOOST_ERROR = str(exc)
    CatBoostClassifier = None  # type: ignore

# from qiskit.primitives import Sampler
from functools import reduce

# ====== Qiskit imports ======
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from sklearn import svm
from sklearn.model_selection import GridSearchCV

import qbiocode.utils.qutils as qutils

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import modeleval
from qbiocode.learning._tuning import (
    build_search_space,
    record_tuned_params,
    run_function_study,
)

# Imported for its availability probe and lazy loader rather than for the estimator
# itself: `tabpfn_is_available` uses importlib.util.find_spec, so asking whether the
# optional extra is present costs neither the tabpfn import nor torch's OpenMP
# runtime. See qbiocode.learning.compute_tabpfn.
from qbiocode.learning.compute_tabpfn import (
    TABPFN_MAX_CLASSES,
    _explain_weight_access_failure,
    _load_tabpfn_classifier,
    tabpfn_is_available,
)


def compute_qpl(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    model="QPL",
    data_key="",
    verbose=False,
    encoding="Z",
    primitive="estimator",
    entanglement="linear",
    reps=2,
    classical_models=None,
):
    """
    This function generates quantum circuits, computes projections of the data onto these circuits,
    and evaluates the performance of classical machine learning models on the projected data.
    It uses a feature map to encode the data into quantum states and then measures the expectation values
    of Pauli operators to obtain the features. The classical models are trained on the projected training data and
    evaluated on the projected test data. The function returns evaluation metrics and model parameters.
    This function requires a quantum backend (simulator or real quantum hardware) for execution.
    It supports various configurations such as encoding methods, entanglement strategies, and repetitions
    of the feature map. The results are saved to files for training and test projections, which are reused
    if they already exist to avoid redundant computations.
    This function is part of the main quantum machine learning pipeline (QProfiler.py) and is intended for use in supervised learning tasks.
    It leverages quantum computing to enhance feature extraction and classification performance on complex datasets.
    The function returns the performance results, including accuracy, F1-score, AUC, runtime, as well as model parameters, and other relevant metrics.

    Args:
        X_train (np.ndarray): Training data features.
        X_test (np.ndarray): Test data features.
        y_train (np.ndarray): Training data labels.
        y_test (np.ndarray): Test data labels.
        args (dict): Arguments containing backend and other configurations.
        model (str): Model type, default is 'QPL'.
        data_key (str): Key for the dataset, default is ''.
        verbose (bool): If True, print additional information, default is False.
        encoding (str): Encoding method for the quantum circuit, default is 'Z'.
        primitive (str): Primitive type to use, default is 'estimator'.
        entanglement (str): Entanglement strategy, default is 'linear'.
        reps (int): Number of repetitions for the feature map, default is 2.
        classical_models (list): List of classical models to train on quantum projections.
            Defaults to ``['rf', 'mlp', 'svc', 'lr', 'xgb', 'catboost']``. ``'tabpfn'`` is
            also accepted but is deliberately absent from the default: it needs the
            optional ``[tabpfn]`` extra, so defaulting it on would make every QPL run warn
            in an ordinary install. Name it explicitly to use it. It needs no API token:
            QBioCode pins the ungated ``v2`` weights.
                                 Options: 'rf', 'mlp', 'svc', 'lr', 'xgb'.
                                 Default is ['rf', 'mlp', 'svc', 'lr', 'xgb'].

    Returns:
        modeleval (pd.DataFrame): A DataFrame containing evaluation metrics and model parameters for all models.
    """

    # Set default classical models if not provided
    if classical_models is None:
        classical_models = ["rf", "mlp", "svc", "lr", "xgb", "catboost"]

    beg_time = time.time()
    feat_dimension = X_train.shape[1]

    # The projection cache used to be keyed on `data_key` alone, in a hardcoded
    # "qpl_projections" directory. Two consequences, both silent:
    #
    #   * Changing `encoding`, `entanglement`, `reps` or `primitive` and rerunning reused
    #     the projection computed for the *previous* settings, so the new circuit was
    #     never run and the reported result described the old one. Tuning made this acute
    #     -- every trial after the first would have scored the same cached projection, so
    #     the search would have compared a hyperparameter against itself.
    #   * A different split of the same dataset (an inner validation split, say) matched
    #     the same file name and was loaded at the wrong length, surfacing downstream as
    #     `ValueError: Found input variables with inconsistent numbers of samples`, which
    #     names neither the cache nor the file.
    #
    # Both are fixed the way compute_pqk already handles it: fingerprint the settings
    # that change the circuit into the file name, validate the row count on load, and let
    # the directory be redirected so throwaway projections stay out of the real cache.
    projection_dir = os.path.expanduser(args.get("qpl_projection_dir", "qpl_projections"))
    os.makedirs(projection_dir, exist_ok=True)

    feature_map_fingerprint = hashlib.sha256(
        repr((encoding, entanglement, reps, primitive, feat_dimension)).encode()
    ).hexdigest()[:10]

    file_projection_train = os.path.join(
        projection_dir,
        "qpl_projection_" + data_key + "_" + feature_map_fingerprint + "_train.npy",
    )
    file_projection_test = os.path.join(
        projection_dir,
        "qpl_projection_" + data_key + "_" + feature_map_fingerprint + "_test.npy",
    )

    def _validate_projection_file(path, expected_len):
        """Refuse a cached projection whose row count does not match the current data."""
        if not os.path.exists(path):
            return
        cached = np.load(path, allow_pickle=False)
        if len(cached) != expected_len:
            raise ValueError(
                f"Projection file {path} has {len(cached)} rows, but the current dataset "
                f"expects {expected_len} rows. Remove this projection file or use a "
                f"different qpl_projection_dir."
            )

    _validate_projection_file(file_projection_train, len(X_train))
    _validate_projection_file(file_projection_test, len(X_test))

    #  This function ensures that all multiplicative factors of data features inside single qubit gates are 1.0
    def data_map_func(x: np.ndarray):
        """
        Define a function map from R^n to R.

        Args:
            x: data

        Returns:
            the mapped value (float or Parameter expression)
        """
        coeff = x[0] / 2 if len(x) == 1 else reduce(lambda m, n: (m * n) / 2, x)
        # Check if coeff is a numeric type before converting to float
        # If it's a Parameter expression, return it as-is for Qiskit to handle
        try:
            return float(coeff)
        except (TypeError, ValueError):
            # If conversion fails, it's likely a Parameter expression
            return coeff

    # choose a method for mapping your features onto the circuit
    feature_map, _ = qutils.get_feature_map(
        feature_map=encoding,
        feat_dimension=X_train.shape[1],
        reps=reps,
        entanglement=entanglement,
        data_map_func=data_map_func,
    )

    # Build quantum circuit
    circuit = QuantumCircuit(feature_map.num_qubits)
    circuit.compose(feature_map, inplace=True)
    num_qubits = circuit.num_qubits

    if (not os.path.exists(file_projection_train)) | (not os.path.exists(file_projection_test)):

        #  Generate the backend, session and primitive
        backend, session, prim = qutils.get_backend_session(
            args, "estimator", num_qubits=num_qubits
        )

        # Transpile
        if args["backend"] != "simulator":
            circuit = qutils.transpile_circuit(
                circuit, opt_level=3, backend=backend, PT=True, initial_layout=None
            )


        # Set the global phase to 0 to avoid header size issues
        circuit.global_phase = 0
        
        for f_tr in [file_projection_train, file_projection_test]:
            if not os.path.exists(f_tr):
                projections = []
                if "train" in f_tr:
                    dat = X_train.copy()
                else:
                    dat = X_test.copy()

                # Identity operator on all qubits
                id = "I" * feat_dimension

                # We group all commuting observables
                # These groups are the Pauli X, Y and Z operators on individual qubits
                # Apply the circuit layout to the observable if mapped to device
                if args["backend"] != "simulator":
                    observables_x = []
                    observables_y = []
                    observables_z = []
                    for i in range(feat_dimension):
                        observables_x.append(
                            Pauli(id[:i] + "X" + id[(i + 1) :]).apply_layout(
                                circuit.layout, num_qubits=backend.num_qubits
                            )
                        )
                        observables_y.append(
                            Pauli(id[:i] + "Y" + id[(i + 1) :]).apply_layout(
                                circuit.layout, num_qubits=backend.num_qubits
                            )
                        )
                        observables_z.append(
                            Pauli(id[:i] + "Z" + id[(i + 1) :]).apply_layout(
                                circuit.layout, num_qubits=backend.num_qubits
                            )
                        )
                else:
                    observables_x = [
                        Pauli(id[:i] + "X" + id[(i + 1) :]) for i in range(feat_dimension)
                    ]
                    observables_y = [
                        Pauli(id[:i] + "Y" + id[(i + 1) :]) for i in range(feat_dimension)
                    ]
                    observables_z = [
                        Pauli(id[:i] + "Z" + id[(i + 1) :]) for i in range(feat_dimension)
                    ]

                # projections[i][j][k] will be the expectation value of the j-th Pauli operator (0: X, 1: Y, 2: Z)
                # of datapoint i on qubit k
                projections = []

                for i in range(len(dat)):
                    if i % 100 == 0:
                        print(f"at datapoint {str(i)}")

                    # Get training sample
                    parameters = dat[i]

                    # We define the primitive unified blocs (PUBs) consisting of the embedding circuit,
                    # set of observables and the circuit parameters
                    pub_x = (circuit, observables_x, parameters)
                    pub_y = (circuit, observables_y, parameters)
                    pub_z = (circuit, observables_z, parameters)

                    job = prim.run([pub_x, pub_y, pub_z])
                    job_result_x = job.result()[0].data.evs
                    job_result_y = job.result()[1].data.evs
                    job_result_z = job.result()[2].data.evs

                    # Record <X>, <Y> and <Z> on all qubits for the current datapoint
                    projections.append([job_result_x, job_result_y, job_result_z])
                np.save(f_tr, projections)

        if not isinstance(session, type(None)):
            session.close()

    # Load computed projections
    projections_train = np.load(file_projection_train)
    projections_train = np.array(projections_train).reshape(len(projections_train), -1)
    projections_test = np.load(file_projection_test)
    projections_test = np.array(projections_test).reshape(len(projections_test), -1)

    # Check if XGBoost is requested but not available
    if "xgb" in classical_models and not XGBOOST_AVAILABLE:
        warnings.warn(
            "XGBoost is not properly installed or configured and will be skipped.\n"
            f"Error: {_XGBOOST_ERROR}\n"
            "On macOS, you may need to install OpenMP:\n"
            "  brew install libomp\n"
            "Then reinstall XGBoost:\n"
            "  pip install --force-reinstall xgboost\n"
            "See installation documentation for more details.\n"
            f"Continuing with other models: {[m for m in classical_models if m != 'xgb']}",
            UserWarning,
        )
        # Remove xgb from the list
        classical_models = [m for m in classical_models if m != "xgb"]

    # Same warn-and-drop treatment for catboost: one unusable head should cost that
    # head, not the whole quantum projection that has already been computed.
    if "catboost" in classical_models and not CATBOOST_AVAILABLE:
        warnings.warn(
            "CatBoost is not properly installed or configured and will be skipped.\n"
            f"Error: {_CATBOOST_ERROR}\n"
            "CatBoost is a core dependency, so this is a broken install; reinstall with:\n"
            "  pip install --force-reinstall catboost\n"
            f"Continuing with other models: {[m for m in classical_models if m != 'catboost']}",
            UserWarning,
        )
        classical_models = [m for m in classical_models if m != "catboost"]

    # TabPFN is an optional extra, so its absence is an ordinary configuration state
    # rather than a broken install -- the message says how to add it and moves on.
    if "tabpfn" in classical_models and not tabpfn_is_available():
        warnings.warn(
            "TabPFN is not installed and will be skipped as a QPL head.\n"
            'Install it with: pip install "qbiocode[tabpfn]"\n'
            f"Continuing with other models: {[m for m in classical_models if m != 'tabpfn']}",
            UserWarning,
        )
        classical_models = [m for m in classical_models if m != "tabpfn"]

    # TabPFN's pretrained head cannot represent more than ten classes, and unlike the
    # row and feature limits that one is not waivable. Checked here so an unsuitable
    # dataset drops the head with an explanation instead of failing the run.
    if "tabpfn" in classical_models:
        n_classes = len(np.unique(np.asarray(y_train)))
        if n_classes > TABPFN_MAX_CLASSES:
            warnings.warn(
                f"TabPFN supports at most {TABPFN_MAX_CLASSES} classes but this target has "
                f"{n_classes}, so it will be skipped as a QPL head.\n"
                f"Continuing with other models: "
                f"{[m for m in classical_models if m != 'tabpfn']}",
                UserWarning,
            )
            classical_models = [m for m in classical_models if m != "tabpfn"]

    # If no models remain after filtering, raise an error
    if not classical_models:
        raise ValueError(
            "No valid classical models specified. Please provide at least one model "
            "from: 'rf', 'mlp', 'svc', 'lr', 'xgb', 'catboost', 'tabpfn'"
        )

    model_res = []
    for method in classical_models:
        if method == "rf":
            model = create_rf_model(args["seed"])
        elif method == "svc":
            model = create_svc_model(args["seed"])
        elif method == "mlp":
            model = create_mlp_model(args["seed"])
        elif method == "lr":
            model = create_lr_model(args["seed"])
        elif method == "xgb":
            model = create_xgb_model(args["seed"])
        elif method == "catboost":
            model = create_catboost_model(args["seed"])
        elif method == "tabpfn":
            model = create_tabpfn_model(args["seed"])
        else:
            warnings.warn(
                f"Unknown model type '{method}' skipped. Valid options: 'rf', 'mlp', "
                f"'svc', 'lr', 'xgb', 'catboost', 'tabpfn'",
                UserWarning,
            )
            continue

        method_qpl = "qpl_" + method
        print(method_qpl)
        try:
            model.fit(projections_train, y_train)
            y_predicted = model.predict(projections_test)
        except Exception as error:  # noqa: BLE001 -- narrowed immediately below
            # Only a weights-unavailable failure is survivable here, and only TabPFN can
            # raise one: its checkpoint sits behind a license acceptance that cannot be
            # detected in advance, so unlike a missing extra it is not caught by the
            # availability filtering above. Dropping the head matches what this function
            # already does for an unusable xgboost or catboost -- and matters more here,
            # because by this point the quantum projection has been computed and paid
            # for. Anything else is a real failure and propagates.
            explained = _explain_weight_access_failure(error, method_qpl)
            if explained is None:
                raise
            warnings.warn(
                f"{method_qpl} could not run and was skipped.\n{explained}",
                UserWarning,
            )
            continue

        hyperparameters = {
            "feature_map": feature_map.__class__.__name__,
            "feature_map_reps": reps,
            "entanglement": entanglement,
            # Every other head is a RandomizedSearchCV and carries best_params_.
            # TabPFN is fitted bare -- see create_tabpfn_model for why -- so there is
            # no search result to report and its own settings are the honest answer.
            "best_params": getattr(model, "best_params_", None) or model.get_params(),
            # Add other hyperparameters as needed
        }
        model_params = hyperparameters

        model_res.append(
            modeleval(
                y_test, y_predicted, beg_time, model_params, args, model=method_qpl, verbose=verbose
            )
        )

    # Every head having been dropped leaves nothing to concatenate, and `pd.concat([])`
    # raises "No objects to concatenate" -- which says nothing about the heads or the
    # projection that produced them. Reachable now that a gated TabPFN is skipped rather
    # than fatal, and already reachable before via the unknown-model-name branch.
    if not model_res:
        raise ValueError(
            f"None of the requested classical models {classical_models} could be fitted "
            f"on the quantum projection, so there are no results to report. See the "
            f"warnings above for why each was skipped."
        )

    model_res = pd.concat(model_res)
    return model_res


def create_xgb_model(seed):
    # Initialize the XGBoost Classifier
    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "XGBoost is not properly installed or configured.\n"
            f"Error: {_XGBOOST_ERROR}\n\n"
            "On macOS, you may need to install OpenMP:\n"
            "  brew install libomp\n\n"
            "Then reinstall XGBoost:\n"
            "  pip install --force-reinstall xgboost\n\n"
            "See installation documentation for more details."
        )
    # random_state=seed, like every sibling create_*_model here: the search grid
    # below varies `subsample` and `colsample_bytree`, both of which sample rows and
    # columns at random, so an unseeded estimator made this model irreproducible even
    # though the search itself was seeded.
    xgb = XGBClassifier(  # type: ignore
        objective="binary:logistic", eval_metric="logloss", random_state=seed
    )

    xgb_param_distributions = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    # Initialize RandomizedSearchCV
    xgb_model = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=xgb_param_distributions,
        n_iter=40,
        cv=5,
        random_state=seed,
        n_jobs=-1,
    )

    return xgb_model


def create_catboost_model(seed):
    """A searched CatBoost head, matching how the other tree-based heads are built.

    Two CatBoost-specific points:

    * ``bootstrap_type`` is pinned to ``'Bernoulli'`` rather than left unset. The grid
      below varies ``subsample``, which CatBoost accepts under Bernoulli/MVS/Poisson
      but rejects under the Bayesian bootstrap -- and Bayesian is exactly what it
      defaults to once the target has more than two classes. Unpinned, this head would
      work on a binary projection and raise ``CatBoostError`` on a multiclass one.

    * ``allow_writing_files=False`` keeps every fit from dropping a ``catboost_info/``
      directory into the working directory; ``verbose=False`` silences the
      per-iteration training log. ``n_jobs=-1`` on the search below means many of these
      run at once, so both matter more here than in a single fit.
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError(
            "CatBoost is not properly installed or configured.\n"
            f"Error: {_CATBOOST_ERROR}\n\n"
            "CatBoost is a core QBioCode dependency, so this is a broken install. "
            "Reinstall it with:\n"
            "  pip install --force-reinstall catboost"
        )
    # random_state=seed for the same reason as create_xgb_model: the grid varies
    # `subsample` and `rsm`, both of which sample at random.
    catboost = CatBoostClassifier(  # type: ignore
        random_state=seed,
        bootstrap_type="Bernoulli",
        verbose=False,
        allow_writing_files=False,
    )

    catboost_param_distributions = {
        "iterations": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "depth": [3, 5, 7],
        "l2_leaf_reg": [1.0, 3.0, 9.0],
        "subsample": [0.7, 0.8, 1.0],
    }

    # Initialize RandomizedSearchCV
    catboost_model = RandomizedSearchCV(
        estimator=catboost,
        param_distributions=catboost_param_distributions,
        n_iter=40,
        cv=5,
        random_state=seed,
        n_jobs=-1,
    )

    return catboost_model


def create_tabpfn_model(seed):
    """A bare TabPFN head -- the only one here that is not wrapped in a search.

    Every sibling factory returns a ``RandomizedSearchCV`` because its estimator has
    training hyperparameters worth searching. TabPFN has none: the weights are
    pretrained and frozen, ``fit`` only memorises the training rows, and what it
    exposes are inference settings that move accuracy very little. Wrapping it as the
    others are would cost ``n_iter * cv`` transformer forward passes -- 200 at the
    settings used above -- per projection, per embedding, per split, to choose between
    near-identical candidates. Running it once at its defaults is both the honest
    configuration and the affordable one, which is rather the point of the model.

    The caller reads ``best_params_`` off the returned object; ``compute_qpl`` falls
    back to ``get_params()`` for exactly this head.

    Raises:
        ImportError: If the optional ``[tabpfn]`` extra is absent. ``compute_qpl``
            checks availability first and drops the head with a warning, so reaching
            this means the factory was called directly.
    """
    classifier_cls = _load_tabpfn_classifier()
    return classifier_cls(random_state=seed)


def create_lr_model(seed):
    # Initialize the Logistic Regression Classifier
    lr = LogisticRegression(random_state=seed, max_iter=1000)

    lr_param_distributions = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
    }

    # Initialize RandomizedSearchCV
    lr_model = RandomizedSearchCV(
        estimator=lr,
        param_distributions=lr_param_distributions,
        n_iter=40,
        cv=5,
        random_state=seed,
        n_jobs=-1,
    )

    return lr_model


def create_rf_model(seed):
    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(random_state=seed)

    rf_param_distributions = {
        "n_estimators": np.arange(100, 1000, 100),
        "max_depth": np.arange(5, 20),
        "min_samples_split": np.arange(2, 10),
        "min_samples_leaf": np.arange(1, 5),
        "bootstrap": [True, False],
    }

    # Initialize RandomizedSearchCV
    rf_model = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_param_distributions,
        n_iter=40,
        cv=5,
        random_state=seed,
        n_jobs=-1,
    )

    return rf_model


def create_mlp_model(seed):
    mlp_param_distributions = {
        "hidden_layer_sizes": [(128, 64, 32, 10), (64, 32, 10), (128, 64, 32)],
        "activation": ["identity", "logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        "alpha": [0.00005, 0.0005],
    }

    # Initialize the MLP Classifier
    mlp = MLPClassifier(random_state=seed)

    # Initialize RandomizedSearchCV
    mlp_model = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=mlp_param_distributions,
        n_iter=40,
        cv=5,
        random_state=seed,
        n_jobs=-1,
    )

    return mlp_model


def create_svc_model(seed):
    svc_param_distributions = {
        "C": [0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1],
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
    }

    # Initialize the SVC
    svc = SVC(random_state=seed)

    # Initialize RandomizedSearchCV
    svc_model = RandomizedSearchCV(
        estimator=svc,
        param_distributions=svc_param_distributions,
        n_iter=40,
        cv=5,
        random_state=seed,
        n_jobs=-1,
    )

    return svc_model

def compute_qpl_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="QPL",
    data_key="",
    encoding=None,
    primitive=None,
    entanglement=None,
    reps=None,
    *,
    n_trials=10,
    validation_split=0.25,
):
    """Tune QPL's hyperparameters with Optuna, then run it at the best ones found.

    The quantum counterpart of the classical ``compute_*_opt`` functions, and driven by
    the same ``gridsearch_qpl_args`` config block -- a list is a choice, a
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
        model (str): Name of the model being used, default is 'QPL'.
        data_key (str): Key for identifying the dataset.
        encoding (list or dict): Feature-map values to search ('Z', 'ZZ', 'P'). None leaves it at the default.
        primitive (list or dict): Qiskit primitives to search ('sampler', 'estimator'). None leaves it at the default.
        entanglement (list or dict): Entanglement patterns to search ('linear', 'full', ...). None leaves it at the default.
        reps (list or dict): Feature-map repetition counts to search. None leaves it at the default.
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
        "encoding": encoding,
        "primitive": primitive,
        "entanglement": entanglement,
        "reps": reps,
    }

    best_params = run_function_study(
        compute_qpl,
        build_search_space("qpl", candidates),
        X_train,
        y_train,
        args,
        model="qpl",
        n_trials=n_trials,
        seed=args.get("seed") if isinstance(args, dict) else None,
        validation_split=validation_split,
    )

    frame = compute_qpl(
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
