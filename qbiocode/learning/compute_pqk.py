# Copyright 2026, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ====== Base class imports ======
import hashlib
import os
import time
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# from qiskit.primitives import Sampler

# ====== Qiskit imports ======
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_ibm_runtime.exceptions import IBMRuntimeError, RuntimeJobFailureError
from sklearn import svm

import qbiocode.utils.qutils as qutils

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import extract_binary_scores, modeleval
from qbiocode.learning._tuning import (
    build_search_space,
    record_tuned_params,
    run_function_study,
)


def compute_pqk(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    # Lower case, matching the dispatch key. This default was "PQK", which was invisible
    # while the body hardcoded its label -- but now that the label is honoured, a direct
    # call with no model= would otherwise file results under a name no config can name.
    # qc_winner_finder's quantum list was also written against the upper-case spelling
    # while every real results table carries the lower-case one.
    model="pqk",
    data_key="",
    verbose=False,
    encoding="Z",
    primitive="estimator",
    entanglement="linear",
    reps=2,
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
        model (str): Model type, default is 'PQK'.
        data_key (str): Key for the dataset, default is ''.
        verbose (bool): If True, print additional information, default is False.
        encoding (str): Encoding method for the quantum circuit, default is 'Z'.
        primitive (str): Primitive type to use, default is 'estimator'.
        entanglement (str): Entanglement strategy, default is 'linear'.
        reps (int): Number of repetitions for the feature map, default is 2.

    Returns:
        modeleval (pd.DataFrame): A DataFrame containing evaluation metrics and model parameters for all models.

    Raises:
        ValueError: if any argument is outside its accepted set or the train/test
            arrays are inconsistent. Every check runs before any directory is
            created or any cached projection is read, so a mistyped ``encoding``
            costs nothing and reports the parameter, the value received and the
            accepted set instead of failing later inside qiskit.
    """
    # --- boundary validation -------------------------------------------------
    # These used to surface far from their cause: a bad `encoding` reached
    # qutils.get_feature_map and came back as the input string, failing with
    # "'str' object has no attribute 'num_qubits'"; a bad `entanglement` reached
    # qiskit and came back as "Something went wrong in Rust space".
    if not isinstance(args, Mapping):
        raise ValueError(
            f"args must be a mapping of run configuration (it is read with "
            f"args['backend'] and args.get(...)); got {type(args).__name__}."
        )
    if "backend" not in args:
        raise ValueError(
            "args is missing the required 'backend' key (e.g. 'simulator', or an "
            f"IBM Quantum backend name). Keys present: {sorted(args)}."
        )
    if encoding not in qutils.SUPPORTED_FEATURE_MAPS:
        raise ValueError(
            f"encoding must be one of {qutils.SUPPORTED_FEATURE_MAPS} "
            f"(case-sensitive); got {encoding!r}."
        )
    if isinstance(entanglement, str) and entanglement not in qutils.SUPPORTED_ENTANGLEMENTS:
        raise ValueError(
            f"entanglement must be one of {qutils.SUPPORTED_ENTANGLEMENTS}; "
            f"got {entanglement!r}."
        )
    if not isinstance(reps, (int, np.integer)) or reps < 1:
        raise ValueError(
            f"reps is the number of feature-map repetitions and must be a "
            f"positive integer; got {reps!r}."
        )
    if primitive != "estimator":
        # PQK projects onto Pauli expectation values, which is an Estimator
        # measurement; the backend below is requested as "estimator"
        # unconditionally. Accepting 'sampler' therefore changed only the cache
        # fingerprint, not the computation -- two cache files holding identical
        # projections, and a caller who believed they had measured something else.
        raise ValueError(
            f"primitive must be 'estimator'; got {primitive!r}. Projected quantum "
            f"kernels are built from Pauli expectation values, which only the "
            f"Estimator primitive provides. For sampler-based models see "
            f"compute_vqc or compute_qnn."
        )
    if not isinstance(data_key, str):
        raise ValueError(
            f"data_key is interpolated into the projection cache filename and "
            f"must be a string; got {type(data_key).__name__} ({data_key!r})."
        )

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            f"X_train and X_test must be 2-D (n_samples, n_features); got "
            f"{X_train.ndim}-D and {X_test.ndim}-D. Reshape a single sample with "
            f"X.reshape(1, -1)."
        )
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"X_train and X_test must have the same number of features -- one "
            f"feature map is built for both -- got {X_train.shape[1]} and "
            f"{X_test.shape[1]}."
        )
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError(
            f"X_train and X_test must both be non-empty; got "
            f"{X_train.shape[0]} training and {X_test.shape[0]} test samples."
        )
    if len(y_train) != X_train.shape[0] or len(y_test) != X_test.shape[0]:
        raise ValueError(
            f"Labels and features must be aligned; got {X_train.shape[0]} train "
            f"samples vs {len(y_train)} train labels, and {X_test.shape[0]} test "
            f"samples vs {len(y_test)} test labels."
        )
    # ------------------------------------------------------------------------

    classical_models = ["svc"]

    beg_time = time.time()
    feat_dimension = X_train.shape[1]

    projection_dir = os.path.expanduser(args.get("pqk_projection_dir", "pqk_projections"))
    if not os.path.exists(projection_dir):
        os.makedirs(projection_dir)

    # The cached projections are only valid for the exact feature map that produced them, so the
    # feature-map parameters must be part of the cache key. Without this, changing `encoding`,
    # `entanglement`, `reps` or `primitive` and rerunning into the same pqk_projection_dir
    # silently reloads the previous run's projections and reports them as the new result.
    # A short digest keeps the filename bounded regardless of how many parameters are added.
    feature_map_fingerprint = hashlib.sha256(
        repr(
            (
                ("model", model),
                ("encoding", encoding),
                ("primitive", primitive),
                ("entanglement", entanglement),
                ("reps", int(reps)),
            )
        ).encode("utf-8")
    ).hexdigest()[:10]

    file_projection_train = os.path.join(
        projection_dir,
        "pqk_projection_" + data_key + "_" + feature_map_fingerprint + "_train.npy",
    )
    file_projection_test = os.path.join(
        projection_dir,
        "pqk_projection_" + data_key + "_" + feature_map_fingerprint + "_test.npy",
    )
    checkpoint_dir = os.path.join(projection_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_every = int(args.get("pqk_checkpoint_every", 1))
    session_chunk_size = int(args.get("pqk_session_chunk_size", 100))
    max_runtime_retries = int(args.get("pqk_runtime_max_retries", 3))

    def _checkpoint_path(final_path):
        base = os.path.basename(final_path)
        return os.path.join(checkpoint_dir, base.replace(".npy", ".partial.npy"))

    def _save_projection_array(path, projections):
        tmp_path = path + ".tmp.npy"
        np.save(tmp_path, np.asarray(projections))
        os.replace(tmp_path, path)

    def _load_checkpoint(path, expected_len):
        if not os.path.exists(path):
            return []
        projections = np.load(path, allow_pickle=False)
        if len(projections) > expected_len:
            raise ValueError(
                f"Checkpoint {path} has {len(projections)} rows, "
                f"but the dataset only has {expected_len} rows."
            )
        return list(projections)

    def _validate_projection_file(path, expected_len):
        if not os.path.exists(path):
            return
        projections = np.load(path, allow_pickle=False)
        if len(projections) != expected_len:
            raise ValueError(
                f"Projection file {path} has {len(projections)} rows, "
                f"but the current dataset expects {expected_len} rows. "
                "Remove this projection file or use a different pqk_projection_dir."
            )
        # Each row holds one expectation value per Pauli-X/Y/Z observable per qubit, so the
        # flattened width must be 3 * feat_dimension. A mismatch means the file was written by a
        # run with a different feature dimension and must not be silently reused.
        expected_width = 3 * feat_dimension
        actual_width = int(np.prod(np.asarray(projections).shape[1:])) if len(projections) else 0
        if len(projections) and actual_width != expected_width:
            raise ValueError(
                f"Projection file {path} has {actual_width} features per row, "
                f"but the current feature map produces {expected_width} "
                f"(3 observables x {feat_dimension} qubits). "
                "Remove this projection file or use a different pqk_projection_dir."
            )

    def _is_closed_session_error(exc):
        return isinstance(exc, IBMRuntimeError) and (
            "Session has been closed" in str(exc) or '"code":1217' in str(exc)
        )

    def _is_retryable_runtime_error(exc):
        return isinstance(exc, RuntimeJobFailureError) and (
            "Temporary Internal Error" in str(exc) or "Error code 9707" in str(exc)
        )

    def _close_session(session):
        if not isinstance(session, type(None)):
            session.close()

    def _refresh_runtime(session):
        _close_session(session)
        _, new_session, new_prim = qutils.get_backend_session(
            args, "estimator", num_qubits=num_qubits
        )
        return new_session, new_prim

    # Shared with qbiocode.embeddings.embed.pqk -- see
    # qutils.unit_coefficient_data_map for why the symbolic case must not be
    # narrowed to a float.
    data_map_func = qutils.unit_coefficient_data_map

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

    _validate_projection_file(file_projection_train, len(X_train))
    _validate_projection_file(file_projection_test, len(X_test))

    if (not os.path.exists(file_projection_train)) | (not os.path.exists(file_projection_test)):

        #  Generate the backend, session and primitive
        backend, session, prim = qutils.get_backend_session(
            args, "estimator", num_qubits=num_qubits
        )
        try:

            # Transpile
            if args["backend"] != "simulator":
                circuit = qutils.transpile_circuit(
                    circuit, opt_level=3, backend=backend, PT=True, initial_layout=None
                )

            # Set the global phase to 0 to avoid header size issues
            circuit.global_phase = 0
        
            for f_tr, dat in [
                (file_projection_train, X_train.copy()),
                (file_projection_test, X_test.copy()),
            ]:
                if not os.path.exists(f_tr):
                    projections = []

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

                    checkpoint_file = _checkpoint_path(f_tr)
                    projections = _load_checkpoint(checkpoint_file, len(dat))
                    if projections:
                        print(
                            f"Resuming {os.path.basename(f_tr)} from "
                            f"datapoint {len(projections)}"
                        )

                    datapoints_in_session = 0
                    for i in range(len(projections), len(dat)):
                        if i % 100 == 0:
                            print(f"at datapoint {str(i)}")
                        if (
                            session is not None
                            and session_chunk_size > 0
                            and datapoints_in_session >= session_chunk_size
                        ):
                            session, prim = _refresh_runtime(session)
                            datapoints_in_session = 0

                        # Get training sample
                        parameters = dat[i]

                        # We define the primitive unified blocs (PUBs) consisting of the embedding circuit,
                        # set of observables and the circuit parameters
                        pub_x = (circuit, observables_x, parameters)
                        pub_y = (circuit, observables_y, parameters)
                        pub_z = (circuit, observables_z, parameters)

                        retry_count = 0
                        while True:
                            try:
                                job = prim.run([pub_x, pub_y, pub_z])
                                job_result = job.result()
                                job_result_x = job_result[0].data.evs
                                job_result_y = job_result[1].data.evs
                                job_result_z = job_result[2].data.evs
                                break
                            except Exception as exc:
                                _save_projection_array(checkpoint_file, projections)
                                if session is not None and _is_closed_session_error(exc):
                                    session, prim = _refresh_runtime(session)
                                    datapoints_in_session = 0
                                    continue
                                if (
                                    session is not None
                                    and _is_retryable_runtime_error(exc)
                                    and retry_count < max_runtime_retries
                                ):
                                    retry_count += 1
                                    print(
                                        f"Retrying datapoint {i} after temporary runtime "
                                        f"failure ({retry_count}/{max_runtime_retries})"
                                    )
                                    session, prim = _refresh_runtime(session)
                                    datapoints_in_session = 0
                                    continue
                                raise

                        # Record <X>, <Y> and <Z> on all qubits for the current datapoint
                        projections.append([job_result_x, job_result_y, job_result_z])
                        datapoints_in_session += 1
                        if checkpoint_every > 0 and len(projections) % checkpoint_every == 0:
                            _save_projection_array(checkpoint_file, projections)

                    _save_projection_array(f_tr, projections)
                    if os.path.exists(checkpoint_file):
                        os.remove(checkpoint_file)

        finally:
            if not isinstance(session, type(None)):
                session.close()

    # Load computed projections
    projections_train = np.load(file_projection_train)
    projections_train = np.array(projections_train).reshape(len(projections_train), -1)
    projections_test = np.load(file_projection_test)
    projections_test = np.array(projections_test).reshape(len(projections_test), -1)

    # `estimator`, not `model`. This assignment used to be `model = create_svc_model(...)`,
    # which overwrote the `model` PARAMETER -- the label this function was told to file its
    # results under -- with the fitted estimator object. The label was therefore gone
    # before it could be used, and `method_pqk = "pqk"` on the next line was the
    # workaround: a hardcoded label that ignored the argument. The visible consequence was
    # that `compute_pqk_opt` passing model="pqk_opt" had no effect, so a TUNED PQK run
    # produced `results_pqk` with model='pqk' -- byte-identical to an untuned one, leaving
    # no way to tell from ModelResults.csv whether a search had run.
    estimator = create_svc_model(args["seed"])

    method_pqk = model
    estimator.fit(projections_train, y_train)
    y_predicted = estimator.predict(projections_test)
    # `auc` is computed from these scores alone, never from y_predicted. The head is a
    # RandomizedSearchCV over SVC, which delegates to `best_estimator_`; `probability`
    # is not searched, so there is no predict_proba and extract_binary_scores falls
    # through to decision_function. Scored on the *projections*, which is the space this
    # estimator was fitted in -- the raw features would silently be the wrong width.
    y_score = extract_binary_scores(estimator, projections_test)

    hyperparameters = {
        "feature_map": feature_map.__class__.__name__,
        "feature_map_reps": reps,
        "entanglement": entanglement,
        "best_params": estimator.best_params_,
        # Add other hyperparameters as needed
    }
    model_params = hyperparameters

    return modeleval(
        y_test,
        y_predicted,
        beg_time,
        params=model_params,
        args=args,
        model=method_pqk,
        verbose=verbose,
        y_score=y_score,
    )





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


def compute_pqk_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    # '_opt', so a DIRECT call is self-describing. model_run always passes
    # model='pqk_opt' explicitly, but a caller using the default would otherwise
    # produce a row labelled as untuned -- and modeleval infers `tuned` from this
    # very string, so the label and the parameter column would BOTH be wrong.
    model="pqk_opt",
    data_key="",
    encoding=None,
    primitive=None,
    entanglement=None,
    reps=None,
    *,
    n_trials=10,
    validation_split=0.25,
):
    """Tune PQK's hyperparameters with Optuna, then run it at the best ones found.

    The quantum counterpart of the classical ``compute_*_opt`` functions, and driven by
    the same ``gridsearch_pqk_args`` config block -- a list is a choice, a
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
        model (str): Name of the model being used, default is 'PQK'.
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
        compute_pqk,
        build_search_space("pqk", candidates),
        X_train,
        y_train,
        args,
        model="pqk",
        n_trials=n_trials,
        seed=args.get("seed") if isinstance(args, dict) else None,
        validation_split=validation_split,
    )

    frame = compute_pqk(
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
