# ====== Base class imports ======
import inspect
import json
import logging
import os
import warnings

import pandas as pd

# ======= Parallelization =====
from joblib import Parallel, delayed

current_dir = os.getcwd()

logger = logging.getLogger(__name__)

#: Search engines ``args['tuner']`` may name. Both are driven by the same
#: ``gridsearch_<model>_args`` blocks; see :mod:`qbiocode.learning._tuning`.
_TUNERS = frozenset({"optuna", "grid"})


def _call_with_global_seeds(compute_fn, seed, q_seed, *fn_args, **fn_kwargs):
    """Re-establish the global RNG seeds inside the worker, then run ``compute_fn``.

    ``qprofiler`` sets ``np.random.seed`` and ``algorithm_globals.random_seed`` in
    the parent process, but the models run under joblib's loky backend, which
    starts fresh interpreters. Neither seed crosses that boundary, so anything
    reading a global RNG -- ``compute_qnn``'s initial weights come from
    ``algorithm_globals.random`` -- started from OS entropy and produced a
    different answer on every run.

    This is a floor, not the mechanism: ``_seeded_kwargs`` below sets
    ``random_state`` on each estimator explicitly, because joblib batches tasks
    and how far an earlier task advanced a shared global stream depends on
    timing. Seeding here covers the randomness that has no ``random_state`` to
    set.
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)
    if q_seed is not None:
        try:
            from qiskit_algorithms.utils import algorithm_globals
        except ImportError:
            # Classical-only install: nothing in this worker reads the quantum
            # global seed, so there is nothing to set.
            pass
        else:
            algorithm_globals.random_seed = q_seed
    return compute_fn(*fn_args, **fn_kwargs)


def model_run(X_train, X_test, y_train, y_test, data_key, args):
    """This function runs the ML methods, with or without a grid search, as specified in the config.yaml file.
    It returns a python dictionary contatining these results, which can then be parsed out. It is designed to run
    each of the ML methods in parallel, for each data set (this is done by calling the Parallel module in results below).
    The arguments X_train, X_test, y_train, y_test are all passed in from the main script (qmlbench.py) as the input
    datasets are processed, while the remaining arguments are passed from the config.yaml file.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        data_key (str): Key for the dataset being processed.
        args (dict): Dictionary containing configuration parameters, including:
            - model: List of models to run.
            - n_jobs: Number of parallel jobs to run.
            - grid_search: Boolean indicating whether to tune hyperparameters.
            - tuner: 'optuna' (default) or 'grid' -- which search to run when
              grid_search is on.
            - n_trials: Trial budget for the Optuna tuner, default 50.
            - cross_validation: Number of cross-validation folds, default 5.
            - gridsearch_<model>_args: Values or ranges to search for each model.
              'catboost' and 'tabpfn' use the same blocks as the other classical
              models; see qbiocode.learning.compute_catboost and .compute_tabpfn for
              the hyperparameters each accepts.
            - <model>_args: Additional arguments for each model.

    Returns:
        model_total_result (dict): A dictionary containing the results of the models run, with keys as model names and values as their respective results.
        This dictionary can readily be converted to a Pandas Dataframe, as seen in the 'ModelResults.csv' files that are produced in the results directory
        when the main profiler is run (qbiocode-profiler.py).

    """

    # Lazy imports to avoid circular dependency
    # These imports happen inside the function, not at module level
    from qbiocode.learning.compute_catboost import compute_catboost, compute_catboost_opt
    from qbiocode.learning.compute_dt import compute_dt, compute_dt_opt
    from qbiocode.learning.compute_lr import compute_lr, compute_lr_opt
    from qbiocode.learning.compute_rf import compute_rf, compute_rf_opt
    from qbiocode.learning.compute_mlp import compute_mlp, compute_mlp_opt
    from qbiocode.learning.compute_xgb import compute_xgb, compute_xgb_opt
    from qbiocode.learning.compute_pqk import compute_pqk, compute_pqk_opt
    from qbiocode.learning.compute_qpl import compute_qpl, compute_qpl_opt
    from qbiocode.learning.compute_qnn import compute_qnn, compute_qnn_opt
    from qbiocode.learning.compute_qsvc import compute_qsvc, compute_qsvc_opt
    from qbiocode.learning.compute_nb import compute_nb, compute_nb_opt
    from qbiocode.learning.compute_svc import compute_svc, compute_svc_opt
    from qbiocode.learning.compute_vqc import compute_vqc, compute_vqc_opt
    # TabPFN imports its own dependency lazily, so naming it here does not require
    # the optional [tabpfn] extra to be installed -- only *selecting* it does.
    from qbiocode.learning.compute_tabpfn import compute_tabpfn, compute_tabpfn_opt
    
    # Build model dictionary
    compute_ml_dict = {
        "svc_opt": compute_svc_opt,
        "svc": compute_svc,
        "dt_opt": compute_dt_opt,
        "dt": compute_dt,
        "lr_opt": compute_lr_opt,
        "lr": compute_lr,
        "nb_opt": compute_nb_opt,
        "nb": compute_nb,
        "rf_opt": compute_rf_opt,
        "rf": compute_rf,
        "xgb_opt": compute_xgb_opt,
        "xgb": compute_xgb,
        "catboost_opt": compute_catboost_opt,
        "catboost": compute_catboost,
        "tabpfn_opt": compute_tabpfn_opt,
        "tabpfn": compute_tabpfn,
        "mlp_opt": compute_mlp_opt,
        "mlp": compute_mlp,
        "qsvc": compute_qsvc,
        "qsvc_opt": compute_qsvc_opt,
        "vqc": compute_vqc,
        "vqc_opt": compute_vqc_opt,
        "qnn": compute_qnn,
        "qnn_opt": compute_qnn_opt,
        "pqk": compute_pqk,
        "pqk_opt": compute_pqk_opt,
        "qpl": compute_qpl,
        "qpl_opt": compute_qpl_opt,
    }

    quantum_models = {"qsvc", "qnn", "vqc", "pqk", "qpl"}

    # Quantum models now have `_opt` twins, but they stay off unless asked for twice:
    # `grid_search: True` alone tunes only the classical models, exactly as before. A
    # quantum fit builds an n-by-n fidelity kernel by circuit simulation, so turning
    # tuning on for a quantum model multiplies its cost by the trial budget -- which
    # would have made every existing config that names one dramatically slower on
    # upgrade, with no change on the user's part.
    tune_quantum = bool(args.get("tune_quantum", False))

    # Validate the requested models before dispatching. An unknown name otherwise
    # reached `compute_ml_dict[method]` inside a joblib worker and came back as a
    # bare KeyError with no indication of what the valid names are.
    requested = list(args["model"])
    if not requested:
        raise ValueError(
            "args['model'] is empty; there is nothing to run. Choose at least one "
            f"of {sorted(compute_ml_dict)}."
        )
    unknown = [m for m in requested if m not in compute_ml_dict]
    if unknown:
        raise ValueError(
            f"Unknown model(s) {unknown} in args['model']. Available models: "
            f"{sorted(compute_ml_dict)} (quantum: {sorted(quantum_models)}). "
            f"Note the '_opt' variants are selected with args['grid_search'], not "
            f"by naming them here."
        )
    if grid_search_requested := bool(args.get("grid_search", False)):
        missing_opt = [
            m for m in requested
            if m not in quantum_models and (m + "_opt") not in compute_ml_dict
        ]
        if missing_opt:
            raise ValueError(
                f"grid_search is enabled but {missing_opt} have no '_opt' "
                f"implementation. Disable grid_search or drop those models."
            )
        # A misspelt tuner would otherwise fall through to the `else` branch inside
        # every `_opt` function and run Optuna, so a config asking for the
        # exhaustive grid would silently not get it.
        missing_blocks = [
            m for m in requested
            if m in quantum_models
            and args.get("tune_quantum", False)
            and not args.get("gridsearch_" + m + "_args")
        ]
        if missing_blocks:
            raise ValueError(
                f"tune_quantum is enabled but {missing_blocks} have no "
                f"'gridsearch_<model>_args' block naming what to search, so there is "
                f"nothing to tune. Add one per model, or drop tune_quantum to run them "
                f"at their configured hyperparameters."
            )
        # There is no exhaustive-grid engine for the quantum models: their `_opt`
        # wrappers score a whole compute function, not an estimator GridSearchCV could
        # drive. Asking for `tuner: grid` and getting Optuna anyway is the kind of
        # silent substitution that makes a result impossible to interpret later.
        if args.get("tune_quantum", False) and args.get("tuner", "optuna") == "grid":
            quantum_requested = [m for m in requested if m in quantum_models]
            if quantum_requested:
                warnings.warn(
                    f"tuner: 'grid' applies to the classical models only. "
                    f"{quantum_requested} will still be tuned with Optuna -- a quantum "
                    f"candidate is scored by running the whole model, so there is no "
                    f"exhaustive-grid engine for them. Set tune_quantum: False to run "
                    f"them at their configured hyperparameters instead.",
                    UserWarning,
                    stacklevel=2,
                )
        tuner = args.get("tuner", "optuna")
        if tuner not in _TUNERS:
            raise ValueError(
                f"Unknown tuner {tuner!r} in args['tuner']. Choose one of "
                f"{sorted(_TUNERS)}: 'optuna' samples args['n_trials'] "
                f"configurations with Optuna, 'grid' fits every combination."
            )
    elif args.get("tune_quantum", False):
        raise ValueError(
            "tune_quantum is enabled but grid_search is not, so no tuning would run. "
            "Set grid_search: True as well, or drop tune_quantum."
        )
    del grid_search_requested

    # Run classical and quantum models
    n_jobs = len(args["model"])
    if "n_jobs" in args.keys():
        n_jobs = min(args["n_jobs"], len(args["model"]))

    grid_search = False
    if "grid_search" in args.keys():
        grid_search = args["grid_search"]

    # Check if any quantum models are in the model list when grid_search is enabled
    if grid_search:
        quantum_in_models = [m for m in args["model"] if m in quantum_models]
        if quantum_in_models and not tune_quantum:
            print("\n" + "=" * 80)
            print("NOTE: Hyperparameter tuning is enabled, but not for these quantum",
                  "models:", quantum_in_models)
            print("=" * 80)
            print("They will run at their configured hyperparameters. Quantum tuning is")
            print("off by default because each trial is a quantum fit: on the simulator a")
            print("single QSVC fit builds an n-by-n fidelity kernel, so a 10-trial study")
            print("costs roughly ten ordinary runs of that model.")
            print("\nTo tune them with Optuna, set both keys and give each model a")
            print("gridsearch_<model>_args block:")
            print("    grid_search: True")
            print("    tune_quantum: True")
            print("    n_trials_quantum: 10")
            print("\nTo sweep them exhaustively instead, generate one config per")
            print("combination and compare across runs:")
            print("  from qbiocode.utils import generate_qml_experiment_configs")
            print("  num_configs, _ = generate_qml_experiment_configs(")
            print("      template_config_path='configs/config.yaml',")
            print("      output_dir='configs/qml_gridsearch',")
            print("      data_dirs=['data/your_data_dir']")
            print("  )")
            print("\nSee documentation: qbiocode.utils.generate_qml_experiment_configs")
            print("=" * 80 + "\n")

    def _model_args(method):
        """Per-model hyperparameters from the config, or the estimator defaults.

        `args[method + "_args"]` raised KeyError for any model whose config block
        was absent -- which includes ``xgb`` and ``qpl`` in the shipped
        config.yaml, so naming either in ``model`` failed before the estimator was
        ever constructed. The grid-search branch below already used ``.get(...,
        {})``; this makes the two agree, and says so in the log rather than
        substituting silently.
        """
        key = method + "_args"
        if key in args:
            return args[key]
        logger.info(
            "No %r block in the config; running %r with its default "
            "hyperparameters.", key, method,
        )
        return {}

    def _seeded_kwargs(compute_fn, model_kwargs):
        """Fill in ``random_state`` from ``args['seed']`` wherever an estimator takes one.

        Two runs at the same seed used to disagree on the decision-tree rows.
        ``DecisionTreeClassifier`` at ``random_state=None`` permutes the features
        before choosing a split, so a tie between two equally-good splits broke
        one way or the other at random; on a 60-sample dataset that moved
        accuracy by a whole test sample (0.889 vs 0.944). The same applies to
        every other estimator here that draws from a global RNG: random forests,
        the MLP's weight init, XGBoost's row subsampling, SVC's probability
        calibration.

        A ``random_state`` already present in the config wins -- this only fills
        the gap. Functions that take no ``random_state`` (naive Bayes) are left
        alone.
        """
        seed = args.get("seed")
        if seed is None or "random_state" in model_kwargs:
            return model_kwargs
        try:
            takes_random_state = "random_state" in inspect.signature(compute_fn).parameters
        except (TypeError, ValueError):  # pragma: no cover - C callables
            return model_kwargs
        if not takes_random_state:
            return model_kwargs
        return {**model_kwargs, "random_state": seed}

    seed = args.get("seed")
    q_seed = args.get("q_seed")

    if grid_search:
        results = []
        for method in args["model"]:
            if method in quantum_models and tune_quantum:
                # Tuned like the classical models, from the same
                # `gridsearch_<model>_args` block, but on its own budget and without
                # `cv`/`tuner`: a quantum candidate is scored on one stratified holdout
                # rather than k folds, and there is no exhaustive-grid engine to select.
                compute_fn = compute_ml_dict[method + "_opt"]
                result = delayed(_call_with_global_seeds)(
                    compute_fn,
                    seed,
                    q_seed,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    args,
                    model=method + "_opt",
                    data_key=data_key,
                    n_trials=args.get("n_trials_quantum", 10),
                    validation_split=args.get("validation_split", 0.25),
                    **_seeded_kwargs(
                        compute_fn, args.get("gridsearch_" + method + "_args", {})
                    ),
                    verbose=False,
                )
            elif method in quantum_models:
                # Untuned: run at the configured hyperparameters, as before.
                compute_fn = compute_ml_dict[method]
                result = delayed(_call_with_global_seeds)(
                    compute_fn,
                    seed,
                    q_seed,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    args,
                    model=method,
                    data_key=data_key,
                    **_seeded_kwargs(compute_fn, args.get(method + "_args", {})),
                    verbose=False,
                )
            else:
                # Classical models have _opt versions with grid search
                compute_fn = compute_ml_dict[method + "_opt"]
                result = delayed(_call_with_global_seeds)(
                    compute_fn,
                    seed,
                    q_seed,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    args,
                    model=method + "_opt",
                    # `args["cross_validation"]` was an unguarded lookup on this
                    # branch only, so tuning a model from a config that omitted the
                    # key died here rather than at validation.
                    cv=args.get("cross_validation", 5),
                    tuner=args.get("tuner", "optuna"),
                    n_trials=args.get("n_trials", 50),
                    **_seeded_kwargs(
                        compute_fn, args.get("gridsearch_" + method + "_args", {})
                    ),
                    verbose=False,
                )
            results.append(result)
        results = Parallel(n_jobs=n_jobs)(results)
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_call_with_global_seeds)(
                compute_ml_dict[method],
                seed,
                q_seed,
                X_train,
                X_test,
                y_train,
                y_test,
                args,
                model=method,
                data_key=data_key,
                **_seeded_kwargs(compute_ml_dict[method], _model_args(method)),
                verbose=False,
            )
            for method in args["model"]
        )

    model_total_result = pd.melt(pd.concat(results)).dropna()  # type: ignore
    model_total_result["i"] = 0
    model_total_result = model_total_result.pivot(columns="variable", values="value", index="i")
    return model_total_result.to_dict()
