# ====== Base class imports ======

import time

import numpy as np

# ====== Scikit-learn imports ======

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
    _XGBOOST_ERROR = None
except Exception as e:
    # Catch all exceptions including XGBoostError, ImportError, OSError
    XGBOOST_AVAILABLE = False
    _XGBOOST_ERROR = str(e)
    XGBClassifier = None  # type: ignore

from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# ====== Additional local imports ======
from qbiocode.learning._grid import build_param_grid, warn_ignored_hyperparameter
from qbiocode.learning._tuning import build_search_space, run_study
from qbiocode.evaluation.model_evaluation import extract_binary_scores, modeleval

# ====== Begin functions ======


def compute_xgb(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="xgb",
    data_key="",
    n_estimators=100,
    *,
    criterion="gini",
    max_depth=None,
    subsample=0.5,
    learning_rate=0.5,
    colsample_bytree=1,
    min_child_weight=1,
    random_state=None,
):
    """
    This function generates a model using an Extreme Gradient Boositing (xgb) Classifier method as implemented in xgboost. It takes in parameter
    arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed.
    The model is trained on the training dataset and validated on the test dataset. The function returns the evaluation of the model
    on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Additional arguments, typically from a configuration file.
        verbose (bool): If True, prints additional information during execution.
        model (str): Name of the model being used, default is 'XGBoost'.
        data_key (str): Key for identifying the dataset, default is an empty string.
        n_estimators (int): Number of trees in the forest, default is 100.
        max_depth (int or None): Maximum depth of the tree, default is None.
        subsample (float) : Subsample ratio of the training instances. Default 0.5
        learning_rate (float): Step size shrinkage used in update to prevent overfitting. Default is 0.5
        colsample_bytree  (float): subsample ratio of columns when constructing each tree. Default is 1
        min_child_weight (int) : Minimum sum of instance weight (hessian) needed in a child. Default is 1
        random_state (int or None): Seed for the estimator's own randomness. QProfiler fills this in from the run's ``seed`` so two runs at one seed agree; None leaves the estimator drawing from the global RNG.
     Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model, including accuracy, AUC, F1 score, and the time taken for training and validation.

    Raises:
        ImportError: If XGBoost is not properly installed or configured.

    """

    if not XGBOOST_AVAILABLE:
        error_msg = (
            "XGBoost is not properly installed or configured.\n"
            f"Error: {_XGBOOST_ERROR}\n\n"
            "On macOS, you may need to install OpenMP:\n"
            "  brew install libomp\n\n"
            "Then reinstall XGBoost:\n"
            "  pip install --force-reinstall xgboost\n\n"
            "See installation documentation for more details."
        )
        raise ImportError(error_msg)

    beg_time = time.time()
    xgb = OneVsOneClassifier(
        XGBClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,  # type: ignore
            subsample=subsample,
            learning_rate=learning_rate,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            random_state=random_state,
        )
    )
    # Fit the training datset
    model_fit = xgb.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = xgb.predict(X_test)
    # `auc` is a ranking metric and is computed from these scores alone -- passing
    # y_predicted, as this used to, silently reported balanced accuracy instead.
    # OneVsOneClassifier publishes no predict_proba, so what comes back here is its
    # decision_function; see extract_binary_scores for why that is a real ranking on a
    # binary target, and None (recorded as NaN) when it is not.
    y_score = extract_binary_scores(xgb, X_test)
    return modeleval(
        y_test,
        y_predicted,
        beg_time,
        model_params,
        args,
        model=model,
        verbose=verbose,
        y_score=y_score,
    )


def compute_xgb_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    cv=5,
    model="xgb",
    bootstrap=None,
    max_depth=None,
    max_features=None,
    learning_rate=None,
    subsample=None,
    colsample_bytree=None,
    n_estimators=None,
    min_child_weight=None,
    random_state=None,
    *,
    tuner="optuna",
    n_trials=50,
):
    """
    This function generates a model using an Extreme Gradient Boositing (xgb) Classifier method as implemented in xgboost.
    The difference here is that this function tunes the model's hyperparameters.
    The values or ranges searched for each parameter are specified in the config.yaml file,
    and ``tuner`` selects the search engine (Optuna by default). The
    combination of parameters that led to the best performance is saved and returned as best_params, which can then be used on similar
    datasets, without having to repeat the search.
    The model is trained on the training dataset and validated on the test dataset. The function returns the evaluation of the model
    on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model across the search.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Additional arguments, typically from a configuration file.
        verbose (bool): If True, prints additional information during execution.
        cv (int): Number of cross-validation folds, default is 5.
        model (str): Name of the model being used, default is 'Random Forest'.
        bootstrap (list): List of bootstrap options for the search.
        max_depth (list): List of maximum depth options for the search.
        subsample (list): List of subsample ratio of the training instances options for the search.
        learning_rate (list): List of step size shrinkage used in update to prevent overfitting options for the search.
        colsample_bytree (list): List of subsample ratio of columns when constructing each tree options for the search.
        n_estimators (list): List of number of estimators options for the search.
        min_child_weight (list): List of minimum sum of instance weight (hessian) needed in a childoptions for the search.
        random_state (int or None): Seed for the estimator's own randomness. QProfiler fills this in from the run's ``seed`` so two runs at one seed agree; None leaves the estimator drawing from the global RNG.

        tuner (str): Which search to run. ``'optuna'`` (default) spends ``n_trials`` on
            Optuna's TPE sampler, which also allows a hyperparameter to be given as a
            ``{low, high}`` range rather than a list. ``'grid'`` restores the exhaustive
            ``GridSearchCV`` sweep over every combination.
        n_trials (int): Trial budget when ``tuner='optuna'``, default is 50. Lowered
            automatically when the configured values describe fewer distinct
            combinations than that, so a small block does not re-evaluate the same
            models.
    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model, including accuracy, AUC, F1 score, and the time taken for training and validation.

    Raises:
        ImportError: If XGBoost is not properly installed or configured.
    """

    if not XGBOOST_AVAILABLE:
        error_msg = (
            "XGBoost is not properly installed or configured.\n"
            f"Error: {_XGBOOST_ERROR}\n\n"
            "On macOS, you may need to install OpenMP:\n"
            "  brew install libomp\n\n"
            "Then reinstall XGBoost:\n"
            "  pip install --force-reinstall xgboost\n\n"
            "See installation documentation for more details."
        )
        raise ImportError(error_msg)

    beg_time = time.time()
    # XGBoost has no bootstrap parameter, but its sklearn wrapper accepts unknown
    # keyword arguments without complaint, so this was never an error -- just a
    # silently doubled search returning identical models.
    if bootstrap:
        warn_ignored_hyperparameter(
            "xgb", "bootstrap", "XGBoost does not implement -- it samples rows via 'subsample'."
        )

    # Only the hyperparameters actually supplied. Passing all of them meant a
    # config that named a subset died in sklearn on the first one it left at its
    # `[]` default; see qbiocode.learning._grid.
    candidates = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "bootstrap": bootstrap,
    }

    # Optuna by default; the exhaustive grid stays reachable so a number published
    # against it can still be reproduced. Both engines are handed the same
    # `candidates`, so switching `tuner` never changes *which* hyperparameters are
    # searched -- only how the search spends its fits.
    if tuner == "grid":
        search = GridSearchCV(
            XGBClassifier(random_state=random_state),
            param_grid=build_param_grid("xgb", candidates),
            cv=cv,
        )  # type: ignore
        search.fit(X_train, y_train)
        best_params = search.best_params_
    else:
        best_params = run_study(
            XGBClassifier,
            build_search_space("xgb", candidates),
            X_train,
            y_train,
            cv=cv,
            n_trials=n_trials,
            seed=random_state,
            fixed={"random_state": random_state},
        )
    best_xgb = XGBClassifier(**best_params, random_state=random_state)  # type: ignore
    best_xgb.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_xgb.predict(X_test)
    # Fitted unwrapped, so predict_proba is available. `auc` is computed from these
    # scores alone; see extract_binary_scores.
    y_score = extract_binary_scores(best_xgb, X_test)
    return modeleval(
        y_test,
        y_predicted,
        beg_time,
        best_params,
        args,
        model=model,
        verbose=verbose,
        y_score=y_score,
        # This function IS the tuned branch, so it states so rather than letting
        # modeleval infer it from the label: a DIRECT call leaves `model` at its
        # display-name default ('Decision Tree'), which carries no _opt marker.
        tuned=True,
    )
