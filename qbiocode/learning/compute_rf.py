# ====== Base class imports ======

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# ====== Additional local imports ======
from qbiocode.learning._grid import build_param_grid
from qbiocode.learning._tuning import build_search_space, run_study
from qbiocode.evaluation.model_evaluation import extract_binary_scores, modeleval

# ====== Scikit-learn imports ======


# ====== Begin functions ======


def compute_rf(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="Random Forest",
    data_key="",
    n_estimators=100,
    *,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None,
):
    """
    This function generates a model using a Random Forest (RF) Classifier method as implemented in
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`__.
    It takes in parameter arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed.
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
        model (str): Name of the model being used, default is 'Random Forest'.
        data_key (str): Key for identifying the dataset, default is an empty string.
        n_estimators (int): Number of trees in the forest, default is 100.
        criterion (str): The function to measure the quality of a split, default is 'gini'.
        max_depth (int or None): Maximum depth of the tree, default is None.
        min_samples_split (int): Minimum number of samples required to split an internal node, default is 2.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node, default is 1.
        min_weight_fraction_leaf (float): Minimum weighted fraction of the sum total of weights required to be at a leaf node, default is 0.0.
        max_features (str or int or float): The number of features to consider when looking for the best split, default is 'sqrt'.
        max_leaf_nodes (int or None): Grow trees with max_leaf_nodes in best-first fashion, default is None.
        min_impurity_decrease (float): A node will be split if this split induces a decrease of the impurity greater than or equal to this value, default is 0.0.
        bootstrap (bool): Whether bootstrap samples are used when building trees, default is True.
        oob_score (bool): Whether to use out-of-bag samples to estimate the generalization accuracy, default is False.
        n_jobs (int or None): Number of jobs to run in parallel for both `fit` and `predict`, default is None.
        random_state (int or None): Controls the randomness of the estimator, default is None.
        warm_start (bool): When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, default is False.
        class_weight (dict or str or None): Weights associated with classes in the form {class_label: weight}, default is None.
        ccp_alpha (float): Complexity parameter used for Minimal
     Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model, including accuracy, AUC, F1 score, and the time taken for training and validation.

    """

    beg_time = time.time()
    rf = OneVsOneClassifier(
        RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
        )
    )
    # Fit the training datset
    model_fit = rf.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = rf.predict(X_test)
    # `auc` is a ranking metric and is computed from these scores alone -- passing
    # y_predicted, as this used to, silently reported balanced accuracy instead.
    # OneVsOneClassifier publishes no predict_proba, so what comes back here is its
    # decision_function; see extract_binary_scores for why that is a real ranking on a
    # binary target, and None (recorded as NaN) when it is not.
    y_score = extract_binary_scores(rf, X_test)
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


def compute_rf_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    cv=5,
    model="Random Forest",
    bootstrap=None,
    max_depth=None,
    max_features=None,
    min_samples_leaf=None,
    min_samples_split=None,
    n_estimators=None,
    random_state=None,
    *,
    tuner="optuna",
    n_trials=50,
):
    """
    This function also generates a model using a Random Forest (RF) Classifier method as implemented in
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`__.
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
        max_features (list): List of maximum features options for the search.
        min_samples_leaf (list): List of minimum samples leaf options for the search.
        min_samples_split (list): List of minimum samples split options for the search.
        n_estimators (list): List of number of estimators options for the search.
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

    """

    beg_time = time.time()
    # Only the hyperparameters actually supplied. Passing all of them meant a
    # config that named a subset died in sklearn on the first one it left at its
    # `[]` default; see qbiocode.learning._grid.
    candidates = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }

    # Optuna by default; the exhaustive grid stays reachable so a number published
    # against it can still be reproduced. Both engines are handed the same
    # `candidates`, so switching `tuner` never changes *which* hyperparameters are
    # searched -- only how the search spends its fits.
    if tuner == "grid":
        search = GridSearchCV(
            RandomForestClassifier(random_state=random_state),
            param_grid=build_param_grid("rf", candidates),
            cv=cv,
        )
        search.fit(X_train, y_train)
        best_params = search.best_params_
    else:
        best_params = run_study(
            RandomForestClassifier,
            build_search_space("rf", candidates),
            X_train,
            y_train,
            cv=cv,
            n_trials=n_trials,
            seed=random_state,
            fixed={"random_state": random_state},
        )
    best_rf = RandomForestClassifier(**best_params, random_state=random_state)
    best_rf.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_rf.predict(X_test)
    # Fitted unwrapped, so predict_proba is available. `auc` is computed from these
    # scores alone; see extract_binary_scores.
    y_score = extract_binary_scores(best_rf, X_test)
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
