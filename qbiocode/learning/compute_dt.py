# ====== Base class imports ======

import time

from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier

# ====== Additional local imports ======
from qbiocode.learning._grid import build_param_grid
from qbiocode.learning._tuning import build_search_space, run_study
from qbiocode.evaluation.model_evaluation import extract_binary_scores, modeleval

# ====== Scikit-learn imports ======


# ====== Begin functions ======


def compute_dt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="Decision Tree",
    data_key="",
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None,
):
    """This function generates a model using a Decision Tree (DT) Classifier method as implemented in
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`__.
    It takes in parameter arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed.
    The model is trained on the training dataset and validated on the test dataset.  The model is trained on the training dataset and validated on the test dataset.
    The function returns the evaluation of the model on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Additional arguments, typically from config.yaml.
        verbose (bool): If True, prints additional information during execution.
        model (str): Name of the model being used, default is 'Decision Tree'.
        data_key (str): Key for the dataset, if applicable.
        criterion (str): The function to measure the quality of a split. Default is 'gini'.
        splitter (str): The strategy used to choose the split at each node. Default is 'best'.
        max_depth (int or None): The maximum depth of the tree. Default is None.
        min_samples_split (int): The minimum number of samples required to split an internal node. Default is 2.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node. Default is 1.
        min_weight_fraction_leaf (float): The minimum weighted fraction of the sum total of weights required to be at a leaf node. Default is 0.0.
        max_features (int, float, str or None): The number of features to consider when looking for the best split. Default is None.
        random_state (int or None): Controls the randomness of the estimator. Default is None.
        max_leaf_nodes (int or None): Grow a tree with max_leaf_nodes in best-first fashion. Default is None.
        min_impurity_decrease (float): A node will be split if this split induces a decrease of the impurity greater than or equal to this value. Default is 0.0.
        class_weight (dict or 'balanced' or None): Weights associated with classes in the form {class_label: weight}. Default is None.
        ccp_alpha (float): Complexity parameter used for Minimal Cost-Complexity Pruning. Default is 0.0.
        monotonic_cst: Monotonic constraints for tree nodes, if applicable. Default is None.
    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics, model parameters, and time taken for training and validation.
    """

    beg_time = time.time()
    dt = OneVsOneClassifier(
        DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst,
        )
    )
    # Fit the training datset
    model_fit = dt.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = dt.predict(X_test)
    # `auc` is a ranking metric and is computed from these scores alone -- passing
    # y_predicted, as this used to, silently reported balanced accuracy instead.
    # OneVsOneClassifier publishes no predict_proba, so what comes back here is its
    # decision_function; see extract_binary_scores for why that is a real ranking on a
    # binary target, and None (recorded as NaN) when it is not.
    y_score = extract_binary_scores(dt, X_test)
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


def compute_dt_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="Decision Tree",
    cv=5,
    criterion=None,
    max_depth=None,
    min_samples_split=None,
    min_samples_leaf=None,
    max_features=None,
    random_state=None,
    *,
    tuner="optuna",
    n_trials=50,
):
    """This function also generates a model using a Decision Tree (DT) Classifier method as implemented in
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`__.
    The difference here is that this function tunes the model's hyperparameters.
    The values or ranges searched for each parameter are specified in the config.yaml file,
    and ``tuner`` selects the search engine (Optuna by default). The
    combination of parameters that led to the best performance is saved and returned as best_params, which can then be used on similar
    datasets, without having to repeat the search.
    The model is trained on the training dataset and validated on the test dataset.  The model is trained on the training dataset and validated on the test dataset.
    The function returns the evaluation of the model on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model across the search.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Additional arguments, typically from config.yaml.
        verbose (bool): If True, prints additional information during execution.
        model (str): Name of the model being used, default is 'Decision Tree'.
        cv (int): Number of cross-validation folds. Default is 5.
        criterion (list): List of criteria to consider for splitting. Default is empty list.
        max_depth (list): List of maximum depths to consider. Default is empty list.
        min_samples_split (list): List of minimum samples required to split an internal node. Default is empty list.
        min_samples_leaf (list): List of minimum samples required to be at a leaf node. Default is empty list.
        max_features (list): List of maximum features to consider when looking for the best split. Default is empty list.
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
        modeleval (dict): A dictionary containing the evaluation metrics, best parameters, and time taken for training and validation.
    """

    beg_time = time.time()
    # Only the hyperparameters actually supplied. Passing all of them meant a
    # config that named a subset died in sklearn on the first one it left at its
    # `[]` default; see qbiocode.learning._grid.
    candidates = {
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
    }
    # Optuna by default; the exhaustive grid stays reachable so a number published
    # against it can still be reproduced. Both engines are handed the same
    # `candidates`, so switching `tuner` never changes *which* hyperparameters are
    # searched -- only how the search spends its fits.
    if tuner == "grid":
        search = GridSearchCV(
            DecisionTreeClassifier(random_state=random_state),
            param_grid=build_param_grid("dt", candidates),
            cv=cv,
        )
        search.fit(X_train, y_train)
        best_params = search.best_params_
    else:
        best_params = run_study(
            DecisionTreeClassifier,
            build_search_space("dt", candidates),
            X_train,
            y_train,
            cv=cv,
            n_trials=n_trials,
            seed=random_state,
            fixed={"random_state": random_state},
        )
    best_dt = DecisionTreeClassifier(**best_params, random_state=random_state)
    best_dt.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_dt.predict(X_test)
    # Fitted unwrapped, so predict_proba is available. `auc` is computed from these
    # scores alone; see extract_binary_scores.
    y_score = extract_binary_scores(best_dt, X_test)
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
