# ====== Base class imports ======

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# ====== Additional local imports ======
from qbiocode.learning._grid import build_param_grid
from qbiocode.learning._tuning import build_search_space, run_study
from qbiocode.evaluation.model_evaluation import modeleval

# ====== Scikit-learn imports ======


# ====== Begin functions ======


def compute_lr(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    model="Logistic Regression",
    data_key="",
    penalty="l2",
    *,
    dual=False,
    tol=0.0001,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver="saga",
    max_iter=10000,
    multi_class="deprecated",
    verbose=False,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None,
):
    """This function generates a model using a Logistic Regression (LR) method as implemented in
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`__.
    It takes in parameter arguments specified in the config.yaml file, but will use the default parameters
    specified above if none are passed. The model is trained on the training dataset and validated on the
    test dataset. The function returns the evaluation of the model on the test dataset, including accuracy,
    AUC, F1 score, and the time taken to train and validate the model. This function is designed to be used
    in a supervised learning context, where the goal is to classify data points.

    Args:
        X_train (numpy.ndarray): Training data features.
        X_test (numpy.ndarray): Test data features.
        y_train (numpy.ndarray): Training data labels.
        y_test (numpy.ndarray): Test data labels.
        args (dict): Additional arguments, such as dataset name and other configurations.
        model (str): Name of the model being used, default is 'Logistic Regression'.
        data_key (str): Key for the dataset, default is an empty string.
        penalty (str): Regularization penalty, default is 'l2'.
        dual (bool): Dual formulation, default is False.
        tol (float): Tolerance for stopping criteria, default is 0.0001.
        C (float): Inverse of regularization strength, default is 1.0.
        fit_intercept (bool): Whether to fit the intercept, default is True.
        intercept_scaling (float): Scaling factor for the intercept, default is 1.
        class_weight (dict or None): Weights associated with classes, default is None.
        random_state (int or None): Random seed for reproducibility, default is None.
        solver (str): Algorithm to use in the optimization problem, default is 'saga'.
        max_iter (int): Maximum number of iterations for convergence, default is 10000.
        multi_class (str): Multi-class option, deprecated in this context.
        verbose (bool): Whether to print detailed logs, default is False.
        warm_start (bool): Whether to reuse the solution of the previous call to fit as initialization,
                           default is False.
        n_jobs (int or None): Number of jobs to run in parallel for both `fit` and `predict`,
                              default is None which means 1 unless in a joblib.parallel_backend context.
        l1_ratio (float or None): The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1.
                                  Only used if penalty='elasticnet', default is None.

    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics, model parameters, and time taken for training and validation.
    """

    beg_time = time.time()
    logres = OneVsOneClassifier(
        LogisticRegression(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )
    )
    # Fit the training datset
    model_fit = logres.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = logres.predict(X_test)
    return modeleval(
        y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose
    )


def compute_lr_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    model="Logistic Regression",
    cv=5,
    penalty=None,
    C=None,
    solver=None,
    verbose=False,
    max_iter=None,
    random_state=None,
    *,
    tuner="optuna",
    n_trials=50,
):
    """This function also generates a model using a Logistic Regression (LR) method as implemented in
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`__.
    The difference here is that this function tunes the model's hyperparameters.
    The values or ranges searched for each parameter are specified in the config.yaml file,
    and ``tuner`` selects the search engine (Optuna by default). The
    combination of parameters that led to the best performance is saved and returned as best_params, which can then be used on similar
    datasets, without having to repeat the search. The function returns the evaluation of the model
    on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model across the search.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Args:
        X_train (numpy.ndarray): Training data features.
        X_test (numpy.ndarray): Test data features.
        y_train (numpy.ndarray): Training data labels.
        y_test (numpy.ndarray): Test data labels.
        args (dict): Additional arguments, such as dataset name and other configurations.
        model (str): Name of the model being used, default is 'Logistic Regression'.
        cv (int): Number of cross-validation folds, default is 5.
        penalty (list): List of penalties to try, default is an empty list.
        C (list): List of inverse regularization strengths to try, default is an empty list.
        solver (list): List of solvers to try, default is an empty list.
        verbose (bool): Whether to print detailed logs, default is False.
        max_iter (list): List of maximum iterations to try, default is an empty list.
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
        "penalty": penalty,
        "C": C,
        "solver": solver,
        "max_iter": max_iter,
    }
    # Optuna by default; the exhaustive grid stays reachable so a number published
    # against it can still be reproduced. Both engines are handed the same
    # `candidates`, so switching `tuner` never changes *which* hyperparameters are
    # searched -- only how the search spends its fits.
    if tuner == "grid":
        search = GridSearchCV(
            LogisticRegression(random_state=random_state),
            param_grid=build_param_grid("lr", candidates),
            cv=cv,
        )
        search.fit(X_train, y_train)
        best_params = search.best_params_
    else:
        best_params = run_study(
            LogisticRegression,
            build_search_space("lr", candidates),
            X_train,
            y_train,
            cv=cv,
            n_trials=n_trials,
            seed=random_state,
            fixed={"random_state": random_state},
        )
    best_logres = LogisticRegression(**best_params, random_state=random_state)
    best_logres.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_logres.predict(X_test)
    return modeleval(y_test, y_predicted, beg_time, best_params, args, model=model, verbose=verbose)
