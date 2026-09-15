# ====== Base class imports ======

import time

from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC

# ====== Additional local imports ======
from qbiocode.learning._grid import build_param_grid
from qbiocode.learning._tuning import build_search_space, run_study
from qbiocode.evaluation.model_evaluation import extract_binary_scores, modeleval

# ====== Scikit-learn imports ======


def compute_svc(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    model="SVC",
    data_key="",
    C=1.0,
    kernel="rbf",
    degree=3,
    gamma="scale",
    coef0=0.0,
    shrinking=True,
    probability=False,
    tol=0.001,
    cache_size=200,
    class_weight=None,
    verbose=False,
    max_iter=-1,
    decision_function_shape="ovr",
    break_ties=False,
    random_state=None,
):
    """This function generates a model using a Support Vector Classifier (SVC) method as implemented in
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`__.
    It takes in parameter arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed.
    The model is trained on the training dataset and validated on the test dataset.  The model is trained on the training dataset and validated on the test dataset.
    The function returns the evaluation of the model on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Additional arguments, typically from a configuration file.
        model (str): The type of model to use, default is 'SVC'.
        data_key (str): Key for the dataset, default is an empty string.
        C (float): Regularization parameter, default is 1.0.
        kernel (str): Specifies the kernel type to be used in the algorithm, default is 'rbf'.
        degree (int): Degree of the polynomial kernel function ('poly'), default is 3.
        gamma (str or float): Kernel coefficient for 'rbf', 'poly', and 'sigmoid', default is 'scale'.
        coef0 (float): Independent term in kernel function, default is 0.0.
        shrinking (bool): Whether to use the shrinking heuristic, default is True.
        probability (bool): Whether to enable probability estimates, default is False.
        tol (float): Tolerance for stopping criteria, default is 0.001.
        cache_size (int): Size of the kernel cache in MB, default is 200.
        class_weight (dict or None): Weights associated with classes, default is None.
        verbose (bool): Whether to print detailed logs, default is False.
        max_iter (int): Hard limit on iterations within solver, -1 means no limit, default is -1.
        decision_function_shape (str): Determines the shape of the decision function, default is 'ovr'.
        break_ties (bool): Whether to break ties in multiclass classification, default is False.
        random_state (int or None): Controls the randomness of the estimator, default is None.
        tuner (str): Which search to run. ``'optuna'`` (default) spends ``n_trials`` on
            Optuna's TPE sampler, which also allows a hyperparameter to be given as a
            ``{low, high}`` range rather than a list. ``'grid'`` restores the exhaustive
            ``GridSearchCV`` sweep over every combination.
        n_trials (int): Trial budget when ``tuner='optuna'``, default is 50. Lowered
            automatically when the configured values describe fewer distinct
            combinations than that, so a small block does not re-evaluate the same
            models.
    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model, including accuracy, AUC, F1 score, and the time taken to train and validate the model.
    """

    beg_time = time.time()
    svc = OneVsOneClassifier(
        SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )
    )
    # Fit the training datset
    model_fit = svc.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = svc.predict(X_test)
    # `auc` is a ranking metric and is computed from these scores alone -- passing
    # y_predicted, as this used to, silently reported balanced accuracy instead.
    # OneVsOneClassifier publishes no predict_proba, so what comes back here is its
    # decision_function; see extract_binary_scores for why that is a real ranking on a
    # binary target, and None (recorded as NaN) when it is not. The `probability=True`
    # above is what a reader would expect to supply this instead; OneVsOneClassifier
    # discards it, which the decision_function route makes harmless rather than broken.
    y_score = extract_binary_scores(svc, X_test)
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


def compute_svc_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    cv=5,
    model="SVC",
    C=None,
    gamma=None,
    kernel=None,
    random_state=None,
    *,
    tuner="optuna",
    n_trials=50,
):
    """This function generates a model using a Support Vector Classifier (SVC) method as implemented in
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`__.
    It takes in parameter arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed. The
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
        args (dict): Additional arguments, typically from a configuration file.
        verbose (bool): Whether to print detailed logs, default is False.
        cv (int): Number of cross-validation folds, default is 5.
        model (str): The type of model to use, default is 'SVC'.
        C (list or float): Regularization parameter(s), default is an empty list.
        gamma (list or str): Kernel coefficient(s) for 'rbf', 'poly', and 'sigmoid', default is an empty list.
        kernel (list or str): Specifies the kernel type(s) to be used in the algorithm, default is an empty list.
        random_state (int or None): Seed for the estimator's own randomness. QProfiler fills this in from the run's ``seed`` so two runs at one seed agree; None leaves the estimator drawing from the global RNG.
     Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model, including accuracy, AUC, F1 score, and the time taken to train and validate the model across the search.
    """

    beg_time = time.time()
    # Only the hyperparameters actually supplied. Passing all of them meant a
    # config that named a subset died in sklearn on the first one it left at its
    # `[]` default; see qbiocode.learning._grid.
    candidates = {
        "C": C,
        "gamma": gamma,
        "kernel": kernel,
    }
    # Optuna by default; the exhaustive grid stays reachable so a number published
    # against it can still be reproduced. Both engines are handed the same
    # `candidates`, so switching `tuner` never changes *which* hyperparameters are
    # searched -- only how the search spends its fits.
    if tuner == "grid":
        search = GridSearchCV(
            SVC(random_state=random_state),
            param_grid=build_param_grid("svc", candidates),
            cv=cv,
        )
        search.fit(X_train, y_train)
        best_params = search.best_params_
    else:
        best_params = run_study(
            SVC,
            build_search_space("svc", candidates),
            X_train,
            y_train,
            cv=cv,
            n_trials=n_trials,
            seed=random_state,
            fixed={"random_state": random_state},
        )
    best_svc = SVC(**best_params, random_state=random_state)
    best_svc.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_svc.predict(X_test)
    # Fitted unwrapped, but `probability` is not among the searched parameters, so this
    # bare SVC has no predict_proba either and extract_binary_scores falls through to
    # decision_function. `auc` is computed from these scores alone.
    y_score = extract_binary_scores(best_svc, X_test)
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
