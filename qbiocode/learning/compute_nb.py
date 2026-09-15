# ====== Base class imports ======

import time

from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB

# ====== Additional local imports ======
from qbiocode.learning._grid import build_param_grid
from qbiocode.learning._tuning import build_search_space, run_study
from qbiocode.evaluation.model_evaluation import extract_binary_scores, modeleval

# ====== Scikit-learn imports ======


def compute_nb(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="Naive Bayes",
    data_key="",
    var_smoothing=1e-09,
):
    """This function generates a model using a Gaussian Naive Bayes (NB) Classifier method as implemented in
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html>`__.
    It takes in parameter arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed.
    The model is trained on the training dataset and validated on the test dataset. The function returns the evaluation of the model
    on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Args:
        X_train (numpy.ndarray): Training features.
        X_test (numpy.ndarray): Test features.
        y_train (numpy.ndarray): Training labels.
        y_test (numpy.ndarray): Test labels.
        args (dict): Additional arguments, such as config parameters.
        verbose (bool): If True, prints additional information during execution.
        model (str): Name of the model being used.
        data_key (str): Key for the dataset, if applicable.
        var_smoothing (float): Portion of the largest variance of all features added to variances for calculation stability.
    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model on the test dataset, including accuracy, AUC, F1 score,
                          and the time taken to train and validate the model, along with the model parameters.
    """

    beg_time = time.time()
    nb = OneVsOneClassifier(GaussianNB(var_smoothing=var_smoothing))
    # Fit the training datset
    model_fit = nb.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = nb.predict(X_test)
    # `auc` is a ranking metric and is computed from these scores alone -- passing
    # y_predicted, as this used to, silently reported balanced accuracy instead.
    # OneVsOneClassifier publishes no predict_proba, so what comes back here is its
    # decision_function; see extract_binary_scores for why that is a real ranking on a
    # binary target, and None (recorded as NaN) when it is not.
    y_score = extract_binary_scores(nb, X_test)
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


def compute_nb_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="Naive Bayes",
    cv=5,
    var_smoothing=[1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02],
    *,
    tuner="optuna",
    n_trials=50,
):
    """This function generates a model using a Gaussian Naive Bayes (NB) Classifier method as implemented in
    `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html>`__.
    It takes in parameter arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed. The
    combination of parameters that led to the best performance is saved and returned as best_params, which can then be used on similar
    datasets, without having to repeat the search.  The model is trained on the training dataset and validated on the test dataset. The function returns the evaluation of the model
    on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model across the search.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.
    Args:
        X_train (numpy.ndarray): Training features.
        X_test (numpy.ndarray): Test features.
        y_train (numpy.ndarray): Training labels.
        y_test (numpy.ndarray): Test labels.
        args (dict): Additional arguments, such as config parameters.
        verbose (bool): If True, prints additional information during execution.
        model (str): Name of the model being used.
        cv (int): Number of cross-validation folds for the search.
        var_smoothing (list): List of values for the var_smoothing parameter to be tested in the search.
        tuner (str): Which search to run. ``'optuna'`` (default) spends ``n_trials`` on
            Optuna's TPE sampler, which also allows a hyperparameter to be given as a
            ``{low, high}`` range rather than a list. ``'grid'`` restores the exhaustive
            ``GridSearchCV`` sweep over every combination.
        n_trials (int): Trial budget when ``tuner='optuna'``, default is 50. Lowered
            automatically when the configured values describe fewer distinct
            combinations than that, so a small block does not re-evaluate the same
            models.
    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model on the test dataset, including accuracy, AUC, F1 score,
                          and the time taken to train and validate the model, along with the best parameters found during the search.
    """

    beg_time = time.time()
    # Only the hyperparameters actually supplied. Passing all of them meant a
    # config that named a subset died in sklearn on the first one it left at its
    # `[]` default; see qbiocode.learning._grid.
    candidates = {
        "var_smoothing": var_smoothing,
    }
    # Optuna by default; the exhaustive grid stays reachable so a number published
    # against it can still be reproduced. Both engines are handed the same
    # `candidates`, so switching `tuner` never changes *which* hyperparameters are
    # searched -- only how the search spends its fits.
    if tuner == "grid":
        search = GridSearchCV(
            GaussianNB(),
            param_grid=build_param_grid("nb", candidates),
            cv=cv,
        )
        search.fit(X_train, y_train)
        best_params = search.best_params_
    else:
        best_params = run_study(
            GaussianNB,
            build_search_space("nb", candidates),
            X_train,
            y_train,
            cv=cv,
            n_trials=n_trials,
            # GaussianNB has no `random_state`, so `model_run._seeded_kwargs` does not
            # give this function one to pass on. The sampler still needs a seed or a
            # range over `var_smoothing` would search differently on every run, so read
            # the run's seed straight off the config.
            seed=args.get("seed") if isinstance(args, dict) else None,
            fixed={},
        )
    best_nb = GaussianNB(**best_params)
    best_nb.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_nb.predict(X_test)
    # Fitted unwrapped, so predict_proba is available. `auc` is computed from these
    # scores alone; see extract_binary_scores.
    y_score = extract_binary_scores(best_nb, X_test)
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
