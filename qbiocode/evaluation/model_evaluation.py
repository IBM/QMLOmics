# ====== Base class imports ======

import time
from typing import Literal

import numpy as np
import pandas as pd

# ====== Scikit-learn imports ======

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from qbiocode.utils.helper_fn import print_results


def _positive_class_column(scores, estimator):
    """Which column of a two-column score matrix holds the positive class.

    ``roc_auc_score`` treats the *larger* of the two labels in ``y_true`` as positive,
    so the score vector handed to it has to be the column for that same label. Both
    scikit-learn and qiskit-machine-learning order their output columns by sorted
    class label, which makes ``scores[:, 1]`` right almost always -- and silently
    inverted (AUC ``1 - a``) on the estimator that does not, which is exactly the kind
    of defect a plausible-looking number in [0, 1] hides. So the column is looked up
    through ``classes_`` rather than hardcoded.

    Args:
        scores (numpy.ndarray): A ``(n_samples, 2)`` score or probability matrix.
        estimator: The fitted estimator that produced ``scores``.

    Returns:
        int or None: The index of the positive-class column, or ``None`` if
        ``classes_`` is present but disagrees with the width of ``scores`` -- a
        mismatch means the mapping cannot be established and no guess should be made.
        Estimators with no ``classes_`` at all (``PegasosQSVC``,
        ``NeuralNetworkClassifier``) fall back to the last column, which is the
        sorted-label convention both of them follow.
    """
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        return scores.shape[1] - 1
    classes = np.asarray(classes)
    if classes.shape[0] != scores.shape[1]:
        return None
    # np.argmax would do for integer labels and break on string ones; comparing
    # against .max() works for both.
    return int(np.flatnonzero(classes == classes.max())[0])


def _score_vector(raw, estimator):
    """Reduce whatever a scoring method returned to one ranking value per sample.

    Args:
        raw (numpy.ndarray): The output of ``predict_proba`` or ``decision_function``.
        estimator: The fitted estimator that produced ``raw``.

    Returns:
        numpy.ndarray or None: A one-dimensional score array, or ``None`` when ``raw``
        carries no single ranking -- three or more classes, or a two-column matrix
        whose columns cannot be mapped to labels.
    """
    if raw.ndim == 1:
        # ``OneVsOneClassifier.decision_function`` and ``SVC.decision_function`` both
        # collapse to this shape on a binary problem, already oriented so that larger
        # means the positive class.
        return raw
    if raw.ndim != 2:
        return None
    if raw.shape[1] == 1:
        # ``EstimatorQNN.forward`` maps to a single value in [-1, +1] and
        # ``NeuralNetworkClassifier`` classifies by its sign, so it is a ranking even
        # though it is not a probability.
        return raw.ravel()
    if raw.shape[1] != 2:
        # Three or more classes. A multiclass AUC needs the whole matrix and an
        # explicit averaging choice, which is a different metric from the binary one
        # this column has always reported; see ``evaluation_metrics`` for that.
        return None
    column = _positive_class_column(raw, estimator)
    if column is None:
        return None
    return raw[:, column]


def extract_binary_scores(estimator, X):
    """Best continuous score a fitted estimator can give for a binary problem.

    ``roc_auc_score`` needs a *ranking* -- probabilities or decision values -- not
    predicted labels. Which method supplies one differs across the learners in this
    package, and the differences are not incidental:

    * The seven older classical learners (``dt``, ``lr``, ``mlp``, ``nb``, ``rf``,
      ``svc``, ``xgb``) are fitted inside a :class:`~sklearn.multiclass.OneVsOneClassifier`.
      That wrapper has **no** ``predict_proba``, which is why the AUC here could not
      simply be read off probabilities -- but it does have ``decision_function``. On a
      binary target the wrapper holds exactly one pairwise estimator, and its
      ``decision_function`` is that estimator's own ranking, returned as a
      ``(n_samples,)`` vector oriented towards the positive class. That is a genuine
      AUC input. ``compute_svc`` asks ``SVC`` for ``probability=True``; the wrapper
      does not expose the result, which costs nothing now -- ``decision_function``
      makes Platt scaling unnecessary here rather than the passing of it broken.
    * A fully grown ``DecisionTreeClassifier`` produces only two distinct
      ``decision_function`` values, so its AUC from scores equals its AUC from labels.
      That is honest: a tree of pure leaves has no ranking to offer. It is not a
      reason to fall back to labels for everything else.
    * ``catboost`` and ``tabpfn``, every ``_opt`` twin, and the searched heads inside
      ``compute_pqk``/``compute_qpl`` are fitted unwrapped, so ``predict_proba`` is
      available and preferred.
    * On the quantum side ``QSVC`` inherits ``SVC.decision_function``; ``PegasosQSVC``,
      ``VQC`` and ``NeuralNetworkClassifier`` all publish ``predict_proba``.

    ``predict_proba`` is tried first and ``decision_function`` second. The two are
    monotonically related wherever both exist, so the order does not change an AUC --
    it only prefers the more interpretable of the two.

    Args:
        estimator: A fitted classifier.
        X (numpy.ndarray): The samples to score, in the same space the estimator was
            fitted in (for ``compute_pqk``/``compute_qpl`` that is the quantum
            projection, not the raw features).

    Returns:
        numpy.ndarray or None: One score per row of ``X``, larger meaning the positive
        class; or ``None`` when the estimator offers no ranking at all. ``None`` is a
        real answer and callers must pass it on -- ``modeleval`` records NaN for it
        rather than substituting a label-based number, which is the bug this helper
        exists to fix.
    """
    for name in ("predict_proba", "decision_function"):
        # getattr-with-default rather than hasattr-then-call: scikit-learn guards
        # ``SVC.predict_proba`` with ``available_if``, which raises AttributeError on
        # *attribute access* when ``probability=False``, and getattr swallows that.
        method = getattr(estimator, name, None)
        if not callable(method):
            continue
        try:
            raw = np.asarray(method(X), dtype=float)
        except (AttributeError, NotImplementedError):
            # An estimator that advertises the method and refuses to run it. Treated
            # as "no score from this route" so the next one still gets a turn.
            continue
        scores = _score_vector(raw, estimator)
        if scores is not None:
            return scores
    return None


def _was_tuned(model, tuned):
    """Whether this row's parameters came from a hyperparameter search.

    Args:
        model (str): The label this row is filed under.
        tuned (bool or None): An explicit answer, or None to infer one.

    Returns:
        bool: ``tuned`` when given, else whether ``model`` ends in ``_opt``.

    Note:
        The inference works because ``model_run`` labels a tuned run ``<name>_opt`` --
        the dispatch key and the column name are the same string. It is deliberately not
        a substring test: ``qpl_opt_rf`` contains ``_opt`` and so does a hypothetical
        head called ``adam_optimizer``, so ``compute_qpl`` states the answer instead of
        relying on a match that would be right by luck.
    """
    if tuned is not None:
        return bool(tuned)
    return str(model).endswith("_opt")


def modeleval(
    y_test,
    y_predicted,
    beg_time,
    params,
    args,
    model: str,
    verbose=True,
    average="weighted",
    y_score=None,
    tuned=None,
):
    """
    Evaluates the model performance using accuracy, F1 score, and ROC AUC.

    ``accuracy`` and ``f1_score`` are computed from ``y_predicted``. ``auc`` is
    computed from ``y_score`` and from nothing else.

    **The ``auc`` column changed meaning here.** It used to be
    ``roc_auc_score(y_test, y_predicted)`` -- ``roc_auc_score`` applied to *hard
    predicted labels*, which on a binary target is identically
    ``balanced_accuracy_score(y_test, y_predicted)`` and not any ranking AUC. Every
    ``auc`` QBioCode wrote before this change is that statistic, including the
    committed ``tutorial/QSage/data/qprofiler_benchmarks.csv`` that QuantumSage trains
    on, so old and new numbers are not comparable.

    ``auc`` is ``float('nan')`` when, and only when, a real AUC cannot be computed:

    * ``y_score`` is ``None`` -- the caller's estimator exposes neither
      ``predict_proba`` nor ``decision_function`` (see :func:`extract_binary_scores`,
      which returns ``None`` for exactly that case);
    * ``y_score`` carries no single ranking, because the target has three or more
      classes;
    * ``y_test`` holds one class only, so no ROC curve is defined.

    NaN is deliberate. Falling back to the label-based number would put a different
    statistic under the same column name, which is what went wrong before, and every
    reader of this column -- ``qc_winner_finder``, QuantumSage, the correlation
    analysis -- would carry it into a published figure believing it to be an AUC. A
    missing value is visible; a mislabelled one is not.

    Args:
        y_test (array-like): True labels for the test set.
        y_predicted (array-like): Predicted labels by the model.
        beg_time (float): Start time for measuring execution time.
        params (dict): Model parameters used during training.
        args (dict): Retained for signature compatibility; no longer read.
            ``args['grid_search']`` was the only key this function ever used, and it was
            the wrong signal -- a run-wide flag deciding a per-row column (see the comment
            at the parameter-column branch below). ``tuned`` replaced it. The parameter
            stays because all 24 call sites pass it positionally, and because dropping it
            would be a breaking change to a public function for no gain. One incidental
            benefit: a direct ``compute_<model>(...)`` call with an ``args`` dict that has
            no ``'grid_search'`` key used to raise ``KeyError`` here, *after* the fit had
            completed. It no longer can.
        model (str): Name of the model being evaluated.
        verbose (bool): If True, prints the evaluation results.
        average (str): Type of averaging to use for F1 score calculation.
            Default is 'weighted'.
        y_score (array-like or None): Continuous scores for the positive class, one
            per test sample -- probabilities or decision values, never labels. This is
            the only input to ``auc``. Default None, which records NaN.
        tuned (bool or None): Whether a hyperparameter search produced ``params``, which
            decides between the ``BestParams_Tuned`` and ``Model_Parameters`` column.
            Default None means infer it from ``model``: the dispatcher labels a tuned run
            ``<name>_opt``, so the label already carries the answer for 13 of the 14
            learners. ``compute_qpl`` passes it explicitly, because its label is
            ``qpl_opt_<head>`` and the marker is not a suffix.

    Returns:
        pd.DataFrame: DataFrame containing the evaluation results, including accuracy, F1 score, AUC, and model parameters.
    """
    # Calculate evaluation metrics
    if y_score is None:
        auc = float("nan")
    else:
        try:
            auc = roc_auc_score(y_test, np.asarray(y_score, dtype=float))
        except ValueError:
            # A single-class y_test, or a multiclass one that slipped past
            # ``_score_vector``. Reported as missing for the reason in the docstring;
            # ``evaluation_metrics`` below answers a malformed AUC request the same way.
            auc = float("nan")
    accuracy = accuracy_score(y_test, y_predicted, normalize=True)
    f1 = f1_score(y_test, y_predicted, average=average)
    compile_time = time.time() - beg_time
    params = params
    if verbose == True:
        print_results(model, accuracy, f1, compile_time, params)

    # The tuned-parameter column is named for the branch that produced it. It used to
    # be 'BestParams_GridSearch' back when an exhaustive grid was the only search;
    # Optuna is now the default, so the name no longer claims an engine. Readers
    # (qc_winner_finder, QuantumSage) accept the old name too, because every
    # ModelResults.csv written before this change carries it.
    #
    # Decided per ROW, not per run. This used to read `args["grid_search"] == True`, a
    # run-wide flag -- so in a `grid_search: True` run every model reported under
    # 'BestParams_Tuned' whether or not it had been tuned. That mattered because tuning
    # is not run-wide: quantum models stay untuned unless `tune_quantum` is also set (the
    # documented default), so their feature-map and kernel *defaults* were filed under a
    # column claiming a search that never ran -- and both qc_winner_finder and
    # QuantumSage prefer that column when reading parameters back.
    #
    # A consequence worth knowing: one run can now carry BOTH columns -- tuned classical
    # models and untuned quantum ones in the same table. Every reader already accepts
    # either name; qc_winner_finder additionally coalesces them per row.
    if _was_tuned(model, tuned):
        return pd.DataFrame(
            {
                "y_test_" + model: [y_test],
                "y_predicted_" + model: [y_predicted],
                "results_"
                + model: [
                    {
                        "model": model,
                        "accuracy": accuracy,
                        "f1_score": f1,
                        "time": compile_time,
                        "auc": auc,
                        "BestParams_Tuned": params,
                    }
                ],
            }
        )
    else:
        return pd.DataFrame(
            {
                "y_test_" + model: [y_test],
                "y_predicted_" + model: [y_predicted],
                "results_"
                + model: [
                    {
                        "model": model,
                        "accuracy": accuracy,
                        "f1_score": f1,
                        "time": compile_time,
                        "auc": auc,
                        "Model_Parameters": params,
                    }
                ],
            }
        )


def evaluation_metrics(predictions, y_test, metrics=["accuracy", "brier"], save=False):
    """
    Calculate evaluation metrics for classification predictions.

    Computes specified metrics for model predictions. Supports accuracy, Brier score,
    F1 score, precision, recall, and AUC-ROC. The Brier score measures the mean
    squared difference between predicted probabilities and actual outcomes, providing
    a measure of calibration quality.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities, shape (n_samples, n_classes)
    y_test : np.ndarray
        True labels, shape (n_samples,)
    metrics : list of str, optional
        List of metrics to compute. Options: 'accuracy', 'brier', 'f1',
        'precision', 'recall', 'auc' (default: ['accuracy', 'brier'])
    save : bool, optional
        Whether to save results (reserved for future use, default: False)

    Returns
    -------
    tuple or dict
        If metrics=['accuracy', 'brier'] (default): returns (accuracy, brier_score)
        Otherwise: returns dict with requested metrics as keys

    Examples
    --------
    >>> import numpy as np
    >>> from qbiocode.evaluation import evaluation_metrics
    >>>
    >>> # Binary classification example - default metrics
    >>> predictions = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    >>> y_test = np.array([0, 1, 0])
    >>> accuracy, brier = evaluation_metrics(predictions, y_test)
    >>> print(f"Accuracy: {accuracy:.2f}, Brier Score: {brier:.3f}")
    Accuracy: 1.00, Brier Score: 0.060

    >>> # Multiple metrics
    >>> results = evaluation_metrics(predictions, y_test,
    ...                              metrics=['accuracy', 'brier', 'f1', 'auc'])
    >>> print(results)
    {'accuracy': 1.0, 'brier': 0.06, 'f1': 1.0, 'auc': 1.0}

    Notes
    -----
    - For binary classification, Brier score is computed using the probability
      of the positive class
    - For multi-class classification, the average Brier score across all classes
      is returned
    - F1, precision, and recall use weighted averaging for multi-class
    - AUC uses one-vs-rest for multi-class
    - Lower Brier scores indicate better calibrated probability predictions

    References
    ----------
    Brier, G. W. (1950). "Verification of forecasts expressed in terms of probability".
    Monthly Weather Review, 78(1), 1-3.
    """
    import numpy as np
    from sklearn.metrics import (
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    # Get predicted classes
    y_pred = np.argmax(predictions, axis=1)

    results = {}

    # Calculate requested metrics
    if "accuracy" in metrics:
        results["accuracy"] = accuracy_score(y_test, y_pred)

    if "brier" in metrics:
        if predictions.shape[1] == 2:
            # Binary classification: use probability of positive class
            results["brier"] = brier_score_loss(y_test, predictions[:, 1])
        else:
            # Multi-class: use average Brier score across all classes
            results["brier"] = np.mean(
                [
                    brier_score_loss(y_test == i, predictions[:, i])
                    for i in range(predictions.shape[1])
                ]
            )

    if "f1" in metrics:
        results["f1"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    if "precision" in metrics:
        results["precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)

    if "recall" in metrics:
        results["recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    if "auc" in metrics:
        try:
            if predictions.shape[1] == 2:
                # Binary classification
                results["auc"] = roc_auc_score(y_test, predictions[:, 1])
            else:
                # Multi-class: one-vs-rest
                results["auc"] = roc_auc_score(
                    y_test, predictions, multi_class="ovr", average="weighted"
                )
        except ValueError:
            # Handle cases where AUC cannot be computed (e.g., single class in y_test)
            results["auc"] = np.nan

    # For backward compatibility: return tuple if default metrics
    if metrics == ["accuracy", "brier"]:
        return results["accuracy"], results["brier"]

    return results
