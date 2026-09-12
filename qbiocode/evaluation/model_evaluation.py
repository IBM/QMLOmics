# ====== Base class imports ======

import time
from typing import Literal
import pandas as pd

# ====== Scikit-learn imports ======

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from qbiocode.utils.helper_fn import print_results


def modeleval(
    y_test, y_predicted, beg_time, params, args, model: str, verbose=True, average="weighted"
):
    """
    Evaluates the model performance using accuracy, F1 score, and AUC.

    Args:
        y_test (array-like): True labels for the test set.
        y_predicted (array-like): Predicted labels by the model.
        beg_time (float): Start time for measuring execution time.
        params (dict): Model parameters used during training.
        args (dict): Additional arguments, including grid search flag.
        model (str): Name of the model being evaluated.
        verbose (bool): If True, prints the evaluation results.
        average (str): Type of averaging to use for F1 score calculation.
            Default is 'weighted'.

    Returns:
        pd.DataFrame: DataFrame containing the evaluation results, including accuracy, F1 score, AUC, and model parameters.
    """
    # Calculate evaluation metrics
    auc = roc_auc_score(y_test, y_predicted)
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
    if args["grid_search"] == True:
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
