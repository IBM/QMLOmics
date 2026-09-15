"""
Evaluation Module for QBioCode
==============================

This module provides comprehensive evaluation tools for machine learning
models and datasets. It includes functions for model performance assessment,
dataset complexity analysis, and automated model execution.

Dataset complexity is measured mostly with `pyMFE
<https://github.com/ealcobaca/pymfe>`_ -- those columns carry an ``mfe.`` prefix --
plus the measures pyMFE does not cover, which
:mod:`qbiocode.evaluation.dataset_evaluation` computes directly. QBioCode ships a
curated subset of pyMFE's meta-features rather than all of them;
:mod:`qbiocode.evaluation.mfe_features` records which, and the measured reason for
every exclusion.

A third block, prefixed ``task.``, comes from
:mod:`qbiocode.evaluation.task_spectrum`. Where the other two describe ``X``, it
describes where ``y`` sits in the *geometric spectrum* of ``X`` -- separating a
structured high-frequency target (parity, checkerboard, alternating-sign) from
one that is merely hard, and from broadband noise.

Available Functions
-------------------
- modeleval: Evaluate model performance with multiple metrics
- evaluation_metrics: Calculate accuracy and Brier score from predictions
- evaluate: Comprehensive dataset complexity evaluation
- model_run: Automated model training and evaluation pipeline
- get_mfe_features: The curated pyMFE meta-feature block alone
- get_task_spectrum_features: The target-spectrum block alone
- detect_complexity_schema: Identify which complexity schema a results table carries
- complexity_feature_columns: The same, tolerating an unrecognized table

Usage
-----
>>> from qbiocode.evaluation import modeleval, evaluate
>>> # Evaluate model performance
>>> metrics = modeleval(y_true, y_pred, y_proba)
>>> # Evaluate dataset complexity (one row; `name` labels it in the output)
>>> complexity_metrics = evaluate(X, y, "my_dataset.csv")
"""

from .dataset_evaluation import (
    LEGACY_COMPLEXITY_COLUMNS,
    NATIVE_COMPLEXITY_COLUMNS,
    complexity_feature_columns,
    detect_complexity_schema,
    evaluate,
)
from .mfe_features import MFE_FEATURES, get_mfe_features
from .model_evaluation import evaluation_metrics, modeleval
from .model_run import model_run
from .task_spectrum import (
    TASK_COLUMN_PREFIX,
    TASK_FEATURES,
    get_task_spectrum_features,
    task_column_names,
)

__all__ = [
    "modeleval",
    "evaluation_metrics",
    "evaluate",
    "model_run",
    # Dataset complexity: the pyMFE block, and the schema of the output
    "get_mfe_features",
    "MFE_FEATURES",
    # Dataset complexity: where y sits in the geometric spectrum of X
    "get_task_spectrum_features",
    "task_column_names",
    "TASK_FEATURES",
    "TASK_COLUMN_PREFIX",
    "detect_complexity_schema",
    "complexity_feature_columns",
    "NATIVE_COMPLEXITY_COLUMNS",
    "LEGACY_COMPLEXITY_COLUMNS",
]
