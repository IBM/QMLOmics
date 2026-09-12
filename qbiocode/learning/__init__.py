"""
Machine Learning Module for QBioCode
====================================

This module provides implementations of classical and quantum machine learning
algorithms for classification tasks. Each algorithm includes both standard and
optimized versions (where applicable) with hyperparameter tuning.

Classical Algorithms
--------------------
- CatBoost (catboost)
- Decision Tree (DT)
- Logistic Regression (LR)
- Multi-Layer Perceptron (MLP)
- Naive Bayes (NB)
- Random Forest (RF)
- Support Vector Classifier (SVC)
- TabPFN (tabpfn) -- pretrained tabular transformer; needs the [tabpfn] extra.
  Pinned to model_version 'v2', the version whose weights permit commercial use.
- XGBoost (XGB)

Quantum Algorithms
------------------
- Quantum Neural Network (QNN)
- Quantum Support Vector Classifier (QSVC)
- Variational Quantum Classifier (VQC)
- Projected Quantum Kernel (PQK)
- Quantum Ensemble (QEnsemble) - supports both fixed swap and random unitary methods

Usage
-----
>>> from qbiocode.learning import compute_rf, compute_qsvc, compute_qensemble
>>> # Train classical model
>>> results = compute_rf(X_train, y_train, X_test, y_test)
>>> # Train quantum model
>>> qresults = compute_qsvc(X_train, y_train, X_test, y_test)
>>> # Train quantum ensemble with fixed swaps (default)
>>> qens_results = compute_qensemble(X_train, X_test, y_train, y_test, args)
>>> # Train quantum ensemble with random unitaries
>>> qens_random = compute_qensemble(X_train, X_test, y_train, y_test, args,
...                                 ensemble_method="random_unitary")
"""

# Classical ML algorithms
from .compute_catboost import compute_catboost, compute_catboost_opt
from .compute_dt import compute_dt, compute_dt_opt
from .compute_lr import compute_lr, compute_lr_opt
from .compute_mlp import compute_mlp, compute_mlp_opt
from .compute_nb import compute_nb, compute_nb_opt
from .compute_rf import compute_rf, compute_rf_opt
from .compute_svc import compute_svc, compute_svc_opt

# compute_xgb.py guards the xgboost import itself and both functions raise an
# actionable ImportError -- naming libomp and the exact reinstall command -- when
# it is missing, so this import is unconditional. Wrapping it in try/except and
# binding None on failure would replace that message with
# "'NoneType' object is not callable" and would additionally hide a genuine
# breakage in the module (a typo, a broken sibling import) as a missing extra.
from .compute_xgb import compute_xgb, compute_xgb_opt

# compute_tabpfn.py imports `tabpfn` lazily, inside its functions, so this line
# neither requires the optional [tabpfn] extra nor maps torch's OpenMP runtime into
# the process -- both of which a module-level `from tabpfn import ...` would do. See
# that module's docstring and tests/test_openmp_import_order.py.
from .compute_tabpfn import compute_tabpfn, compute_tabpfn_opt

from .compute_pqk import compute_pqk, compute_pqk_opt
from .compute_qpl import compute_qpl, compute_qpl_opt

# Quantum ML algorithms
from .compute_pqk import compute_pqk, compute_pqk_opt
from .compute_qensemble import compute_qensemble
from .compute_qnn import compute_qnn, compute_qnn_opt
from .compute_qsvc import compute_qsvc, compute_qsvc_opt
from .compute_vqc import compute_vqc, compute_vqc_opt

__all__ = [
    # Classical algorithms
    "compute_catboost",
    "compute_catboost_opt",
    "compute_dt",
    "compute_dt_opt",
    "compute_lr",
    "compute_lr_opt",
    "compute_mlp",
    "compute_mlp_opt",
    "compute_nb",
    "compute_nb_opt",
    "compute_rf",
    "compute_rf_opt",
    "compute_svc",
    "compute_svc_opt",
    "compute_tabpfn",
    "compute_tabpfn_opt",
    "compute_xgb",
    "compute_xgb_opt",
    # Quantum algorithms
    'compute_qnn',
    'compute_qnn_opt',
    'compute_qsvc',
    'compute_qsvc_opt',
    'compute_vqc',
    'compute_vqc_opt',
    'compute_pqk',
    'compute_pqk_opt',
    'compute_qpl',
    'compute_qpl_opt',
    'compute_qensemble',
]

