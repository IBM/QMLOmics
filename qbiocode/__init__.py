# Copyright 2026, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
QBioCode: Quantum Machine Learning for Biological Data Analysis
================================================================

QBioCode is a comprehensive Python package for quantum machine learning (QML)
research and applications in biological data analysis. It provides tools for
data generation, classical and quantum machine learning algorithms, evaluation
metrics, and visualization utilities.

Main Modules
------------
- learning: Classical and quantum machine learning algorithms
- embeddings: Feature embedding and encoding methods
- evaluation: Model, dataset and graph-complexity evaluation tools
- data_generation: Synthetic dataset generators
- visualization: Result visualization and correlation analysis
- utils: Helper functions and utilities
- apps: Command-line applications (QProfiler, QSage, QuVINE)

Quick Start
-----------
>>> from qbiocode import compute_rf, generate_data
>>> # Generate synthetic data
>>> generate_data(type_of_data='circles', save_path='data/circles')
>>> # Train a random forest model
>>> results = compute_rf(X_train, y_train, X_test, y_test)
>>> # Use QProfiler programmatically
>>> from qbiocode.apps.qprofiler import qprofiler
>>> qprofiler.main(config)
"""

# ====== Order the OpenMP runtimes before importing anything else ======
# Must come first: xgboost and torch each vendor a copy of libomp, and whichever
# starts second can crash the process with SIGSEGV during model fitting. See
# qbiocode.utils._openmp for the measurements behind this.
from .utils._openmp import preload_openmp_libraries

preload_openmp_libraries()

# ====== Import data generation functions ======
from .data_generation import (
    generate_circles_datasets,
    generate_classification_datasets,
    generate_moons_datasets,
    generate_s_curve_datasets,
    generate_spheres_datasets,
    generate_spirals_datasets,
    generate_swiss_roll_datasets,
)
from .data_generation.generator import generate_data

# ====== Import embedding functions ======
from .embeddings.embed import (
    QUVINE_HEADLINE_METHODS,
    QUVINE_METHODS,
    SKLEARN_METHODS,
    get_embeddings,
    is_transductive,
    pqk,
)

# ====== Import evaluation functions ======
from .evaluation.dataset_evaluation import evaluate
from .evaluation.graph_evaluation import evaluate_graph
from .evaluation.model_evaluation import modeleval
from .evaluation.model_run import model_run

# ====== Import learning functions ======
from .learning.compute_dt import compute_dt, compute_dt_opt
from .learning.compute_catboost import compute_catboost, compute_catboost_opt
from .learning.compute_lr import compute_lr, compute_lr_opt
from .learning.compute_mlp import compute_mlp, compute_mlp_opt
from .learning.compute_nb import compute_nb, compute_nb_opt
# Both twins of every quantum learner, matching the classical ones. The `_opt`
# variants and compute_qpl are all reachable through `model_run`'s dispatch table, so
# leaving them off the package root made 6 of the 28 dispatchable functions importable
# only by their private module path -- an inconsistency rather than a decision, since
# every classical learner exports both twins. Guarded by
# tests/test_model_contract_matrix.py.
from .learning.compute_pqk import compute_pqk, compute_pqk_opt
from .learning.compute_qnn import compute_qnn, compute_qnn_opt
from .learning.compute_qpl import compute_qpl, compute_qpl_opt
from .learning.compute_qsvc import compute_qsvc, compute_qsvc_opt
from .learning.compute_rf import compute_rf, compute_rf_opt
from .learning.compute_svc import compute_svc, compute_svc_opt
from .learning.compute_vqc import compute_vqc, compute_vqc_opt

# compute_xgb.py guards the xgboost import itself and both functions raise an
# actionable ImportError -- naming libomp and the exact reinstall command -- when
# it is missing, so this import is unconditional. Wrapping it in try/except and
# binding None on failure would replace that message with
# "'NoneType' object is not callable" and would additionally hide a genuine
# breakage in the module (a typo, a broken sibling import) as a missing extra.
from .learning.compute_xgb import compute_xgb, compute_xgb_opt

# TabPFN is an optional extra ([tabpfn]) and its module imports `tabpfn` lazily, so
# this import costs nothing when the extra is absent and -- importantly -- does not
# pull torch in, which would undo preload_openmp_libraries() above. Both functions
# raise an actionable ImportError naming the extra and the license step when called
# without it.
from .learning.compute_tabpfn import compute_tabpfn, compute_tabpfn_opt

# ====== Import helper functions ======
from .utils.dataset_checkpoint import checkpoint_restart
from .utils.helper_fn import feature_encoding, scale_train_test, scaler_fn
from .utils.qc_winner_finder import qml_winner
from .utils.tutorial_data import tutorial_data_dirs, tutorial_data_path
from .version import __version__

# ====== Import visualization functions ======
from .visualization.visualize_correlation import (
    compute_results_correlation,
    plot_results_correlation,
    publication_style,
)

# ====== Expose apps submodule ======
# Apps are available as qbiocode.apps.qprofiler, qbiocode.apps.sage
from . import apps  # noqa: F401

__all__ = [
    # Version
    "__version__",
    # Classical ML algorithms
    "compute_catboost",
    "compute_catboost_opt",
    "compute_svc",
    "compute_svc_opt",
    "compute_dt",
    "compute_dt_opt",
    "compute_nb",
    "compute_nb_opt",
    "compute_lr",
    "compute_lr_opt",
    "compute_rf",
    "compute_rf_opt",
    "compute_xgb",
    "compute_xgb_opt",
    "compute_tabpfn",
    "compute_tabpfn_opt",
    "compute_mlp",
    "compute_mlp_opt",
    # Quantum ML algorithms
    "compute_qnn",
    "compute_qnn_opt",
    "compute_qsvc",
    "compute_qsvc_opt",
    "compute_vqc",
    "compute_vqc_opt",
    "compute_pqk",
    "compute_pqk_opt",
    "compute_qpl",
    "compute_qpl_opt",
    # Embeddings
    "get_embeddings",
    "is_transductive",
    "SKLEARN_METHODS",
    "QUVINE_HEADLINE_METHODS",
    "QUVINE_METHODS",
    "pqk",
    # Utilities
    "scaler_fn",
    "scale_train_test",
    "feature_encoding",
    "qml_winner",
    "checkpoint_restart",
    "tutorial_data_path",
    "tutorial_data_dirs",
    # Evaluation
    "modeleval",
    "evaluate",
    "evaluate_graph",
    "model_run",
    # Visualization
    "plot_results_correlation",
    "compute_results_correlation",
    "publication_style",
    # Data generation
    "generate_data",
    "generate_circles_datasets",
    "generate_moons_datasets",
    "generate_classification_datasets",
    "generate_s_curve_datasets",
    "generate_spheres_datasets",
    "generate_spirals_datasets",
    "generate_swiss_roll_datasets",
    # Apps submodule
    "apps",
]
