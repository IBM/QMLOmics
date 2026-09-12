"""
Utilities Module for QBioCode
=============================

This module provides helper functions and utilities for data preprocessing,
model management, IBM Quantum account handling, and result analysis.

Available Functions
-------------------
- scaler_fn: Data scaling and normalization
- feature_encoding: Encode features for quantum circuits
- qml_winner: Identify best performing quantum model
- checkpoint_restart: Save and load model checkpoints
- track_progress: Track progress of dataset processing
- combine_results: Combine evaluation results from multiple runs
- find_duplicate_files: Find duplicate entries in datasets
- find_string_in_files: Search for strings in files
- generate_qml_experiment_configs: Generate config files for QML grid search
- get_creds: Get IBM Quantum credentials
- instantiate_runtime_service: Instantiate Qiskit Runtime Service
- get_backend_session: Get backend session for quantum execution
- get_sampler: Get sampler primitive
- get_estimator: Get estimator primitive
- get_ansatz: Get quantum ansatz circuit
- get_feature_map: Get quantum feature map
- get_optimizer: Get classical optimizer
- normalize_data: Normalize data for quantum state encoding
- label_to_array: Convert binary labels to one-hot encoding
- prepare_training_set: Prepare balanced training subset
- retrieve_probabilities: Extract probabilities from measurement counts
- execute_circuit: Execute quantum circuit on Aer simulator
- tutorial_data_path: Locate a tutorial fixture across the repo data directories
- tutorial_data_dirs: The directories tutorial_data_path searches, in order

Usage
-----
>>> from qbiocode.utils import scaler_fn, feature_encoding
>>> # Scale data
>>> X_scaled = scaler_fn(X, scaling='StandardScaler')
>>> # Encode features for quantum circuits
>>> X_encoded = feature_encoding(X, feature_encoding='OneHotEncoder')
>>> # Prepare data for quantum ensemble
>>> from qbiocode.utils import normalize_data, prepare_training_set
>>> X_norm = normalize_data(X[0])
>>> X_train, Y_train = prepare_training_set(X, y, n=4, seed=42)
"""

from .combine_evals_results import combine_results, track_progress
from .dataset_checkpoint import checkpoint_restart
from .find_duplicates import find_duplicate_files
from .find_string import find_string_in_files
from .generate_qml_configs import generate_qml_experiment_configs
from .helper_fn import feature_encoding, scaler_fn
from .ibm_account import get_creds, instantiate_runtime_service, redacted
from .tabpfn_account import (
    check_tabpfn_access,
    describe_token_source,
    load_tabpfn_token,
    write_token_template,
)
from .qc_winner_finder import qml_winner
from .tutorial_data import tutorial_data_dirs, tutorial_data_path
from .qutils import (
    execute_circuit,
    get_ansatz,
    get_backend_session,
    get_estimator,
    get_feature_map,
    get_optimizer,
    get_sampler,
    retrieve_probabilities,
)
from .data_encoding import (
    label_to_array,
    normalize_data,
    prepare_training_set,
)

__all__ = [
    "check_tabpfn_access",
    "redacted",
    "describe_token_source",
    "load_tabpfn_token",
    "write_token_template",
    # Data preprocessing
    "scaler_fn",
    "feature_encoding",
    # Model management
    "qml_winner",
    "checkpoint_restart",
    # Results management
    "track_progress",
    "combine_results",
    # Configuration generation
    "generate_qml_experiment_configs",
    # File utilities
    "find_duplicate_files",
    "find_string_in_files",
    # IBM Quantum utilities
    "get_creds",
    "instantiate_runtime_service",
    # Quantum utilities
    "get_backend_session",
    "get_sampler",
    "get_estimator",
    "get_ansatz",
    "get_feature_map",
    "get_optimizer",
    # Ensemble utilities
    "normalize_data",
    "label_to_array",
    "prepare_training_set",
    "retrieve_probabilities",
    "execute_circuit",
    # Tutorial fixtures
    "tutorial_data_path",
    "tutorial_data_dirs",
]
