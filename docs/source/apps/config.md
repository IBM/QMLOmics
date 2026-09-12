# QProfiler Configuration Guide

This guide explains how to configure QProfiler using YAML configuration files for reproducible experiments and batch processing.

## Overview

QProfiler uses YAML configuration files to define:
- Input datasets and output directories
- Machine learning models to evaluate
- Quantum backend settings
- Embedding methods and parameters
- Train/test split configuration
- Model hyperparameters

An example configuration file can be found at [`qbiocode/apps/qprofiler/configs/config.yaml`](https://github.com/qiskit-community/QBioCode/blob/main/qbiocode/apps/qprofiler/configs/config.yaml).

## Quick Start

Here's a minimal configuration to get started:

```yaml
# Basic configuration
config_file_name: 'my_experiment'
folder_path: 'data/'
file_dataset: 'my_dataset.csv'
seed: 42

# Models to evaluate
model: ['rf', 'svc', 'qsvc']

# Embedding
embeddings: ['none']
n_components: 3

# Train/test split
test_size: 0.2
stratify: ['y']
scaling: ['True']

# Quantum backend (for QML models)
backend: 'simulator'
shots: 1024
```

---

## Configuration Sections

### Input Data

Specify the location and selection of input datasets.

**Single Dataset:**

```yaml
config_file_name: 'experiment_name'
folder_path: 'data/'
file_dataset: 'dataset.csv'
```

**All Datasets in Folder:**

```yaml
folder_path: 'data/'
file_dataset: 'ALL'  # Process all CSV files in folder
```

**Multiple Specific Datasets:**

```yaml
file_dataset: ['dataset1.csv', 'dataset2.csv', 'dataset3.csv']
```

**Output Directory:**

```yaml
output_dir: 'results/'  # Where to save results
```

### Random Seeds

Set random seeds for reproducibility:

```yaml
seed: 42      # Seed for classical ML algorithms
q_seed: 42    # Seed for quantum algorithms
```

```{tip}
Always set seeds for reproducible experiments. Use the same seed across runs to compare results.
```

### Quantum Backend Configuration

Configure quantum computing backend for QML models (QSVC, VQC, QNN, PQK).

**Simulator (Default):**

```yaml
backend: 'simulator'
shots: 1024
```

Uses the Qiskit statevector simulator for exact, noiseless quantum simulation.

**AerSimulator with Custom Simulation Method:**

```yaml
backend: 'simulator_aer'
sim_method: 'statevector'  # Options: statevector, matrix_product_state, tensor_network, etc.
shots: 1024
```

Provides access to AerSimulator's various simulation methods. Useful for:
- **GPU acceleration**: Use `sim_method: 'tensor_network'` with GPU support
- **Memory efficiency**: Use `sim_method: 'matrix_product_state'` for low-entanglement circuits
- **Clifford circuits**: Use `sim_method: 'stabilizer'` for fast simulation

**Noisy Simulation Based on IBM Device:**

```yaml
backend: 'noisy_ibm_cleveland'  # Noisy simulation modeled on IBM device
sim_method: 'matrix_product_state'  # Simulation method for AerSimulator
shots: 1024
```

Simulates quantum circuits with realistic noise models from actual IBM Quantum devices. This feature:
- Extracts the noise model from a specified IBM device (e.g., `ibm_cleveland`, `ibm_kyoto`)
- Runs simulation locally using AerSimulator with the device's noise characteristics
- Allows testing quantum algorithms under realistic noise conditions without queue time
- Supports any simulation method available in AerSimulator

**Format:** `'noisy_<device_name>'` where `<device_name>` is any IBM Quantum device name.

**Examples:**
- `'noisy_ibm_cleveland'` - Noise model from IBM Cleveland
- `'noisy_ibm_kyoto'` - Noise model from IBM Kyoto
- `'noisy_ibm_sherbrooke'` - Noise model from IBM Sherbrooke

```{tip}
**Choosing a Simulation Method for Noisy Simulations:**

- **`matrix_product_state`**: Recommended for most quantum machine learning circuits. Efficient for circuits with moderate entanglement.
- **`tensor_network`**: Best for GPU-accelerated simulations. Requires `qiskit-aer-gpu` installation.
- **`statevector`**: Most accurate but memory-intensive. Limited to ~20-25 qubits depending on available RAM.
- **`automatic`**: Let AerSimulator choose the best method based on circuit properties.
```

**IBM Quantum Hardware:**

```yaml
backend: 'ibm_least'  # Use least busy device
# OR
backend: 'ibm_kyoto'  # Specific device name
shots: 4096
resil_level: 1  # Error mitigation level (1-3)
```

Runs circuits on actual IBM Quantum hardware.

**IBM Quantum Credentials:**

```yaml
qiskit_json_path: '~/.qiskit/qiskit-ibm.json'
name: 'account_qbc'  # Account alias in JSON
ibm_instance: 'ibm-q/open/main'  # Optional: specific instance
```

```{important}
**IBM Credentials Required for Noisy Simulations:**

When using `noisy_<device_name>` backends, you must provide valid IBM Quantum credentials even though the simulation runs locally. This is because QBioCode needs to:
1. Connect to IBM Quantum services to retrieve the device's noise model
2. Download the latest calibration data for accurate noise simulation

The actual circuit execution happens locally on your machine using AerSimulator, so you won't incur queue wait times or consume IBM Quantum compute credits.
```

```{note}
**Backend Options:**
- `'simulator'`: Local Qiskit statevector simulator (exact, noiseless)
- `'simulator_aer'`: AerSimulator with configurable simulation method
- `'noisy_<device_name>'`: Noisy simulation based on IBM device noise model (e.g., `'noisy_ibm_cleveland'`)
- `'ibm_least'`: Automatically select least busy IBM Quantum device
- `'ibm_<device_name>'`: Specific IBM Quantum device (e.g., 'ibm_kyoto')

**Simulation Methods (for AerSimulator):**
When using `'simulator_aer'` or `'noisy_<device_name>'` backends, specify the simulation method via `sim_method`:
- `'statevector'`: Exact statevector simulation (default, memory intensive)
- `'matrix_product_state'`: Efficient for low-entanglement circuits
- `'tensor_network'`: GPU-accelerated tensor network simulation
- `'stabilizer'`: Fast simulation for Clifford circuits
- `'extended_stabilizer'`: Approximate simulation for near-Clifford circuits
- `'automatic'`: Automatically select best method

**Shots:** Number of circuit executions. Higher = more accurate but slower.

**Resilience Level:** Error mitigation strength (1=light, 2=medium, 3=heavy). Higher = more accurate but slower.
```

### Embedding Methods

Dimensionality reduction techniques to apply before model training.

**No Embedding:**

```yaml
embeddings: ['none']
```

**Single Embedding Method:**

```yaml
embeddings: ['pca']
n_components: 3  # Reduce to 3 dimensions
```

**Multiple Embedding Methods:**

```yaml
embeddings: ['pca', 'nmf', 'umap', 'autoencoder']
n_components: 5
```

**Available Embedding Methods:**
- `'none'`: No dimensionality reduction
- `'pca'`: Principal Component Analysis
- `'nmf'`: Non-negative Matrix Factorization
- `'umap'`: Uniform Manifold Approximation and Projection
- `'autoencoder'`: Neural network autoencoder

```{tip}
Start with `'none'` to establish baseline performance, then try `'pca'` for faster quantum model training.
```

### Train/Test Split

Configure data splitting and preprocessing.

```yaml
test_size: 0.2      # 80% train, 20% test
stratify: ['y']     # Maintain class distribution
scaling: ['True']   # Standardize features
```

**Parameters:**
- `test_size`: Proportion of data for testing (0.0-1.0)
- `stratify`: `['y']` to maintain class balance, `['n']` for random split
- `scaling`: `['True']` to standardize features (recommended), `['False']` for raw data

```{warning}
Always use `stratify: ['y']` for imbalanced datasets to ensure both train and test sets have representative class distributions.
```

### Model Selection

Specify which machine learning models to evaluate.

**All Models:**

```yaml
model: ['svc', 'dt', 'lr', 'nb', 'rf', 'mlp', 'xgb', 'catboost', 'tabpfn',
        'qsvc', 'vqc', 'qnn', 'pqk']
```

**Classical Models Only:**

```yaml
model: ['rf', 'svc', 'lr', 'mlp', 'xgb', 'catboost']
```

**Quantum Models Only:**

```yaml
model: ['qsvc', 'vqc', 'qnn', 'pqk']
```

**Available Models:**

| Model | Type | Description |
|-------|------|-------------|
| `svc` | Classical | Support Vector Classifier |
| `dt` | Classical | Decision Tree |
| `lr` | Classical | Logistic Regression |
| `nb` | Classical | Naive Bayes |
| `rf` | Classical | Random Forest |
| `mlp` | Classical | Multi-Layer Perceptron |
| `xgb` | Classical | XGBoost |
| `catboost` | Classical | CatBoost gradient boosting |
| `tabpfn` | Classical | TabPFN pretrained tabular transformer (needs the `[tabpfn]` extra) |
| `qsvc` | Quantum | Quantum Support Vector Classifier |
| `vqc` | Quantum | Variational Quantum Classifier |
| `qnn` | Quantum | Quantum Neural Network |
| `pqk` | Quantum | Projected Quantum Kernel |

### Model Hyperparameters

Configure hyperparameters for each model. Each model has:
- **Standard arguments**: Single values for quick runs
- **Tuned arguments** (`gridsearch_<model>_args`): what to search when `grid_search: True`

A tuned argument may be written two ways:

| Syntax | Meaning |
|---|---|
| `C: [0.1, 1, 10]` | a list -- one of these values is chosen |
| `C: {low: 0.001, high: 100}` | a range -- sampled continuously |
| `C: {low: 0.001, high: 100, log: true}` | as above, on a log scale |
| `n_estimators: {low: 10, high: 500}` | integer bounds give integer values |

Two keys control the search itself:

```yaml
grid_search: True    # tune hyperparameters at all
tuner: optuna        # 'optuna' (default) or 'grid'
n_trials: 50         # Optuna's trial budget
cross_validation: 5  # folds used to score each candidate
```

`tuner: optuna` spends `n_trials` fits, steering them with a TPE sampler toward the
region that has been scoring well. `tuner: grid` restores the exhaustive
`GridSearchCV` sweep, which fits *every* combination -- the `gridsearch_rf_args`
block below is 576 combinations, or 2,880 fits at `cross_validation: 5`. Ranges
require `tuner: optuna`; under `tuner: grid` every entry must be a list.

#### Tuning the quantum models

The quantum classifiers (`qsvc`, `vqc`, `qnn`, `pqk`, `qpl`) tune through the same
`gridsearch_<model>_args` blocks, but tuning them is **off by default** and needs a
second key:

```yaml
grid_search: True        # tune at all
tune_quantum: True       # ... including the quantum models
n_trials_quantum: 10     # their budget: smaller, because each trial is a quantum fit
validation_split: 0.25   # holdout fraction used to score a quantum candidate
```

Two differences from the classical path, both driven by cost -- a quantum fit builds an
n-by-n fidelity kernel by circuit simulation, seconds rather than milliseconds:

- Each candidate is scored **once**, on a stratified holdout carved out of the training
  data, rather than on `cross_validation` folds. The test set is never touched by the
  search.
- `n_trials_quantum` defaults to 10 rather than 50.

Every quantum model named in `model` needs its own `gridsearch_<model>_args` block when
`tune_quantum` is on; a model with nothing to search is reported by name rather than
silently skipped. The five blocks, as shipped:

```yaml
gridsearch_qsvc_args:
  encoding:     ['Z', 'ZZ']
  reps:         [1, 2]
  entanglement: ['linear', 'full']
  C:            {low: 1.0e-2, high: 1.0e+2, log: true}

gridsearch_vqc_args:
  encoding:        ['Z', 'ZZ']
  reps:            [1, 2]
  ansatz_type:     ['amp']
  local_optimizer: ['COBYLA', 'L_BFGS_B']
  maxiter:         {low: 50, high: 200}

gridsearch_qnn_args:    # same keys as vqc
  encoding:        ['Z', 'ZZ']
  reps:            [1, 2]
  ansatz_type:     ['amp']
  local_optimizer: ['COBYLA', 'L_BFGS_B']
  maxiter:         {low: 50, high: 200}

gridsearch_pqk_args:
  encoding:     ['Z', 'ZZ']
  reps:         [1, 2]
  entanglement: ['linear', 'full']

gridsearch_qpl_args:    # same keys as pqk
  encoding:     ['Z', 'ZZ']
  reps:         [1, 2]
  entanglement: ['linear', 'full']
```

What is tunable per model:

| Model | Tunable |
|---|---|
| `qsvc` | `encoding`, `entanglement`, `reps`, `primitive`, `C`, `gamma`, `pegasos` |
| `vqc`, `qnn` | `encoding`, `entanglement`, `reps`, `primitive`, `ansatz_type`, `local_optimizer`, `maxiter` |
| `pqk`, `qpl` | `encoding`, `entanglement`, `reps`, `primitive` |

There is no `n_qubits`: the qubit count follows from the width of the data reaching the
model, so it is set by the embedding's `n_components`, not by tuning.

**Tuning on real hardware is refused** unless you also set `allow_hardware_tuning: True`.
Every trial is a separate queued job billed against your instance, and the failure mode
is silent -- the run simply never appears to finish. Tune on `backend: simulator`, then
run the winning configuration on the device.

`tune_quantum` without `grid_search` is an error rather than a no-op. `tuner: grid` only
ever applies to the classical models -- a quantum candidate is scored by running the whole
model, so there is no exhaustive-grid engine for one; setting both warns and still uses
Optuna for the quantum models.

QPL is scored on the **mean** accuracy across the classical heads it fits on the quantum
projection, which is what tuning the projection is meant to improve. Taking the best head
instead would let one lucky head choose the projection, and every head is reported anyway.

#### Projection caches

`pqk` and `qpl` cache their projected feature matrices so a rerun does not recompute
circuits. The file name includes a fingerprint of the settings that change the circuit
(`encoding`, `entanglement`, `reps`, `primitive`, feature width), and the row count is
checked on load. Redirect either cache if you want throwaway projections kept apart from
your real ones:

```yaml
pqk_projection_dir: pqk_projections   # default, relative to the working directory
qpl_projection_dir: qpl_projections   # default
```

Tuning does this automatically: every trial writes to a temporary directory that is
deleted when the search ends, so trial projections never collide with the final run's.

#### The results column

With tuning on, `ModelResults.csv` records the chosen hyperparameters in a
**`BestParams_Tuned`** column; with tuning off it records `Model_Parameters` instead,
never both. `BestParams_Tuned` was called `BestParams_GridSearch` before Optuna became
the default engine, and every reader (`qbiocode.utils.qc_winner_finder`, `QuantumSage`)
still accepts the old name, so results files written earlier keep working.

**Example: Support Vector Classifier (SVC)**

```yaml
# Standard run with fixed parameters
svc_args:
  C: 1.0
  gamma: 0.1
  kernel: 'rbf'

# Tuned: lists and ranges may be mixed freely
gridsearch_svc_args:
  C: {low: 0.001, high: 100, log: true}
  gamma: {low: 0.0001, high: 1, log: true}
  kernel: ['linear', 'rbf', 'poly', 'sigmoid']
```

**Example: Random Forest (RF)**

```yaml
rf_args:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 2

gridsearch_rf_args:
  n_estimators: [50, 100, 200]
  max_depth: [5, 10, 15, 20]
  min_samples_split: [2, 5, 10]
```

**Example: XGBoost (XGB)**

```yaml
xgb_args:
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 6

gridsearch_xgb_args:
  n_estimators: [50, 100, 200]
  learning_rate: [0.01, 0.1, 0.3]
  max_depth: [3, 6, 9]
```

**Example: CatBoost**

Any parameter you leave out stays at CatBoost's own default. That is not the same as
writing the documented default in: CatBoost *derives* several defaults from the data and
from each other, so naming one can change more than the one value. It auto-selects
`learning_rate` for `Logloss` and `MultiClass` unless `l2_leaf_reg` is set, and it
chooses `bootstrap_type` from the loss it inferred.

```yaml
catboost_args:
  iterations: 200
  learning_rate: 0.1
  depth: 6
  l2_leaf_reg: 3.0

gridsearch_catboost_args:
  iterations: [100, 200, 400]
  learning_rate: {low: 1.0e-2, high: 3.0e-1, log: true}
  depth: [4, 6, 8]
  l2_leaf_reg: {low: 1.0, high: 10.0}
  random_strength: [0.5, 1.0, 2.0]
```

```{note}
`min_data_in_leaf` is accepted but **not searchable**. CatBoost honours it only under
`grow_policy: Depthwise` or `Lossguide`; at the default `SymmetricTree` every value produces
an identical model, so searching it multiplies the fits for nothing. Give it a single value
alongside a `grow_policy` that honours it — several values are refused with a message saying
so.
```

```{warning}
**`subsample` and `bagging_temperature` are not interchangeable, and which one is legal
depends on the loss.** They belong to mutually exclusive CatBoost bootstrap schemes, and
CatBoost picks the default scheme from the loss: `MVS` under `Logloss`, `Bayesian` under
`MultiClass`. QProfiler is a binary-classification tool, so the inferred loss is
`Logloss` and `subsample` works at the default — but setting `loss_function: MultiClass`,
which is legal even on a two-class target, flips the scheme and fails:

    CatBoostError: default bootstrap type is Bayesian, which does not support subsample

QBioCode pins `bootstrap_type` for you as soon as you name either parameter —
`Bernoulli` for `subsample`, `Bayesian` for `bagging_temperature` — so the behaviour no
longer depends on the loss. Naming *both*, or searching `bootstrap_type` across values
that contradict the one you named, is rejected before the search starts with a message
naming the config key. The shipped block above simply stays clear of the area.

`loss_function: MultiClass` also makes CatBoost's `predict()` return an `(n, 1)` column
rather than a flat array. QBioCode flattens it, so the stored predictions keep the same
shape as every other model's — scikit-learn scores a column vector correctly either way,
so this affects the results frame rather than the metrics.
```

**Example: TabPFN**

TabPFN needs only the optional extra — no API key, no license acceptance:

```bash
pip install "qbiocode[tabpfn]"
```

QBioCode pins `model_version: v2`, whose weights are published under the Prior Labs
License (Apache 2.0 plus an attribution clause) and download anonymously on first fit.

```{warning}
**The model version is a licensing choice.** TabPFN's *code* is Apache 2.0 plus
attribution, but its *weights* are licensed per version and the regimes differ sharply:

| `model_version` | Weights license | Commercial use |
| --- | --- | --- |
| `v2` (default) | Prior Labs License v1.1 (Apache 2.0 + attribution) | **Permitted** |
| `v2.5` | TABPFN-2.5 Non-Commercial License | No |
| `v2.6` | TABPFN-2.6 Non-Commercial License | No |
| `v3` | TABPFN-3 Non-Commercial License | No |

The three newest are **non-production as well as non-commercial**: their license permits
testing, evaluation, internal benchmarking and academic research, but not revenue-generating
activity, production systems, or training other models for commercial use. They also require
accepting that license against a Prior Labs account, which upstream does interactively — so
they cannot be fetched unattended, and an API key alone is not enough. Setting
`model_version` to one of them warns and names the license.

`v2` is the default precisely because QBioCode is Apache-2.0 software whose users include
companies; a default that quietly imposed a non-commercial license on them would be the
wrong default whatever its accuracy.
```

Only if you opt into a restricted version do you need an API key. There are two ways to
supply one, and the first is preferred:

```yaml
# The key lives in a file OUTSIDE the repository, so it cannot be committed.
tabpfn_json_path: '~/.config/qbiocode/tabpfn.json'
```

Create that file with the correct permissions rather than by hand:

```bash
python -c "from qbiocode.utils import write_token_template; print(write_token_template())"
# -> ~/.config/qbiocode/tabpfn.json, created mode 0600
```

then paste the key into its `token` field:

```json
{
  "token": "<your api key>"
}
```

QProfiler reads it automatically whenever `tabpfn` is in the `model` list. Outside
QProfiler, call `qbiocode.utils.load_tabpfn_token()` before fitting.

The alternative is to export the variable TabPFN itself reads, which takes precedence over
the file:

```bash
export TABPFN_TOKEN="<your api key>"
```

```{warning}
**Do not put the key in `config.yaml`, or in any file inside the repository.** The
`~/.config/` location is the supported one specifically because it is outside the
checkout: a gitignored file in the tree is *unlikely* to be committed, not unable to be —
`git add -f` overrides the rule, a rewritten `.gitignore` stops covering it, and a copy
made into a sibling clone is not covered at all.

QBioCode never logs or prints the token. `qbiocode.utils.describe_token_source()` reports
*whether* a token is configured and where it came from, with no key material at all — not
even a prefix or fingerprint — because tutorial notebooks are published with their
committed outputs.
```

```yaml
tabpfn_args:
  n_estimators: 4
  softmax_temperature: 0.9
  balance_probabilities: False
  average_before_softmax: False
  device: cpu

gridsearch_tabpfn_args:
  n_estimators: [1, 4, 8]
  softmax_temperature: {low: 0.5, high: 1.5}
  balance_probabilities: [True, False]
  device: cpu
```

```{note}
**Nothing in the TabPFN block is a training hyperparameter.** The weights are frozen and
pretrained; `fit` only memorises the training rows, and every setting above is an
inference knob -- none of them changes model capacity. `n_estimators` buys ensemble
members over differently preprocessed views of the same rows.

Two consequences for tuning. A trial costs `cross_validation` full transformer forward
passes rather than five cheap tree fits, so keep `gridsearch_tabpfn_args` small. And
`device` is pinned to `cpu` above deliberately: MPS and CUDA are not numerically
identical to CPU, which would make a benchmark irreproducible across machines.

TabPFN also supports **at most 10 classes**. Unlike its row and feature limits, that one
cannot be waived with `ignore_pretraining_limits`; a dataset with more is rejected up
front, naming its class count.
```

```{seealso}
For detailed parameter descriptions, see the upstream documentation:
- [SVC Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Random Forest Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [CatBoost Training Parameters](https://catboost.ai/docs/en/references/training-parameters/common)
- [TabPFN](https://github.com/PriorLabs/TabPFN)
```

### Quantum Model Hyperparameters

For quantum models, hyperparameter tuning requires generating separate config files for each combination.

```{important}
**QML Grid Search:**

Quantum model grid search is handled differently than classical models. Use the `generate_experiments.ipynb` notebook in `archive/tutorial_notebooks/qml_experiment_generators/` to generate individual config files for each parameter combination.

This approach is necessary because:
1. Quantum jobs are submitted to IBM Quantum queue
2. Each configuration may take hours to complete
3. Separate configs allow parallel job submission
```

**Example: Quantum SVC (QSVC)**

```yaml
qsvc_args:
  feature_map: 'ZZFeatureMap'
  reps: 2
  entanglement: 'linear'
```

**Example: Variational Quantum Classifier (VQC)**

```yaml
vqc_args:
  feature_map: 'ZZFeatureMap'
  ansatz: 'RealAmplitudes'
  reps: 3
  optimizer: 'COBYLA'
```

---

## Complete Example Configuration

Here's a comprehensive example combining all sections:

```yaml
# Experiment identification
config_file_name: 'comprehensive_experiment'

# Input data
folder_path: 'datasets/'
file_dataset: ['cancer_data.csv', 'diabetes_data.csv']
output_dir: 'results/experiment_001/'

# Reproducibility
seed: 42
q_seed: 42

# Quantum backend - Noisy simulation example
backend: 'noisy_ibm_cleveland'
sim_method: 'matrix_product_state'
shots: 1024
resil_level: 1
qiskit_json_path: '~/.qiskit/qiskit-ibm.json'
name: 'my_ibm_account'

# Dimensionality reduction
embeddings: ['none', 'pca']
n_components: 5

# Data splitting
test_size: 0.2
stratify: ['y']
scaling: ['True']

# Models to evaluate
model: ['rf', 'svc', 'mlp', 'xgb', 'catboost', 'qsvc', 'pqk']

# Classical model parameters
rf_args:
  n_estimators: 100
  max_depth: 10

gridsearch_rf_args:
  n_estimators: [50, 100, 200]
  max_depth: [5, 10, 15]

svc_args:
  C: 1.0
  kernel: 'rbf'

gridsearch_svc_args:
  C: [0.1, 1, 10]
  kernel: ['linear', 'rbf']

# Quantum model parameters
qsvc_args:
  feature_map: 'ZZFeatureMap'
  reps: 2
```

**Alternative Backend Configurations:**

```yaml
# For exact noiseless simulation
backend: 'simulator'

# For AerSimulator with GPU acceleration
backend: 'simulator_aer'
sim_method: 'tensor_network'

# For actual IBM Quantum hardware
backend: 'ibm_kyoto'
shots: 4096
resil_level: 2
```

---

## Best Practices

```{tip}
**Configuration Tips:**

1. **Start Simple**: Begin with a minimal config and add complexity gradually
2. **Use Descriptive Names**: Name configs by experiment purpose (e.g., `cancer_baseline.yaml`)
3. **Version Control**: Keep configs in git to track experiment history
4. **Document Changes**: Add comments in YAML to explain non-obvious choices
5. **Test Locally First**: Use `backend: 'simulator'` before submitting to quantum hardware
```

```{warning}
**Common Pitfalls:**

- **Missing Seeds**: Always set `seed` and `q_seed` for reproducibility
- **Too Many Combinations Under `tuner: grid`**: the exhaustive sweep fits every
  combination; start small to estimate runtime, or use the default `tuner: optuna`
```

---

## Troubleshooting

**Problem: "Config file not found"**
- Ensure config file is in `configs/` directory
- Check file name matches `--config-name` argument
- Use relative path from project root

**Problem: "Invalid backend"**
- Verify IBM Quantum credentials are configured
- Check device name spelling (use `ibm_<device>` format)
- Ensure you have access to the specified instance

**Problem: "Hyperparameter tuning taking too long"**
- Lower `n_trials` (the Optuna tuner's budget maps directly onto fits)
- Use fewer cross-validation folds
- If you set `tuner: grid`, the cost is the *whole* cross product regardless of
  `n_trials` -- switch back to `tuner: optuna` unless you specifically need the
  exhaustive sweep to reproduce an older result

**Problem: "Out of memory"**
- Reduce `n_components` for embeddings
- Use smaller `test_size` to reduce data size
- Process datasets one at a time instead of batch

---

## See Also

- :doc:`QProfiler Usage Guide <profiler>` - How to run QProfiler
- :doc:`QSage Configuration <sage>` - Meta-learning model selection
- :doc:`Tutorial Notebooks <../tutorials>` - Step-by-step examples
