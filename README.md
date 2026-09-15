# QBioCode

**A comprehensive suite of computational resources for quantum machine learning applications in healthcare and life sciences.**

[![PyPI version](https://badge.fury.io/py/qbiocode.svg)](https://badge.fury.io/py/qbiocode) [![Minimum Python Version](https://img.shields.io/badge/Python-%3E=%203.10-blue)](https://www.python.org/downloads/) [![Maximum Python Version Tested](https://img.shields.io/badge/Python-%3C=%203.12-blueviolet)](https://www.python.org/downloads/) [![Supported Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/) [![GitHub Pages](https://img.shields.io/badge/docs-sphinx-blue)](https://qiskit-community.github.io/QBioCode/)

<img src="docs/source/img/QBioCode_logo.png" width="300" />

QBioCode provides tools for benchmarking quantum and classical machine learning models, analyzing data complexity, and making informed model selection decisions for healthcare and life science applications.

## 🌟 Key Features

- **QProfiler**: Automated ML benchmarking with data complexity analysis
- **QSage**: Meta-learning tool for intelligent model selection
- **Data Generation**: Create artificial datasets with controlled complexity
- **Quantum ML Support**: QSVC, PQK, VQC, QNN, Quantum Ensemble implementations
- **Classical ML Baselines**: RF, SVM, LR, DT, NB, MLP, XGBoost, CatBoost, and TabPFN (a pretrained tabular transformer, via the optional `[tabpfn]` extra)
- **Comprehensive Documentation**: Detailed tutorials and API reference

## 📋 Requirements

QBioCode requires Python **3.10 or higher** and has been tested with Python versions **3.10, 3.11, and 3.12**.

**Note:** Python 3.10+ is required for compatibility with the latest Qiskit ecosystem (qiskit-ibm-runtime 0.44.0+).

## 🚀 Quick Start

### Installation

#### Install from PyPI (Recommended)

```bash
# Install the latest stable version
pip install qbiocode

# Install with apps support (QProfiler, QSage)
pip install 'qbiocode[apps]'

# Install with QuVINE graph embeddings (quvine_rwr, quvine_dtqw, node2vec, ...)
pip install 'qbiocode[quvine]'

# Install with all optional dependencies
pip install 'qbiocode[all]'
```

<details>
<summary><b>Optional dependency extras</b></summary>

A plain `pip install qbiocode` gives you the full classical and quantum pipeline.
Everything below is additive, and extras combine (`'qbiocode[apps,quvine]'`).

| Extra | Command | What it adds |
| --- | --- | --- |
| *(none)* | `pip install qbiocode` | Core library: embeddings (`pca`, `nmf`, `umap`, `tsne`, `spectral`, ...), PQK, classical + quantum models, `evaluate_graph`, `scale_train_test` |
| `apps` | `pip install 'qbiocode[apps]'` | Hydra-driven CLIs for the QProfiler and QSage apps |
| `quvine` | `pip install 'qbiocode[quvine]'` | QuVINE quantum/classical graph embeddings — 83 methods via `get_embeddings("quvine_*", ...)` |
| `docs` | `pip install 'qbiocode[docs]'` | Sphinx toolchain for building the documentation |
| `dev` | `pip install 'qbiocode[dev]'` | `pytest`, `pytest-cov`, `black`, `isort`, `flake8`, `mypy` |
| `all` | `pip install 'qbiocode[all]'` | Union of every extra above |

QuVINE is deliberately one all-or-nothing extra: its dependencies overlap across
the walk, spectral and neural method families, so a partial install would leave
most method names resolving and a few raising at call time. Without it,
`import qbiocode` and all classical embeddings still work — only the
`quvine_*` names raise, with a message naming the missing module and the exact
install command. See
[Installation](https://qiskit-community.github.io/QBioCode/installation.html) for details.

</details>

#### Install with Conda

QBioCode will be available on conda-forge and bioconda after the initial release review process.

```bash
# Once available on conda-forge (recommended)
conda install -c conda-forge qbiocode

# Or from bioconda (includes bioinformatics dependencies)
conda install -c bioconda -c conda-forge qbiocode

# Create a new environment with qbiocode
conda create -n qbiocode -c conda-forge qbiocode
conda activate qbiocode
```

**Current Status**: Conda packages are pending submission to conda-forge and bioconda. In the meantime, use pip within a conda environment:

```bash
conda create -n qbiocode python=3.10
conda activate qbiocode
pip install qbiocode
```

#### Install from Source

```bash
# Clone the repository
git clone https://github.com/qiskit-community/QBioCode.git
cd QBioCode

# Create virtual environment
python -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate

# Install QBioCode in editable mode
pip install -e .

# Install with apps support (QProfiler, QSage)
pip install -e '.[apps]'

# Install with QuVINE graph embeddings
pip install -e '.[quvine]'
```

**macOS Users:** XGBoost requires OpenMP. Install it using Homebrew:
```bash
brew install libomp
pip install --force-reinstall xgboost
```

For detailed installation instructions, see the [Installation Guide](https://qiskit-community.github.io/QBioCode/installation.html).

### Running Tests

```bash
# Install the package with development dependencies
pip install -e '.[dev]'

# Run the test suite
python -m pytest
```

The current test suite focuses on utility modules and data-generation helpers that do not require a full runtime setup for all optional quantum workflows.

### Basic Usage

```python
import qbiocode as qbc

# Generate artificial data
qbc.generate_data(
    type_of_data='moons',
    save_path='data/moons',
    n_samples=[100, 200],
    noise=[0.1, 0.2],
    random_state=42
)

# Run QProfiler
from qbiocode.apps.qprofiler import qprofiler
import yaml

config = yaml.safe_load(open('configs/config.yaml'))
qprofiler.main(config)
```

## 📚 Applications

### QProfiler

**Automated ML Benchmarking with Data Complexity Analysis**

QProfiler provides a comprehensive benchmarking pipeline that:
- Evaluates both classical and quantum ML models
- Computes 125 data complexity metrics (pyMFE + QBioCode-native)
- Correlates model performance with data characteristics
- Generates detailed performance reports and visualizations

**Before you run — the input data must exist.** QProfiler reads a folder of CSV
datasets. The `folder_path` in the config is resolved **relative to the
`QBioCode` repo root** (the code truncates your current directory at `QBioCode`
and joins `folder_path` to it). So:

1. Run the command from **inside the QBioCode repo tree** (your working directory
   path must contain `QBioCode`).
2. The data must already be present at `<QBioCode-repo-root>/<folder_path>`.
   QProfiler does **not** create it. The tutorial dataset already ships in the
   repo at `tutorial/QProfiler/data/ld_data`.

**Usage:**
```bash
# Always run from inside the QBioCode repo so folder_path resolves correctly
cd /path/to/QBioCode

# 1) Run with the bundled default config
#    (expects data at <repo>/tutorial_test_data/lower_dim_datasets)
qprofiler

# 2) Point at the tutorial config + bundled data
#    (folder_path in this config = tutorial/QProfiler/data/ld_data)
qprofiler --config-dir=tutorial/QProfiler/configs --config-name=config

# 3) Override any value inline (Hydra syntax — key=value, no leading --)
#    folder_path is relative to the QBioCode repo root
qprofiler folder_path=tutorial/QProfiler/data/ld_data file_dataset=ALL backend=simulator
```

Results are written to `results/<config_file_name>/dataset=<file_dataset>/<backend>_<timestamp>/`
under the repo root. Run `qprofiler --help` to see every overridable key.

```python
# Python API
from qbiocode.apps.qprofiler import qprofiler
qprofiler.main(config)
```

[📖 QProfiler Documentation](https://qiskit-community.github.io/QBioCode/apps/profiler.html) | [📓 Tutorial](tutorial/QProfiler/example_qprofiler.ipynb)

### QSage

**Intelligent Model Selection via Meta-Learning**

QSage uses surrogate models trained on extensive benchmarking data to:
- Predict model performance without running experiments
- Recommend best models based on dataset characteristics
- Save computational resources
- Provide interpretable predictions

**Usage:**
```bash
# Command line
qsage --data your_data.csv --output predictions.csv

# Python API
from qbiocode.apps.sage.sage import QuantumSage
sage = QuantumSage(data=benchmark_df, features=features, metrics=metrics)
predictions = sage.predict(new_dataset_features)
```

[📖 QSage Documentation](https://qiskit-community.github.io/QBioCode/apps/sage.html) | [📓 Tutorial](tutorial/QSage/qsage.ipynb)

### QuVINE

**Quantum View-based Network Embeddings**

QuVINE embeds the **nodes of a graph** — a different modality from QProfiler's
tabular datasets — by combining classical and quantum random walks with
SGNS-based representation learning:
- Multi-view graph construction from a single input graph
- Random walk with restart (RWR) plus discrete- and continuous-time quantum walks
- Quantum-calibrated filter / GAT / GraphGPS variants, and classical baselines
  (node2vec, NetMF, APPNP) for comparison
- 83 named methods selectable with a single `method` string
- Usable through `qbiocode.get_embeddings` alongside `pca`, `nmf` and `umap`

**Installation.** QuVINE's dependencies (gensim, hiperwalk, node2vec,
torch-geometric, python-louvain, ripser, omegaconf) are heavy, so they sit behind
an optional extra:

```bash
pip install "qbiocode[quvine]"
```

A plain `pip install qbiocode` still imports fine; requesting a QuVINE method
then raises a `QuvineDependencyError` naming the extra and the missing module.

**Usage:**
```bash
# List every available method (unavailable ones are reported, not hidden)
quvine --list-methods

# Embed a graph given as a 2- or 3-column edge list
quvine --edgelist edges.csv --method quvine_fused --output out/

# A classical baseline on a tab-separated, weighted edge list
quvine --edgelist edges.tsv --sep '\t' --weighted --method node2vec --output out/
```

Writes `embedding.csv` (index `node`, columns `dim_0`…`dim_{d-1}`),
`embedding_meta.json`, and `embedding.npy` with `--npy`.

```python
# Python API
import networkx as nx
from qbiocode.apps.quvine import embed

G = nx.karate_club_graph()
result = embed(G, "quvine_fused", base_seed=0)
print(result.embedding.shape)          # (n_nodes, dim)
```

```python
# As a QBioCode embedding, anywhere pca/nmf/umap works — including
# QProfiler's `embeddings:` config list. Note: QuVINE methods are
# transductive; get_embeddings emits a UserWarning saying so.
from qbiocode import get_embeddings
X_train_emb, X_test_emb = get_embeddings("quvine_rwr", X_train, X_test, n_components=8)
```

Graph-complexity metrics are deliberately *not* part of the app — `evaluate_graph`
is core QBioCode and needs only the base install:

```python
from qbiocode import evaluate_graph
metrics = evaluate_graph(G, name="karate")   # 1 x 88 DataFrame
```

[📖 QuVINE Documentation](https://qiskit-community.github.io/QBioCode/apps/quvine.html) | [⚙️ Configuration Guide](https://qiskit-community.github.io/QBioCode/apps/quvine_config.html) | [📊 Graph-Complexity Measures](https://qiskit-community.github.io/QBioCode/apps/quvine.html#graph-complexity-measures) | [📓 Tutorial](tutorial/QuVINE/example_quvine.ipynb)

## 📖 Tutorials

Comprehensive Jupyter notebook tutorials are available:

### 1. [Artificial Data Generation](tutorial/Artificial_data_generation/example_data_generation.ipynb)
Learn how to create synthetic datasets with controlled properties:
- 2D manifolds (circles, moons, spirals)
- 3D manifolds (swiss_roll, s_curve, spheres)
- High-dimensional classification data
- Customizable complexity parameters

### 2. [Single-Cell Preprocessing & QC](tutorial/Preprocessing/sc-qc.ipynb)
The provenance notebook — it generates the balanced PBMC `h5ad` fixtures that the
single-cell notebooks below read:
- Quality control and filtering of raw single-cell data
- Leakage-safe highly-variable-gene selection
- Building balanced per-task subsets

### 3. [QProfiler Tutorial](tutorial/QProfiler/example_qprofiler.ipynb)
Step-by-step guide to benchmarking ML models:
- Data generation and preparation
- Configuration setup
- Running QProfiler
- Analyzing results and visualizations
- Understanding data complexity metrics

### 4. [QProfiler on Single-Cell Data](tutorial/QProfiler/sc_binary_qprofiler.ipynb)
QProfiler on a real benchmark — **CD4 vs CD8** T-cell classification from PBMC data:
- Benchmark classical baselines against the projected quantum kernel (PQK)
- Tune a shallow, linearly-entangled ZZ feature map to avoid kernel concentration
- Quantify the quantum-vs-classical gap with a paired Cohen's *d_z*
- Explain task difficulty from data-complexity measures

### 5. [QuVINE — Getting Started](tutorial/QuVINE/example_quvine.ipynb)
12 embedding methods on synthetic graphs — fully self-contained, no data files:
- Turn a `networkx.Graph` into an embedding matrix with `qbiocode.apps.quvine.embed`
- Score node classification across 12 methods × 3 stochastic block models × 5 iterations
- Summarize a graph with `evaluate_graph` (88 spectral, topological and structural metrics)
- Correlate complexity (spectral gap, IPR, spectral degeneracy, entropy) with macro-F1

### 6. [QuVINE on Single-Cell Data](tutorial/QuVINE/quvine_sc_cd4_vs_cd8.ipynb)
Multi-view graph embeddings on the CD4 vs CD8 task:
- Build a graph from single-cell data and inspect it with `evaluate_graph`
- Run classical (node2vec, NetMF, APPNP) and quantum-calibrated walk embeddings
- Fuse multiple graph views into a single embedding
- Compare classical vs. quantum embeddings on a downstream classification task

### 7. [QuVINE on T vs. Monocyte](tutorial/QuVINE/quvine_sc_t_vs_mono.ipynb)
A transductive, semi-supervised task on an 800-cell two-view graph with soft seeds:
- Embed two view-graphs separately and as an early-fusion concatenation
- Compare against a no-embedding label-spreading baseline on the same graph
- Rank nodes by seed similarity and score recall@k / precision@k
- Read degree- and distance-matched null controls that separate real recovery
  from a preference for hubs

### 8. [QuVINE Embeddings in QProfiler](tutorial/QProfiler/sc_binary_quvine_2x2_qprofiler.ipynb)
Drive QuVINE through `qbiocode.get_embeddings` like any other embedding:
- A 2×2 design crossing a classical and a quantum walk with a classical and a
  quantum learner
- Why graph embeddings are *transductive* — test features join graph
  construction, test labels never do

### 9. [Quantum Ensemble Learning](tutorial/QEnsemble/QEnsemble_example_blobs.ipynb)
Learn quantum ensemble methods for improved classification:
- Fixed swap-based ensemble approach
- Random unitary-based ensemble approach
- Quantum superposition for evaluating multiple training configurations
- Comparison with classical ensemble methods

### 10. [QSage Tutorial](tutorial/QSage/qsage.ipynb)
Learn to use meta-learning for model selection:
- Loading pre-trained QSage models
- Making predictions on new datasets
- Analyzing prediction accuracy
- Understanding feature importance

### 11. [Quantum Projection Learning](tutorial/Quantum_Projection_Learning/QPL_example.ipynb)
Advanced quantum ML techniques with classical baselines:
- Apply quantum feature maps to create quantum projections
- Train SVC, RF, XGBoost, CatBoost, MLP and LR on quantum features
- Compare quantum-enhanced against classical baselines

### 12. [PQK on Ovarian Cancer](tutorial/PQK%20-%20OV.ipynb)
Projected Quantum Kernels on real multi-omics cancer genomics:
- Automatically download and process TCGA ovarian-cancer multi-omics data
- Create 3-year survival labels from clinical data
- Compare quantum-enhanced against classical SVM performance across four modalities

All twelve are also rendered on the [documentation site](https://qiskit-community.github.io/QBioCode/tutorials.html).

## 🔧 Core Modules

### Data Generation
```python
import qbiocode as qbc

# Generate various dataset types
qbc.generate_data(type_of_data='circles', ...)
qbc.generate_data(type_of_data='moons', ...)
qbc.generate_data(type_of_data='classes', ...)
```

### Machine Learning Models

**Classical Models:**
- Random Forest (RF)
- Support Vector Machine (SVM)
- Logistic Regression (LR)
- Decision Tree (DT)
- Naive Bayes (NB)
- Multi-Layer Perceptron (MLP)
- XGBoost
- CatBoost
- TabPFN — pretrained tabular transformer (needs the `[tabpfn]` extra; no API key or
  license acceptance, as QBioCode pins the commercially-usable `v2` weights)

**Quantum Models:**
- Quantum Support Vector Classifier (QSVC)
- Projected Quantum Kernel (PQK)
- Variational Quantum Classifier (VQC)
- Quantum Neural Network (QNN)
- Quantum Ensemble (QEnsemble) - swap and random unitary methods

**Quantum Ensemble Usage:**
```python
from qbiocode.learning import compute_qensemble
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Run quantum ensemble with swap method
results_swap = compute_qensemble(
    X_train, X_test, y_train, y_test,
    ensemble_method='swap',
    n_ensemble=4,
    seed=42
)

# Run quantum ensemble with random unitary method
results_random = compute_qensemble(
    X_train, X_test, y_train, y_test,
    ensemble_method='random_unitary',
    n_ensemble=4,
    seed=42
)
```

### Embeddings
- PCA, LLE, Isomap, Spectral Embedding
- UMAP, NMF
- Autoencoder

### Evaluation
- Model performance metrics (accuracy, F1, AUC)
- Data complexity analysis
- Correlation studies

## 🛠️ Utilities

### QML Config Generation

Generate configuration files for quantum model hyperparameter tuning:

```python
from qbiocode.utils import generate_qml_experiment_configs

num_configs, used_files = generate_qml_experiment_configs(
    template_config_path='configs/config.yaml',
    output_dir='configs/qml_gridsearch',
    data_dirs=['data/my_datasets'],
    qmethods=['qnn', 'vqc', 'qsvc'],
    reps=[1, 2],
    n_components=[5, 10],
    embeddings=['none', 'pca', 'isomap']
)
```

## 📊 Documentation

Full documentation is available at: **[https://qiskit-community.github.io/QBioCode/](https://qiskit-community.github.io/QBioCode/)**

- [Installation Guide](https://qiskit-community.github.io/QBioCode/installation.html)
- [API Reference](https://qiskit-community.github.io/QBioCode/api/qbiocode.html)
- [Tutorials](https://qiskit-community.github.io/QBioCode/tutorials.html)
- [Background](https://qiskit-community.github.io/QBioCode/background.html)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📝 Citation

If you use QBioCode in your research, please cite:

```bibtex
@software{qbiocode2024,
  title = {QBioCode: Quantum Machine Learning for Healthcare and Life Sciences},
  author = {Raubenolt, Bryan and Bose, Aritra and Rhrissorrakrai, Kahn and 
            Utro, Filippo and Mohan, Akhil and Blankenberg, Daniel and Parida, Laxmi},
  year = {2024},
  url = {https://github.com/qiskit-community/QBioCode}
}
```

See [CITATION.cff](CITATION.cff) for more details.

## 👥 Authors

**Core Contributors:**

- Bryan Raubenolt (raubenb@ccf.org) - Cleveland Clinic
- Aritra Bose (a.bose@ibm.com) - IBM Research
- Kahn Rhrissorrakrai (krhriss@us.ibm.com) - IBM Research
- Filippo Utro (futro@us.ibm.com) - IBM Research
- Akhil Mohan (mohana2@ccf.org) - Cleveland Clinic
- Daniel Blankenberg (blanked2@ccf.org) - Cleveland Clinic
- Laxmi Parida (parida@us.ibm.com) - IBM Research

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on [GitHub](https://github.com/qiskit-community/QBioCode/issues)
- Check the [documentation](https://qiskit-community.github.io/QBioCode/)
- Contact the authors

---

**QBioCode** - Advancing quantum machine learning for healthcare and life sciences 🧬⚛️
