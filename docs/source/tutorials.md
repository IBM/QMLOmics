# Tutorials

Welcome to the QBioCode tutorials! These Jupyter notebooks provide hands-on examples demonstrating how to use various features and applications of QBioCode for quantum healthcare and life sciences applications.

## Getting Started

Before running these tutorials, make sure you have:
- Installed QBioCode following the [Installation Guide](installation.md)
- Set up your Python environment with all required dependencies
- Access to quantum computing resources (if running quantum algorithms)

---

## Tutorial Gallery

### 1. Artificial Data Generation

Learn how to generate synthetic datasets for testing and benchmarking quantum machine learning algorithms.

<a href="tutorials/Artificial_data_generation/example_data_generation.html">📓 <strong>View Tutorial Notebook</strong></a>

---

### 2. QProfiler - Automated ML Model Benchmarking

Learn how to use QProfiler to systematically benchmark and compare quantum and classical machine learning models on artificial datasets. This tutorial demonstrates:

<a href="tutorials/QProfiler/example_qprofiler.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Generate artificial datasets with specific characteristics
- Configure and run QProfiler experiments via YAML configuration
- Evaluate multiple ML models (quantum and classical) automatically
- Analyze performance metrics (accuracy, F1-score, AUC)
- Visualize model comparisons and correlations
- Interpret results for model selection

#### QProfiler on Single-Cell Data - Binary Task Prediction (with PQK)

Apply QProfiler to a real single-cell RNA-seq benchmark: **CD4 vs CD8 T-cell** classification from PBMC data. This tutorial benchmarks classical baselines against the **projected quantum kernel (PQK)** on the one non-trivial binary task, and shows how to read data-complexity measures to explain the result.

<a href="tutorials/QProfiler/sc_binary_qprofiler.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Build per-task CSV datasets from balanced PBMC `h5ad` files (leakage-safe HVG selection)
- Add and tune a PQK feature map (shallow, linearly-entangled ZZ) to avoid quantum-kernel concentration
- Quantify the quantum-vs-classical gap with a paired Cohen's *d_z* (PQK vs classical)
- Correlate data-complexity measures with that gap
- Use complexity context (Fisher ratio, mutual information, silhouette) to explain task difficulty

---

### 3. QSage - Quantum-Inspired Feature Importance

Explore QSage, an intelligent meta-learning system that predicts which machine learning models will perform best on your dataset *before* you run them. By learning from data complexity patterns across multiple datasets, QSage provides data-driven model recommendations. This tutorial shows how to:

<a href="tutorials/QSage/qsage.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Load pre-trained QSage models
- Analyze dataset characteristics (intrinsic dimension, Fisher discriminant ratio, etc.) from compiled ML benchmark results
- Apply QSAGE to predict the model

---

### 4. QuVINE - Quantum View-based Network Embeddings

QuVINE embeds the **nodes of a graph** using classical and quantum random walks combined with
SGNS-based representation learning. Where QProfiler characterizes a tabular dataset, QuVINE works on
a different modality - a graph - and returns a node-embedding matrix, with `evaluate_graph` as the
graph analogue of QProfiler's dataset summary.

QuVINE's dependencies ship behind an optional extra, so install it with
`pip install "qbiocode[quvine]"` before running these notebooks.

The graph summary these notebooks lean on is documented in full under
{ref}`Graph-Complexity Measures <graph-complexity-measures>`, and the YAML config
they configure QuVINE with in the {doc}`QuVINE Configuration Guide
<apps/quvine_config>`.

Four notebooks, in reading order: the synthetic-graph walkthrough introduces the API and the method
registry, two single-cell notebooks apply it to real data, and the last drives QuVINE through
QProfiler.

#### Getting Started - 12 Embedding Methods on Synthetic Graphs

Benchmark QuVINE's quantum walks, its quantum-calibrated neural baselines, and classical baselines on
three stochastic block models of increasing difficulty, then correlate each graph's complexity against
node-classification performance. Fully synthetic - no data files needed.

<a href="tutorials/QuVINE/example_quvine.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Turn a `networkx.Graph` into an embedding matrix with `qbiocode.apps.quvine.embed`
- Score node classification across 12 methods × 3 graphs × 5 iterations
- Summarize a graph with `evaluate_graph` (~80 spectral, topological, and structural metrics)
- Correlate complexity (spectral gap, IPR, spectral degeneracy, entropy) with macro-F1
- Reproduce the same embedding from the `quvine` console script

**Key Concepts:**
- Planted-community stochastic block models, with `p_out` as the difficulty knob
- Quantum-calibrated spectral filters, GAT, and GraphGPS variants alongside RWR/CTQW/DTQW walks
- Graph complexity as a predictor of embedding quality

#### Single-Cell Application - CD4 vs. CD8

Build a multi-view graph from single-cell data and compare classical against quantum-calibrated
embeddings on a downstream classification task.

<a href="tutorials/QuVINE/quvine_sc_cd4_vs_cd8.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Build a graph from single-cell data and inspect it with `evaluate_graph`
- Select among QuVINE's 83 embedding methods with a single `method` string
- Run classical (node2vec, NetMF, APPNP) and quantum-calibrated walk embeddings
- Fuse multiple graph views into a single embedding
- Compare classical vs. quantum embeddings on a downstream classification task

**Key Concepts:**
- Multi-view graph construction
- Random walk with restart (RWR) and discrete-/continuous-time quantum walks
- Skip-gram negative sampling (SGNS) embedding learning
- Reproducible, seed-controlled embedding pipelines

#### Semi-Supervised Node Classification and Ranking - T vs. Monocyte

A transductive task on an 800-cell graph with two views (RNA and protein) and soft seed labels.
QuVINE is evaluated the two ways it natively supports - training a classifier on the seed nodes'
embeddings, and ranking non-seed nodes by similarity to the seeds - each against honest baselines.

<a href="tutorials/QuVINE/quvine_sc_t_vs_mono.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Embed two view-graphs separately and as an early-fusion concatenation
- Compare embedding arms against a no-embedding label-spreading baseline on the same graph
- Rank nodes by seed similarity with `seed_centroid_scores` and score recall@k / precision@k
- Read `SeedTargetEvaluator`'s degree- and distance-matched null controls
- Use spectral gap and modularity per view to explain which view diffusion helps

**Key Concepts:**
- Transductive, semi-supervised node problems with soft seeds
- Cross-modality fusion versus QuVINE's internal fusion across walk kinds
- Null controls that separate real recovery from a preference for hubs or near neighbors
- 95% confidence intervals: overlapping CIs mean no reproducible difference at this scale

#### QuVINE as a QProfiler Embedding - 2×2 Classical/Quantum Comparison

Drive QuVINE through `qbiocode.get_embeddings` exactly as you would `pca` or `umap`, and let
QProfiler benchmark the resulting embeddings. The 2×2 design crosses a classical and a quantum
walk with a classical and a quantum learner on the CD4 vs. CD8 task.

<a href="tutorials/QProfiler/sc_binary_quvine_2x2_qprofiler.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Call a `quvine_*` method through the same `get_embeddings` entry point as the classical embeddings
- Understand why graph embeddings are *transductive* - test features join graph construction, test labels never do
- Read the `UserWarning` QBioCode emits to make that transductivity explicit
- Compare quantum-walk against classical-walk embeddings under an identical learner

---

### 5. Quantum Ensemble Learning

Learn how to use quantum ensemble methods to improve classification performance by leveraging quantum superposition to evaluate multiple training set configurations simultaneously. This tutorial demonstrates two quantum ensemble approaches.

<a href="tutorials/QEnsemble/QEnsemble_example_blobs.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Generate blob datasets for binary classification
- Implement fixed swap-based quantum ensemble method
- Implement random unitary-based quantum ensemble method
- Use quantum SWAP test for cosine similarity measurement
- Compare quantum ensemble with classical baselines (Random Forest, XGBoost)
- Evaluate performance using accuracy and Brier score metrics
- Understand quantum superposition for ensemble learning

**Key Concepts:**
- Quantum ensemble learning via superposition
- SWAP test for quantum state comparison
- Controlled-SWAP operations for deterministic data rearrangement
- Haar-random unitaries for general mixing
- One-hot encoding for quantum state preparation
- Quantum ensembling of multiple classifiers

**Methods:**
1. **Swap Method**: Uses fixed controlled-SWAP operations to create deterministic permutations of training data
2. **Random Unitary Method**: Applies Haar-random unitary transformations for more general data mixing

**References:**
- Macaluso et al. (2023) - "A variational algorithm for quantum neural networks"
- Rhrissorrakrai et al. (2025) - "Quantum Ensemble Learning" (arXiv:2506.02213)

---

### 6. Quantum Projection Learning (QPL)

Learn about Quantum Projection Learning (QPL), a technique that combines quantum feature maps with multiple classical machine learning algorithms. This comprehensive tutorial demonstrates how to systematically evaluate quantum-enhanced features across different learners.

<a href="tutorials/Quantum_Projection_Learning/QPL_example.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Generate synthetic datasets with controlled complexity
- Apply quantum feature maps to create quantum projections
- Train multiple classical models (SVC, RF, XGBoost, MLP, LR) on quantum features
- Compare quantum-enhanced vs. classical baseline performance
- Visualize and analyze comprehensive performance metrics
- Use QProfiler for automated QPL experiments

**Key Concepts:**
- Quantum projection methods and expectation value measurements
- Ensemble learning with quantum features
- Data complexity analysis to predict where quantum features help
- Systematic model comparison and evaluation
- Integration with classical ML pipelines

**Workflow:**
1. Generate or load classification datasets
2. Configure QPL experiments via YAML files
3. Apply quantum feature maps (ZZ, Pauli, etc.)
4. Extract quantum projections from circuits
5. Train 5+ classical models on quantum features
6. Compare with classical baselines
7. Analyze results and identify where quantum features help

---

### 7. Projected Quantum Kernel (PQK) - Ovarian Cancer Survival Prediction

Learn how to apply Projected Quantum Kernels (PQK) to real-world cancer genomics data for survival prediction. This advanced tutorial demonstrates quantum-enhanced machine learning on multi-omics ovarian cancer data from the Multi-Omics Cancer Benchmark (TCGA preprocessed data).

<a href="tutorials/PQK%20-%20OV.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Automatically download and process multi-omics cancer data
- Create 3-year survival labels from clinical data
- Apply quantum feature maps to high-dimensional genomics data
- Use PQK to create quantum feature representations
- Compare quantum-enhanced vs. classical SVM performance
- Work with multi-omics data (miRNA, methylation, gene expression)
- Perform comprehensive hyperparameter tuning for quantum kernels
- Evaluate quantum performance on real biomedical datasets

**Dataset:**
- Ovarian cancer (OV) multi-omics data from [Multi-Omics Cancer Benchmark](https://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html)
- TCGA preprocessed data with automatic download
- 3-year survival prediction task
- Four data modalities: miRNA, DNA methylation, gene expression, and integrated

**Key Techniques:**
- Automated data download and preprocessing pipeline
- Patient ID standardization across multi-omics datasets
- Survival label creation from clinical data
- Quantum kernel methods with ZZ feature maps
- Pairwise qubit entanglement strategies
- PCA dimensionality reduction for quantum encoding
- Stratified cross-validation for robust evaluation

---

## Additional Resources

- [API Documentation](api_overview.rst) - Detailed API reference
- [QProfiler App](apps/profiler.rst) - Standalone profiling application
- [QSage App](apps/sage.rst) - Feature selection application
- [QuVINE App](apps/quvine.rst) - Quantum view-based network embeddings
- [QuVINE Configuration Guide](apps/quvine_config.md) - The shipped QuVINE YAML config, section by section
- [GitHub Repository](https://github.com/qiskit-community/QBioCode) - Source code and examples

## Support

If you encounter any issues or have questions about the tutorials:
- Check the [GitHub Issues](https://github.com/qiskit-community/QBioCode/issues)
- Review the [Contributing Guide](https://github.com/qiskit-community/QBioCode/blob/main/CONTRIBUTING.md)
- Consult the API documentation for detailed function references

---

### 8. Hyperparameter Tuning - Optuna vs Grid Search

Learn how QBioCode tunes hyperparameters, and what changed when Optuna replaced the exhaustive grid as the default. This tutorial times the two searches against each other on the shipped configuration blocks, then tunes a quantum classifier.

<a href="tutorials/Hyperparameter_Tuning/optuna_vs_gridsearch.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Why the shipped `gridsearch_rf_args` block costs 2,880 fits at `cross_validation: 5`
- Run the same configuration under `tuner: grid` and `tuner: optuna`, timed side by side
- Write a hyperparameter as a continuous `{low, high, log}` range instead of a list of decades
- Recognise the error you get when a range meets `tuner: grid`
- Tune a quantum classifier (`qsvc`) with `tune_quantum: True`
- Budget a quantum search: why `n_trials_quantum` defaults to 10, not 50

**Key Concepts:**
- TPE sampling against exhaustive enumeration: fixed budget versus fixed grid
- Categorical choices and continuous ranges in one configuration block
- Why quantum candidates are scored on one holdout rather than k folds
- Reproducing a pre-Optuna result with `tuner: grid`
- The hardware guard: why tuning on a real device is refused by default

---

### 9. CatBoost and TabPFN - Two New Classical Learners

Run the two classical learners added alongside XGBoost, and see the three ways they differ from the scikit-learn estimators around them. CatBoost is a core dependency and works everywhere; TabPFN is a pretrained transformer behind an optional extra, and this tutorial degrades gracefully when its weights are unavailable rather than failing.

<a href="tutorials/CatBoost_and_TabPFN/catboost_and_tabpfn.html">📓 <strong>View Tutorial Notebook</strong></a>

**What You'll Learn:**
- Run `catboost` beside `xgb` and `rf`, untuned and then tuned with Optuna
- Why two boosters are worth running rather than picking one in advance
- Recognise the one real trap in CatBoost's hyperparameter surface: `subsample` and `bagging_temperature` belong to incompatible bootstrap schemes
- See why that conflict is reachable on *binary* data through `loss_function`, not only by adding classes
- Measure a hyperparameter that silently does nothing (`min_data_in_leaf` at the default grow policy)
- Understand what TabPFN actually needs to run: neither a GPU nor an API key
- See why the pinned model version is a licensing decision, not just a modelling one: `v2`'s weights permit commercial use, `v2.5`/`v2.6`/`v3` are non-commercial

**Key Concepts:**
- Gradient boosting variants: symmetric trees and ordered boosting against XGBoost's level-wise growth
- In-context learning: a frozen pretrained model whose `fit` only memorises rows, so no hyperparameter changes its capacity
- Configuration validity that depends on a derived default rather than on what you wrote
- Optional extras with a manual gate, and why such a model cannot be a base dependency
- TabPFN's fixed ceilings: 10 classes (unwaivable), 50,000 rows, 2,000 features

```{toctree}
:hidden:
:maxdepth: 1

Artificial Data Generation <tutorials/Artificial_data_generation/example_data_generation>
Single-Cell Preprocessing & QC <tutorials/Preprocessing/sc-qc>
QProfiler <tutorials/QProfiler/example_qprofiler>
QProfiler on Single-Cell Data <tutorials/QProfiler/sc_binary_qprofiler>
QuVINE - Getting Started <tutorials/QuVINE/example_quvine>
QuVINE on Single-Cell Data <tutorials/QuVINE/quvine_sc_cd4_vs_cd8>
QuVINE on T vs. Monocyte <tutorials/QuVINE/quvine_sc_t_vs_mono>
QuVINE Embeddings in QProfiler <tutorials/QProfiler/sc_binary_quvine_2x2_qprofiler>
Quantum Ensemble Learning <tutorials/QEnsemble/QEnsemble_example_blobs>
QSage <tutorials/QSage/qsage>
Quantum Projection Learning <tutorials/Quantum_Projection_Learning/QPL_example>
PQK on Ovarian Cancer <tutorials/PQK - OV>
Hyperparameter Tuning: Optuna vs Grid Search <tutorials/Hyperparameter_Tuning/optuna_vs_gridsearch>
CatBoost and TabPFN <tutorials/CatBoost_and_TabPFN/catboost_and_tabpfn>
```
