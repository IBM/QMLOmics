# Changelog

All notable changes to QBioCode will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [Unreleased]

### Added

#### QProfiler tutorial v2

- **`tutorial/QProfiler_v2/example_qprofiler_v2.ipynb`**, a second pass over QProfiler built
  so the correlation it ends on is worth reading. It runs **all ten models** -- adding
  `catboost` and `tabpfn` to the v1 notebook's set -- with **Optuna** tuning every classical
  one, over 3 datasets x 5 splits x 2 embeddings, and reports all three dataset-complexity
  blocks: the hand-curated native measures, the `mfe.` pyMFE block, and the new `task.`
  target spectrum. ~5 minutes on a laptop; its own directory, config, data and PQK cache, so
  it shares no output file with the v1 notebook.

- **15 observations per (model, embedding) group, not 6.** The complexity block is recomputed
  on each split's *training* rows, so 3 datasets x 5 splits give 15 distinct complexity
  vectors rather than 3 repeated. Over the v1 tutorial's 6, a Spearman correlation is
  decoration. The notebook says plainly what 15 can and cannot support -- |rho| must clear
  ~0.52 for p < 0.05 at n = 15, and ~140 features are tested per group, so some clear it by
  chance.

- **The three datasets vary one geometric property**, `n_clusters_per_class` in (1, 2, 4),
  rather than the v1 notebook's class balance. That is the axis the `task.` block measures,
  and it moves both sides of the correlation: measured, `task.graph_spec_entropy` rises
  0.33 -> 0.42 -> 0.62 and median f1 across all ten models falls as the classes break into
  more interleaved lobes. Class balance is already covered by a dozen existing columns and
  changes how a metric is computed more than what the problem is.

- **Search spaces are deliberately small and the notebook says so** rather than implying the
  shipped defaults were used: `n_trials: 8` (not 50), 3-fold CV (not 5), PQK at `reps: 2`
  (not 4), and trimmed TabPFN/MLP/CatBoost/RF spaces. Every cut is annotated in
  `configs/config.yaml` with what it saved; together they took the sweep from 29.2 s to
  14.7 s per (dataset, split). The single largest was TabPFN: writing its search space as
  lists with no continuous range makes the tuner lower `n_trials` from 8 to 4 on its own,
  which is 12 transformer forward passes instead of 24.

#### The target spectrum: where `y` sits in the geometry of `X`

- **`evaluate()` gains a 16-column `task.` block** from the new
  `qbiocode.evaluation.task_spectrum`, alongside the `mfe.` and native blocks. Every
  other complexity measure QBioCode computes describes `X` alone, or describes `y` only
  through a cheap classifier's view of it. This block describes neither: it builds a
  weighted mutual k-NN graph on `X` with local scaling, diagonalizes the normalized
  Laplacian to get a Fourier basis intrinsic to the data geometry, expands the centred
  label vector in that basis, and reports functionals of the resulting distribution of
  *target power over geometric frequency*.

- **What it buys is one distinction the existing block cannot make.** A target can be
  perfectly deterministic and still sit almost entirely in the high-frequency part of
  the geometry -- parity, checkerboard and alternating-sign targets all do -- and
  classifiers with low-pass inductive bias fail on those for reasons unrelated to label
  noise. Measured on 160 points on a circle: an alternating `+-+-` target reads
  `graph_hf_mass` 1.00 with `graph_spec_entropy` 0.00 (one geometric mode), where random
  labels on the same points read 0.77 and 0.86 (power spread over the whole spectrum).
  Both are "high frequency"; only the entropy separates structured oscillation from
  broadband noise, and that separation is the point of the block.

- **The eight features** are `graph_dirichlet` (the spectral centroid of the labels,
  bounded in `[0, 2]`), `graph_hf_mass`, `graph_spec_entropy`, `graph_bandwidth90`,
  `diffusion_half_life`, `pca_tail_signal` (label power outside the PCs carrying 90 % of
  `X`'s variance -- the signal unsupervised dimensionality reduction throws away),
  `h0_fragmentation` (class-conditioned H0 persistence, exact from MST edge weights, so
  no `ripser`/`gudhi` dependency), and `purity_auc`.

- **Each ships with a permutation z-score, and that is not decorative.** In sparse
  high-dimensional data almost nothing is smooth, so random labels look high-frequency
  by default and a raw reading means little on its own. Measured: a `p=501` dataset
  whose discriminative direction was real and low-variance scored `graph_hf_mass` 0.53
  and `graph_spec_entropy` 0.85 -- indistinguishable from noise by the raw values, and
  correctly flagged by every z-score coming back below 1.1. The null is cheap because a
  permutation leaves the graph untouched, so the eigendecomposition is computed once per
  `k` and each draw costs one matrix-vector product.

- **Every graph feature is the median over `k` in `{5, 10, 20}`**, so a descriptor has
  to survive a modest change of neighbourhood scale to be reported at all.
  `stability=True` adds the five coefficient-of-variation columns for inspecting that
  directly; they are off by default rather than spending five columns of a
  few-hundred-row meta-dataset on a mostly-noise diagnostic.

- **Cost: 0.18 s at `n=100, p=20000`**, the shape at which the curated pyMFE set takes
  about 50 s. The scaling differs, though -- the dominant term is an `O(n^3)`
  eigendecomposition per `k`, so it is 2.7 s at `n=400` and 12 s at `n=800`. Pass
  `task_spectrum=False` for an unusually tall dataset.

- **Two decisions in the construction are load-bearing, and both were found by
  measurement rather than reasoning:**

  - *High-frequency mass is thresholded at an absolute eigenvalue, not a quantile.* The
    obvious reading of "the highest-frequency quarter of the spectrum" is
    `lambda >= Q_0.75(lambda)`, and it is wrong: the *shape* of the k-NN Laplacian
    spectrum changes with `k`, not just its scale (`[0.002, 1.513]` at `k=5` against
    `[0.006, 1.231]` at `k=10` on identical points), so the alternating target -- a
    single mode at `lambda ~ 1.19` both times -- was reported as **low**-frequency at
    `k=5` and scored 0.009 after aggregation, *below* the 0.231 that random labels get.
    The threshold is `lambda > 1` instead, which is where a mode's neighbour-weighted
    autocorrelation turns negative, so it is derived from `L_sym` rather than chosen.
  - *Repair edges are floored at the smallest affinity already in the graph.* Mutual
    k-NN disconnects readily, so the graph is MST-augmented to one component -- but with
    local scaling, a bridge between well-separated components has an affinity that
    **underflows to exactly 0.0** in double precision (three blobs at mutual distance 20
    with `sigma ~ 0.5` give an exponent of -5903, against an underflow limit near -745).
    The edge was added and the graph stayed disconnected, so `L_sym` kept three zero
    eigenvalues and the trivial-mode removal took one of three, leaving two `lambda = 0`
    modes in the basis to absorb target power. Nothing about that was visible in the
    output. `_laplacian_basis` now refuses a disconnected graph outright.

- **`detect_complexity_schema` extends the mixed-schema guard to the new block.** A
  table concatenating a pre-`task.` results file with a fresh run is refused rather than
  trained on, for the same reason the legacy/pyMFE mix is: the older rows hold `NaN`
  across the whole block, training maps those to zeros, and zero is a value the target
  spectrum can genuinely take -- so nothing downstream could notice. A `pymfe`-schema
  table with no `task.` columns at all still trains exactly as before, which every
  `RawDataEvaluation.csv` written before this change is.

#### pyMFE replaces the hand-rolled dataset complexity measures

- **`evaluate()` now returns 125 complexity measures instead of 23**, of which 115 come
  from [pyMFE](https://github.com/ealcobaca/pymfe) and carry an `mfe.` prefix. `pymfe`
  is a new base dependency (it adds one transitive dependency, `gower`; everything else
  it needs was already declared). The `mfe.` prefix is functional, not decorative:
  `QuantumSage` selects the whole pyMFE block with one `startswith` test, and it
  guarantees no collision with the natively-computed names.

- **This closes a real gap rather than restating the old columns in new names.** The
  legacy block had no landmarking features and no classification-complexity features at
  all. It described the data and hoped the description predicted model performance;
  landmarking (`mfe.one_nn`, `mfe.naive_bayes`, `mfe.linear_discr`, `mfe.best_node`,
  `mfe.elite_nn`) instead measures cheap-learner accuracy directly, which is the
  strongest known predictor for exactly what QSage does. The Lorena et al. (2019)
  F/L/N/T/C families, decision-tree model-based measures, cluster-validity indices and
  concept measures are new as well.

- **Only 76 of pyMFE's ~105 meta-features are shipped, and the exclusions are
  measurements rather than taste.** This matters because the failures are silent: pyMFE
  reports a measure it could not compute as `NaN` and warns, and QBioCode must pass
  `suppress_warnings=True` (a wide matrix otherwise emits thousands of lines per
  dataset), so a dead measure would quietly occupy a column that QSage then trains on.
  The notable ones:

  - `f1v` raises `ValueError` on **every** dataset, iris included, under NumPy >= 2 --
    pymfe 0.4.4 assigns a `(1, 1)` array into a scalar slot at `complexity.py:912`.
    QBioCode cannot avoid NumPy 2, since `qiskit-machine-learning==0.9.0` requires it.
    This is why `get_fdr` is *kept* rather than replaced: `f1v` is pyMFE's multivariate
    Fisher measure, and the surviving `f1` is univariate **and inverted** (larger means
    harder), so substituting it into `calculate_SLGH` would have flipped the sign of
    that term and silently changed what QSage learns.
  - The `itemset` group is O(p^2) with no subsampling: 94 s at p=2000, so hours on an
    omics matrix.
  - `eigenvalues` accounted for 957 s of a 1007 s extraction at p=20000, and its `.mean`
    is bit-identical to `var.mean` (the mean covariance eigenvalue is trace/p).
  - `attr_conc` and `class_conc` default to `max_attr_num=12`, so on a wide matrix they
    silently describe 12 randomly chosen columns.
  - `lh_trace` and `roy_root` go to `+inf` once p >= n, which is worse than `NaN`
    because it propagates into QSage's regressors instead of being caught.
  - `g_mean`/`h_mean`/`sd_ratio`/`num_to_cat` are always `NaN` here; `t1`, `sc`, `nre`,
    `sparsity`, `var_importance`, `attr_ent`, `random_node` and `nr_disc` are constant;
    `t2` duplicates `attr_to_inst` exactly and `c1` duplicates `class_ent` for binary
    labels.

  `qbiocode/evaluation/mfe_features.py` records each reason next to the list it
  justifies, and `tests/test_dataset_evaluation.py` asserts the excluded names stay
  excluded -- so re-adding one fails loudly instead of shipping a dead column.

- **The `.sd` of a one-vs-one measure is dropped when the labels are binary.** With two
  classes there is exactly one class pair, so `f2.sd`, `f3.sd`, `f4.sd`, `l1.sd`,
  `l2.sd`, `l3.sd` and `can_cor.sd` are `NaN` by construction, not by failure. They are
  suppressed for binary input and retained for multiclass. The result is that no
  column in a QProfiler run is guaranteed empty: `evaluate()` returns zero `NaN` and
  zero `inf` on the committed fixtures and at 100x20000.

- **`evaluate()` takes `random_state` (default 0), and it is load-bearing.** The
  landmarking measures cross-validate and the clustering measures run k-means, so
  without a fixed seed the whole pyMFE block changed between two runs on identical
  input.

- **Cost.** ~0.2 s at 100x10, ~0.6 s at 500x50, ~5 s at 100x2000, ~53 s at 100x20000.

### Changed

#### QSage detects its feature schema instead of naming it

- **`QuantumSage._columns_data_features` is now derived from the input**, via the new
  `detect_complexity_schema()`. A table with `mfe.`-prefixed columns trains on the
  current 125-feature schema; a table carrying all 23 legacy columns trains on those.

  This is not gold-plating. `tutorial/QSage/data/qprofiler_benchmarks.csv`, the 576-row
  table the QSage tutorial trains on, is in the legacy schema and **cannot be
  regenerated from this repository**: it covers `class_data-{1..16}.csv` and only
  `class_data-1`, `-2` and `-3` are committed. The datasets are synthetic and seeded, but
  the sweep that produced those 16 is recorded nowhere, and rebuilding the table would
  additionally need a full 16 datasets x 3 embeddings x 6 models x 2 iterations
  QProfiler run. Hardcoding the new schema would have left the tutorial with no training
  data. To rebuild it once those datasets exist:

  ```bash
  qprofiler folder_path=<dir with class_data-1..16.csv> file_dataset=ALL \
            embeddings='[none,pca,nmf]' model='[dt,lr,mlp,nb,rf,svc]' iter=2
  ```

  A half-migrated table -- pyMFE columns without the native ones, which means it was
  subset or concatenated across QProfiler versions -- is refused by name rather than
  trained on.

- **`calculate_SLGH()` looks up the sample-count column** (`# Samples` in the legacy
  schema, `mfe.nr_inst` in the current one) rather than naming it. Its other two inputs,
  `Intrinsic_Dimension` and `Fisher Discriminant Ratio`, are retained natively in both
  schemas, so the formula itself is unchanged.

- **`Total Correlations` is replaced by `mfe.cor.mean` and `mfe.nr_cor_attr`.** The old
  column was the unnormalized `sum |rho_ij|` over feature pairs, which grows with p^2 --
  so it was dominated by the feature count and not comparable between datasets of
  different widths, which is exactly the comparison QSage makes.

### Fixed

#### `ModelResults.csv` was unreadable whenever tuned and untuned models shared a run

- **`pandas.read_csv` refused the file outright** with `ParserError: Expected 150 fields in
  line 7, saw 151`. QProfiler writes one row per model as each finishes, using a header
  written when the file is empty -- which holds only while every model contributes the same
  keys. Two do not: a *tuned* model reports `BestParams_Tuned` and an *untuned* one reports
  `Model_Parameters`. `grid_search: True` with `tune_quantum: False` -- the ordinary way to
  sweep, since one Optuna trial on a quantum model is a full quantum fit -- puts both in one
  run, and from the first untuned row onward every row was one column wider than the header.
  Nothing downstream could read the results: not QSage, not `compute_results_correlation`,
  not a notebook. This is why it went unnoticed: the committed tutorial table was written
  with `grid_search: False`, where every model agrees.

- **Worse, values leaked between rows.** The writer merged each model's results into a single
  dict reused across models, so a column one model reported persisted into every later row
  that did not report it. The `pqk` row carried the *preceding* model's `BestParams_Tuned`
  value -- a naive-Bayes `var_smoothing` reported as PQK's tuned hyperparameters. Unlike the
  raggedness this failed silently, in a readable file, attributed to the wrong model.

- Rows are now built independently from a snapshot of the shared per-split block, and written
  through `csv.DictWriter` against the header on disk: a row introducing a column rewrites
  the file with the union header and pads earlier rows, and a row missing one writes an empty
  cell rather than shifting every later value left. `tests/test_model_results_csv.py` pins
  both failures, including the silent misalignment direction.

#### The `auc` column was never an AUC

- **`modeleval` computed `roc_auc_score(y_test, y_predicted)` -- `roc_auc_score` applied
  to hard predicted labels.** A ROC AUC needs a *ranking*; given labels it has only two
  points to integrate under, and on a binary target the result is identically
  `balanced_accuracy_score(y_test, y_predicted)`. So the column named `auc` in every
  `ModelResults.csv`, for all 28 dispatch entries, held balanced accuracy. It went
  unnoticed because the number looks right in every cheap way: finite, inside [0, 1], and
  correlated with accuracy, so a range check passes and a reader has nothing to catch.

- **Recorded `auc` values change, and old ones are not comparable with new ones.** This
  is a change in *what the column measures*, not a precision fix, so any results table
  written before this release carries the old statistic under the new name. That includes
  the committed `tutorial/QSage/data/qprofiler_benchmarks.csv`, which `QuantumSage`
  trains on: a model fitted on that file has learned to predict balanced accuracy, and
  its predictions are not on the same scale as an `auc` a fresh QProfiler run now
  reports. Mixing pre- and post-fix rows in one table, or comparing a published figure
  against a re-run, silently compares two different metrics. Re-run rather than reconcile.

- **`modeleval` takes a new optional `y_score`, and `auc` is computed from that alone.**
  All 24 call sites across the 15 `compute_*` modules now pass the best score their
  fitted estimator can produce. The new
  `qbiocode.evaluation.model_evaluation.extract_binary_scores(estimator, X)` helper does
  the extraction: `predict_proba` if available (positive-class column located through
  `classes_` rather than hardcoded as `[:, 1]`), else `decision_function`, else `None`.

- **A missing score is recorded as `float('nan')`, never as the old label-based number.**
  Falling back would put a different statistic under the same column name, which is the
  bug itself; a missing value is visible where a mislabelled one is not. `auc` is NaN
  when, and only when, no real AUC exists: the estimator offers neither scoring method,
  the target has three or more classes, or `y_test` holds a single class. Every learner
  shipped in QBioCode does produce a score, so no configuration goes NaN today -- the
  NaN path is what keeps a future learner from quietly reintroducing the defect.

- **A three-class target no longer aborts the run.** It used to raise `ValueError` from
  inside `roc_auc_score`, naming a `multi_class` parameter the user never set, after a
  perfectly successful fit. `accuracy` and `f1_score` are well defined for any class
  count and now come back; only `auc` is NaN, because a single score column cannot rank
  three classes and a multiclass AUC is a different metric with an averaging choice to
  make (`evaluation_metrics` in the same module offers it).

- **Where each learner's score comes from**, since it differs and the differences drove
  the shape of the fix:

  - The seven older classical learners (`dt`, `lr`, `mlp`, `nb`, `rf`, `svc`, `xgb`) are
    fitted inside a `OneVsOneClassifier`, which has **no** `predict_proba` -- the reason
    this was not a one-line fix. It does have `decision_function`, and on a binary target
    the wrapper holds exactly one pairwise estimator and returns that estimator's own
    margin as an `(n_samples,)` vector oriented towards the positive class. Verified to
    give AUCs identical to the unwrapped estimator's `predict_proba` and
    `decision_function`.
  - `compute_svc` passes `probability=True`, which `OneVsOneClassifier` discards. The
    `decision_function` route makes that harmless rather than broken, and it is now
    commented where it is passed so the next reader does not chase it.
  - `dt` is the one learner whose `auc` still equals its balanced accuracy. `compute_dt`
    grows the tree to pure leaves, so its `decision_function` takes two values and has no
    ranking to offer. That is honest degeneracy, not the old bug: constraining `max_depth`
    or setting `ccp_alpha` gives it a real ranking.
  - `catboost`, `tabpfn` and every `_opt` twin are fitted unwrapped, so `predict_proba`
    is used -- except `svc_opt`, where `probability` is not a searched hyperparameter, so
    it falls through to `decision_function` like its base twin.
  - `qsvc` subclasses scikit-learn's `SVC` and inherits `decision_function`;
    `PegasosQSVC` publishes `predict_proba` (a sigmoid of its own decision values).
  - `vqc` and `qnn` use `NeuralNetworkClassifier.predict_proba`, the network's forward
    pass. With the default sampler primitive that is `(n, 2)` probabilities over the
    parity outcomes; with `primitive='estimator'` it is a single `(n, 1)` expectation
    value in [-1, +1] -- not a probability, but exactly the quantity `predict` takes the
    sign of, so still a ranking. Column ordering was checked against `predict` for both.
  - `pqk` and `qpl` score the **quantum projection**, which is the space their classical
    heads were fitted in; the raw features would be the wrong width.
  - `qensemble` is the one learner with no fitted estimator to interrogate, and needs
    none: it measures `[p0, p1]` per test sample directly and `p1` is the positive-class
    score the argmax was throwing away.

- **The old behaviour had been pinned by a passing test.**
  `tests/test_classical_models.py` asserted `row['auc'] == balanced_accuracy_score(...)`
  for all nine classical models -- an accurate record of the defect that also meant any
  fix would fail the suite. It is replaced by `TestTheAucColumnIsARankingAuc`, which pins
  the column against the score array the learner actually handed `modeleval` (captured at
  the hand-off, since neither the score nor the fitted estimator survives into the
  results frame), that the array is a ranking rather than a relabelling of the
  predictions, and that `lr`'s `auc` now *differs* from its balanced accuracy -- the one
  assertion that fails if the fix is reverted. `TestAMissingScoreIsRecordedAsNaN` pins
  the NaN policy directly on `modeleval`. In
  `tests/test_model_contract_matrix.py`, `test_the_recorded_metrics_are_the_metrics_of_the_recorded_predictions`
  no longer recomputes `auc` from `y_predicted` -- it cannot, which is the point -- and
  checks range and finiteness instead.

#### pyMFE and dataset evaluation

- **`evaluate()` returned an object-dtype frame.** Building the row from a dict that also
  holds the `Dataset` string made every numeric column `object`. This was invisible while
  the only consumer wrote it to CSV and read it back -- which restores the dtypes -- but an
  in-memory caller got a frame whose float columns reject NumPy ufuncs, and QSage's
  `calculate_SLGH` calls `np.log` on them. It now coerces the measure columns to numeric.

- **`std_entropy` was always exactly 0.** `get_entropy` returned `np.std()` of a scalar.
  The column is gone; `mfe.class_ent` and `mfe.c2` carry the class-distribution signal.

- **`get_complexity()` ignored its own `n_neighbors` and `n_components` arguments**,
  hardcoding `Isomap(n_neighbors=10, n_components=2)`, so passing anything else silently
  did nothing. Both are now forwarded.

- **`dataset_evaluation.py` had no tests at all.** `tests/test_dataset_evaluation.py`
  covers the output schema, dtype, finiteness, seed reproducibility, the `p > n` regime,
  and the curation guards.

#### CatBoost and TabPFN as classical classifiers

- **Two new classical models, `catboost` and `tabpfn`**, selectable from `model` in the
  config exactly like `rf` or `xgb`, each with an `_opt` twin driven by
  `gridsearch_<model>_args` through both the Optuna and the exhaustive-grid engines.
  Both are also available as `qpl` heads on a quantum projection, and both are exported
  from `qbiocode` and `qbiocode.learning`. Guarded by `tests/test_catboost_tabpfn.py`
  and `tests/integration/test_catboost_tabpfn_integration.py`.

- **`catboost` is a core dependency; `tabpfn` is the new `[tabpfn]` extra.** CatBoost is
  an unconditional scikit-learn-compatible estimator with no gate, so it sits in
  `requirements-base.txt` beside xgboost. TabPFN could not: its pretrained weights are
  downloaded behind a **one-time license acceptance**, so it can never run unattended
  from a default install. Naming it without the extra raises an ImportError naming the
  extra; the license failure is translated into one naming `TABPFN_TOKEN` and the
  acceptance URL, instead of surfacing as a bare `TabPFNLicenseError` six frames inside
  `tabpfn.model_loading`.

- **Neither model is wrapped in `OneVsOneClassifier`**, unlike `compute_rf` and
  `compute_xgb`. CatBoost selects `MultiClass` as its loss automatically and TabPFN is
  natively multiclass, so a wrapper would fit n(n-1)/2 models to reach the answer the
  native loss already gives -- and for TabPFN would multiply an expensive forward pass
  by the same factor.

- **CatBoost as a QuantumSage surrogate**, as `sage_type='catboost_optuna'` (CLI:
  `--model-type catboost`), mirroring `_sage_xgboost_optuna` including its Optuna study
  and R-squared scoring. Having two independent boosters is useful precisely because
  they disagree on the small, wide tables QProfiler produces.

- **TabPFN is pinned to model version `v2`, and that is a licensing decision.** TabPFN's
  *code* is Apache 2.0 plus an attribution clause, but its *weights* are licensed per
  version and the regimes diverge: `v2` is under the Prior Labs License v1.1 (Apache 2.0 +
  attribution) and permits commercial use, while `v2.5`, `v2.6` and `v3` are under
  per-version **non-commercial and non-production** licenses. Upstream's constructor
  defaults to `v3`. QBioCode is Apache-2.0 software whose users include companies, so
  inheriting that default would quietly impose a non-commercial license on them.

  `model_version` (default `'v2'`) selects the checkpoint on both learners and is resolved
  into `model_path` before any fit — so it is visible in `get_params()`, survives
  `sklearn.base.clone` mid-search, and cannot differ between trials. It is deliberately not
  searchable: trials would otherwise compare different models under different licenses and
  report the winner as a hyperparameter. Selecting a restricted version warns and names the
  license; an explicit `model_path` still wins, which is how to run offline.

  A practical consequence: **`v2` needs no API token and no license acceptance at all.** It
  downloads anonymously, so `tabpfn` now works from a bare `pip install "qbiocode[tabpfn]"`
  — the previously-skipped TabPFN tests now run, and the tutorial notebook ships real
  numbers (0.840 accuracy here, ahead of both tuned boosters, with no training and no
  tuning).

- **A place to keep the TabPFN API token that cannot be committed.**
  `qbiocode.utils.tabpfn_account` reads the key from `~/.config/qbiocode/tabpfn.json`
  (created mode `0600` by `write_token_template()`) and puts it in `TABPFN_TOKEN`, which is
  the only channel TabPFN accepts. The location is **outside the repository** on purpose: a
  gitignored file in the tree is unlikely to be committed, not unable to be — `git add -f`
  overrides the rule, a rewritten `.gitignore` stops covering it, and a copy in a sibling
  clone is not covered at all. `tabpfn_json_path` in a config mirrors the existing
  `qiskit_json_path`, and QProfiler loads the token in the parent process (so joblib
  workers inherit it) whenever `tabpfn` is in the model list.

  Nothing in the module returns, logs or prints the token. `describe_token_source()`
  reports *whether* a token is configured and where it came from and nothing else — no
  value, no prefix, no fingerprint — because the tutorial notebooks are published with
  their committed outputs. `tests/test_tabpfn_token.py` asserts that, checks the
  `.gitignore` rules with real `git check-ignore` rather than by reading patterns, warns on
  a group-readable file, refuses to overwrite an existing token file, and treats the
  unfilled template as "not configured" so a half-finished setup does not look like a key
  TabPFN mysteriously rejects. It also adds a general guard that no committed notebook
  output carries a token-shaped value.

- **`qbiocode.utils.check_tabpfn_access()` tells the three TabPFN gates apart**, because
  upstream conflates them. With a valid API key whose license has not been accepted,
  `tabpfn.browser_auth.ensure_license_accepted` falls through to a browser login, fails for
  want of a TTY, and reports *"no interactive terminal is available... set the TABPFN_TOKEN
  environment variable"* — advising a step that is already done. The real cause is that an
  API key **authenticates** you and does not **accept** the license, which is a separate
  action on the account. The diagnostic distinguishes: extra missing, token absent, token
  rejected, token valid but license unaccepted, server unreachable, or nothing blocking —
  and returns no token material, so it is safe to paste into an issue. The
  weights-unavailable error now points at it.

- **New tutorial: `tutorial/CatBoost_and_TabPFN/catboost_and_tabpfn.ipynb`.** Runs
  `catboost` against `xgb` and `rf` untuned and then tuned, and spends most of its length
  on the two things that are not obvious from the config: the bootstrap conflict (showing
  all four combinations and what QBioCode says about each), and `min_data_in_leaf` being
  *measured* inert at the default grow policy rather than merely asserted to be. Its
  TabPFN section runs the model if the weights are reachable and otherwise prints the
  translated error plus the one-time license step, so the notebook executes end to end
  either way -- it is in `tests/integration/test_notebook_execution.py`, which proves that.
  (Once the v2 pin removed the licence gate, that cell was changed to fit TabPFN for real
  rather than degrade, so the notebook now *requires* the `[tabpfn]` extra and the execution
  test skips it when the extra is absent. CI never reached this, because the default
  `addopts` excludes `slow`.)
  It also makes the point that TabPFN needs a license rather than a GPU, since assuming
  the opposite is the likely first guess.

- **`tutorial/QProfiler/example_qprofiler.ipynb` now includes `catboost`** in its model
  list, so the tutorial sweep compares two boosters. Its first cell also gained the
  `OMP_NUM_THREADS` guard the newer notebooks carry -- that config runs `pqk` (qiskit-aer)
  alongside xgboost and catboost, which is exactly the combination that dies without a
  traceback on macOS -- and a `warnings.simplefilter("ignore")` ahead of the imports,
  because without ipywidgets installed `tqdm.auto` warns at import time and the warning
  text carries the absolute path of the environment into the published page.

- **`compute_tabpfn` imports `tabpfn` lazily**, inside its functions. `import tabpfn`
  imports `torch`, which would have undone `qbiocode.utils._openmp`'s ordering guard and
  mapped torch's OpenMP runtime into every QBioCode process -- including ones that only
  wanted a random forest. `tabpfn_is_available()` answers the availability question with
  `importlib.util.find_spec`, which does not execute the module.

### Fixed

- **A CatBoost config's validity depended on the loss rather than on the config.**
  `subsample` and `bagging_temperature` belong to mutually exclusive CatBoost bootstrap
  schemes, and CatBoost derives the default scheme from the loss: `MVS` for `Logloss`,
  `Bayesian` for `MultiClass`. QProfiler is a binary-classification tool, so the
  inferred loss is `Logloss` and `subsample` works at the default -- but
  `loss_function: MultiClass` is legal on a two-class target and flips the scheme:

      CatBoostError: default bootstrap type is Bayesian, which does not support subsample

  Because `run_study`'s Optuna objective does not catch exceptions, one such corner
  aborted the whole study rather than costing a trial. `bootstrap_type` is now pinned as
  soon as either parameter is named (`Bernoulli` for `subsample`, `Bayesian` for
  `bagging_temperature`), on both the tuned and untuned paths. A block that asks for both
  schemes at once, or searches `bootstrap_type` across values contradicting the one it
  named, is rejected before the search starts with a message naming the config key. The
  shipped `gridsearch_catboost_args` stays clear of the area entirely.

- **A TabPFN fit killed the process outright on macOS.** `qbiocode/__init__.py` initialises
  xgboost's `libomp` first, deliberately; importing `tabpfn` brings torch's copy in as a
  second LLVM OpenMP runtime under the same install name, and the second one to open a
  parallel region dies below Python — exit 139, no traceback, no exception, a notebook front
  end reporting only "kernel died". Measured: a `compute_tabpfn` call in a process that had
  imported `qbiocode` exited 139 without `OMP_NUM_THREADS` set and returned a score with it.
  Reordering cannot fix it — xgboost-first breaks torch and torch-first breaks XGBoost, so
  the orderings are mutually exclusive. `_cap_openmp_threads` now sets `OMP_NUM_THREADS=1`
  immediately before the `tabpfn` import, which is early enough for torch to read it, and
  warns that it did. Scoped to macOS, and it never overrides a value the caller set.

- **`get_creds` printed the IBM Quantum API token to stdout on every call.** It reads the
  token from `~/.qiskit/qiskit-ibm.json` -- a file the user deliberately keeps outside any
  repository -- and then `print`ed the assembled dictionary, token included, moving a
  secret from a protected location to an unprotected one: a terminal scrollback, a CI log,
  a pasted issue report, and, because the tutorial notebooks are committed *with their
  outputs* and published, potentially a public page. The new
  `qbiocode.utils.ibm_account.redacted()` replaces secret values with `<redacted>` while
  keeping the rest, so the diagnostic the print existed for -- which channel and instance
  resolved -- still works. The caller still receives the real token; only the echo is
  redacted.

  No notebook in the tree had leaked one: the only notebook mentioning `get_creds` runs on
  the simulator and never called it. `tests/test_credential_hygiene.py` keeps that true
  rather than lucky, by capturing real stdout with a token present rather than by reading
  the source -- a redaction helper that exists and is not called is the failure mode worth
  catching.

- **The untuned and tuned CatBoost paths disagreed about which bootstrap combinations
  are legal.** `compute_catboost` had its own copy of the pinning, which set `Bernoulli`
  whenever `subsample` appeared and then left CatBoost to raise about
  `bagging_temperature` if both were given -- so `catboost_args` got
  `CatBoostError: bagging temperature available for bayesian bootstrap only` for a
  combination `gridsearch_catboost_args` rejected up front with a message naming the key.
  It had no check at all for `subsample` under an explicitly requested Bayesian bootstrap.
  Both paths now call `_resolve_bootstrap`, which takes the block name for its messages,
  so the two cannot diverge again.

- **`bootstrap_type` written as a `{low, high}` range was destroyed rather than refused.**
  `list({'low': 1, 'high': 3})` is `['low', 'high']`, which satisfied the compatibility
  checks; `build_search_space` then read the mapping as an *integer* range and proposed
  `bootstrap_type=2`, so CatBoost failed with `Can't parse parameter "type" with value: 2`
  -- naming neither the config entry nor the mistake. A range for the scheme is now
  rejected by name, since it selects an algorithm rather than a number.

- **QProfiler now refuses a non-binary dataset instead of warning about one.** It used to
  log "multi-class support is experimental" and continue, but multi-class support is not
  experimental, it is absent: `modeleval` scores every model with
  `roc_auc_score(y_test, y_predicted)` and passes no `multi_class`, so a three-class dataset
  raised `ValueError: multi_class must be in ('ovo', 'ovr')` a few hundred lines later, after
  the embeddings had been computed and the models fitted. The warning invited the user into a
  certain failure and then reported it against a parameter name they had never seen. The
  check now raises up front, naming the dataset, its class count, its label mapping, and the
  fact that this is a boundary of the tool rather than a defect.

- **`min_data_in_leaf` was searchable but inert at CatBoost's default grow policy.**
  CatBoost accepts it under every policy and honours it only under `Depthwise` and
  `Lossguide`: measured, the predictions at `min_data_in_leaf=1` and `=60` are identical
  under the default `SymmetricTree` and differ under `Depthwise`. Searching it therefore
  multiplied the fits while every value returned the same model. `compute_catboost_opt`
  is therefore no longer searchable: several values are **refused** with a message naming
  `grow_policy`, rather than warned about, since a warning let the wasted fits happen anyway.
  A single value is still accepted and passed to every trial, so the parameter stays usable
  with `grow_policy: Depthwise` -- and stays accepted by *both* twins, because dropping it
  from `_opt` alone would recreate the `loss_function` asymmetry, where a config key only one
  twin takes dies on an unexpected keyword argument inside a joblib worker. The CatBoost
  QuantumSage surrogate no longer searches it either.

- **A license-gated TabPFN QPL head took the whole run down**, after the quantum
  projection had already been computed and paid for. The availability filtering cannot
  catch it -- the extra is installed and imports fine, and only the fit discovers the
  checkpoint is gated. `compute_qpl` now drops that head with a warning, as it already
  did for an unusable xgboost or catboost, and keeps the other heads' results. Relatedly,
  every head being dropped reached `pd.concat([])` and raised "No objects to
  concatenate"; that case now names the models and points at the warnings.

- **`test_every_third_party_import_under_qbiocode_is_declared` did not read every runtime
  tier.** It checked imports against `requirements-base.txt` and `requirements-quvine.txt`
  only, so `tabpfn` counted as undeclared once `tabpfn_account` imported it with a real
  `from tabpfn... import` — `compute_tabpfn` reaches it via
  `importlib.import_module("tabpfn")`, a string the AST scan cannot see, which is why this
  surfaced late. The tier list now includes `requirements-tabpfn.txt`, with a note that it
  must grow with each new optional tier or the guard silently stops covering it.

- **`compute_catboost_opt` did not accept `loss_function`,** though `compute_catboost`
  did. `model_run` splats the config block into whichever twin it selects, so naming it
  in `gridsearch_catboost_args` raised `unexpected keyword argument` inside a joblib
  worker. It is now accepted and passed to every trial fixed rather than searched -- it
  specifies the problem rather than tuning it.

- **CatBoost wrote `catboost_info/` into the working directory on every fit.** Under
  QProfiler that directory is shared by every joblib worker, making it both a race and
  litter left in the user's project; a tuned run wrote it once per trial per fold.
  `allow_writing_files=False` is now passed to every CatBoost estimator the package
  builds -- the learners, the QPL head and the Sage surrogate.

- **CatBoost narrated one line per boosting iteration.** Silenced independently of
  QBioCode's own `verbose` flag, which selects the one-line result summary and had been
  distinct from CatBoost's identically-named parameter.

- **CatBoost's `predict()` returns an (n, 1) column under the `MultiClass` loss**,
  reachable on a two-class target via `loss_function`. Flattened to the 1-D shape every
  other learner returns. This is a consistency fix rather than a crash fix: scikit-learn
  1.9 silently squeezes a column vector, so the metrics were already correct -- but the
  array is also stored in the results frame as `y_predicted_<model>`, and leaving its
  shape to depend on undocumented squeezing behaviour is a latent trap for anything
  reading that column.

### Changed

- **`compute_qpl`'s default `classical_models` is now
  `['rf', 'mlp', 'svc', 'lr', 'xgb', 'catboost']`.** `tabpfn` is accepted but
  deliberately not a default: defaulting a license-gated optional extra on would make
  every ordinary QPL run warn about something the user never asked for. An unavailable
  head is dropped with a warning rather than wasting the quantum projection that has
  already been computed.

- **`compute_qpl` reads `best_params_` defensively.** Every other head is a
  `RandomizedSearchCV`; the TabPFN head is fitted bare, because wrapping it as the
  others are would cost 200 transformer forward passes per projection to choose between
  near-identical inference settings. It reports `get_params()` instead.

#### Optuna tuning for the quantum classifiers

- **`qsvc`, `vqc`, `qnn`, `pqk` and `qpl` can now be tuned**, through the same
  `gridsearch_<model>_args` blocks as the classical models. Previously they had no tuning
  path at all: comparing `Z` against `ZZ` at two `reps` meant writing four config files
  and diffing four runs. Tunable per model are `encoding`, `entanglement`, `reps` and
  `primitive`, plus `C`/`gamma`/`pegasos` for QSVC and `ansatz_type`/`local_optimizer`/
  `maxiter` for VQC and QNN. There is no `n_qubits` to tune -- the qubit count follows
  from the width of the data reaching the model. Guarded by `tests/test_quantum_tuning.py`.

- **Off unless asked for twice: `grid_search: True` *and* `tune_quantum: True`.** A
  quantum fit builds an n-by-n fidelity kernel by circuit simulation, so tuning
  multiplies a quantum model's cost by the trial budget. Had it switched on with
  `grid_search` alone, every existing config naming a quantum model would have become an
  order of magnitude slower on upgrade with no change on the user's part. `tune_quantum`
  without `grid_search` is an error rather than a silent no-op, and a quantum model with
  no `gridsearch_<model>_args` block is named rather than skipped.

- **A quantum candidate is scored once, not `cross_validation` times.** Each trial is
  scored on a single stratified holdout carved out of the training data
  (`validation_split`, default 0.25); k-fold would multiply an already expensive search
  by k. The caller's test set is never touched by the search. The budget is
  `n_trials_quantum` (default 10) rather than `n_trials` (50).

- **Tuning against real hardware is refused** unless `allow_hardware_tuning: True` is
  set. Every trial would be a separate queued job billed against the user's instance,
  and the failure mode is silent: the run simply never appears to finish.

- **A hyperparameter combination the quantum stack cannot build costs a trial, not the
  run.** Not every `(encoding, entanglement, primitive)` triple is constructible; a
  failing trial is pruned and reported in a summary warning. If *every* trial fails the
  error says so and points at the config block, rather than reporting an empty study.

- **New tutorial: `tutorial/Hyperparameter_Tuning/optuna_vs_gridsearch.ipynb`**, which
  times the exhaustive sweep against Optuna on the shipped blocks (14x apart on the
  committed run, at indistinguishable accuracy), demonstrates range syntax, and tunes a
  QSVC. It scales with `scale_train_test` *after* splitting, and says why: fitting a
  scaler on the full matrix leaks test-set statistics into training and quietly flatters
  every number in the notebook. It also reports a tuned QSVC losing to the untuned
  default and explains that too -- six candidates selected on an 11-sample validation
  holdout is selection on noise -- rather than presenting tuning as a free win. Executed
  end to end by `tests/integration/test_notebook_execution.py`. Its first cell prints which
  QBioCode actually answered, and its results reader accepts the pre-rename column name and
  explains what finding it means -- reading only `BestParams_Tuned` raised a bare
  `KeyError` when the notebook ran on a kernel belonging to a different checkout, which is
  easy to hit because VS Code only auto-discovers a `.venv` at the workspace root.

### Fixed

- **A budget below one trial blamed the search space.** `n_trials: 0` (or a negative
  value) ran nothing, so `study.best_params` raised Optuna's "No trials are completed
  yet" on the classical path, and on the quantum path the all-trials-failed branch caught
  it first and reported that *every trial had failed* and the `gridsearch_<model>_args`
  block was probably unbuildable -- pointing at a config block that was perfectly fine.
  The budget is now validated up front and the message names the value it was given.

- **An invalid `validation_split` was reported against `test_size`.** A value of 0, 1 or
  a negative number reached `train_test_split` and surfaced as
  `InvalidParameterError: The 'test_size' parameter ... must be a float in the range
  (0.0, 1.0)` -- naming a parameter that appears nowhere in a QBioCode config. It is now
  checked against its own name.

- **`tuner: grid` silently ran Optuna for the quantum models.** There is no
  exhaustive-grid engine for them -- a quantum candidate is scored by running the whole
  compute function, not an estimator `GridSearchCV` can drive -- so `tuner: grid` with
  `tune_quantum: True` quietly did something other than what was asked, which makes the
  resulting numbers impossible to interpret later. It now warns, names the models
  affected, and says how to opt out.

- **The shipped `gridsearch_mlp_args` buried its own output in warnings.** Its
  `hidden_layer_sizes` is `[[20], [50], [100]]`, and a categorical choice Optuna cannot
  persist makes it warn once per trial: "Choices for a categorical distribution should be
  a tuple of None, bool, int, float and str for persistent storage but contains [20]
  which is of type list." The studies here are created in memory and discarded when the
  search returns, so the limitation the warning describes cannot be reached; it is now
  suppressed by message, narrowly and only around the search. The values are still
  reported back exactly as configured.

- **Every config key is now checked against the documentation.** `tests/test_docs_structure.py`
  fails when `config.yaml` offers a key that appears nowhere under `docs/source`. Adding a
  key and forgetting the docs is otherwise invisible -- the yaml parses and the only
  symptom is a user who cannot find out what the key does. It caught five
  `gridsearch_<quantum model>_args` block names, both projection-cache directories and the
  `BestParams_Tuned` column, all of which are now documented; ten pre-existing keys are
  recorded as debt rather than exempted.

- **QPL's projection cache was keyed on `data_key` alone, so changing a quantum
  hyperparameter silently reused the wrong projection.** `compute_qpl` cached its
  projected feature matrix under a name built only from `data_key`, in a hardcoded
  `qpl_projections` directory. Changing `encoding`, `entanglement`, `reps` or `primitive`
  and rerunning loaded the projection computed for the *previous* settings: the new
  circuit was never run and the reported result described the old one. The name now
  includes a fingerprint of the settings that change the circuit, the row count is
  validated on load with both numbers named, and the directory can be redirected with
  `qpl_projection_dir` -- matching what `compute_pqk` already did. Tuning made this
  acute rather than merely wrong: every trial after the first would have scored the same
  cached projection, so the search would have compared a hyperparameter against itself.
  Guarded by `tests/test_quantum_tuning.py::TestTheQplProjectionCache`.

- **`compute_qsvc_opt` reports a `local_optimizer` grid instead of silently searching
  it.** `compute_qsvc` accepts `local_optimizer` and never reads it -- the kernel is
  fitted by libsvm, not an optimizer -- so a grid over it would have multiplied the
  trials while every one returned the same model. Same treatment as `bootstrap` on
  XGBoost.

### Changed

#### Optuna replaces the exhaustive grid as the default hyperparameter search

- **`grid_search: True` now tunes with Optuna instead of fitting every combination.**
  `gridsearch_rf_args` as shipped is 4x2x4x3x3x2 = 576 combinations, so a single random
  forest cost 2,880 fits at `cross_validation: 5` — and the sweep was blind, spending as
  much time in a hopeless corner as in a promising one. The new default spends a fixed
  `n_trials` (50) steered by Optuna's TPE sampler; on the shipped RF block that is 6s
  against 96s for the full grid. The exhaustive sweep is still reachable with
  `tuner: grid`, which is what to use when reproducing a number published against it.
  Guarded by `tests/test_grid_search_partial.py`, which now runs all seven `_opt`
  learners under both engines.

- **A hyperparameter can now be given as a range, not just a list of decades.**
  `C: [1.e-03, 1.e-02, ... 1.e+03]` could only ever propose a power of ten; the same
  parameter written `C: {low: 1.0e-3, high: 1.0e+2, log: true}` is sampled continuously,
  so 3.7 is reachable. Integer bounds give integer values, `log: true` gives a log scale
  and `step: n` fixed spacing. **Existing configs are unchanged** — a list is still a
  list, searched as a categorical choice — and the two forms mix freely inside one block.
  Ranges require `tuner: optuna`; under `tuner: grid` every entry must be a list.
  `build_search_space` and `build_param_grid` are deliberately fed the same candidate
  dict and agree on which entries count as "not tuned", so switching `tuner` cannot
  change *which* hyperparameters are searched. See `qbiocode.learning._tuning`.

- **A malformed range names the model and the hyperparameter.** `{low: 10, high: 1}`,
  a log scale starting at zero, `log` together with `step`, an unrecognised key, or
  non-numeric bounds are all reported against the config entry that caused them, rather
  than surfacing later as an Optuna complaint about an anonymous distribution.
  Guarded by `tests/test_optuna_tuning.py`.

- **An unknown `tuner` is rejected at validation.** A misspelt value would otherwise
  fall through to the Optuna branch inside every `_opt` function, so a config asking
  for the exhaustive grid silently did not get it.

- **`cross_validation` is no longer a required key on the tuning path.**
  `args["cross_validation"]` was an unguarded lookup reached only when `grid_search` was
  on, so tuning from a config that omitted it died in `model_run` rather than at
  validation. It now defaults to 5.

- **The tuned-parameter results column is `BestParams_Tuned`, was
  `BestParams_GridSearch`.** The old name claimed an engine that is no longer the
  default. Every reader — `qbiocode.utils.qc_winner_finder` and `QuantumSage` — accepts
  both names, so `ModelResults.csv` files written before this change still load.
  `qc_winner_finder` previously assumed the absence of `Model_Parameters` implied the
  presence of `BestParams_GridSearch`, which made any third name a `KeyError` raised
  from inside `groupby`; it now looks for each name it knows and says which columns it
  found if none are present. Guarded by `tests/test_sage_contract.py`.


## [0.2.0] - 2026-09-10

### Added
- **QuVINE — Quantum View-based Network Embeddings** (`qbiocode.apps.quvine`), a new
  in-tree app that turns a graph into low-dimensional node embeddings by combining
  classical and quantum walks with skip-gram negative sampling.
  - Multi-view graph construction, random walk with restart (RWR) and discrete-/
    continuous-time quantum walks (DTQW/CTQW), SGNS embedding learning, quantum-calibrated
    filter / GAT / GraphGPS variants, view fusion, and classical baselines
    (node2vec, NetMF, APPNP, GraphSAGE, GCN-MF).
  - 83 method names selectable by a single `method` string, resolved by
    `qbiocode.apps.quvine.resolve_method()` and listed by `list_methods()`.
  - New `quvine` command-line tool: embeds a 2- or 3-column edge list and writes
    `embedding.csv` + `embedding_meta.json`.
  - Programmatic API: `embed(G, method)` returning an `EmbedResult`, plus
    `load_config`, `run_sgns`, `build_quantum_targets`.
  - **Installed via an optional extra**: `pip install "qbiocode[quvine]"`. The Python
    modules ship with QBioCode, but the heavy third-party dependencies (gensim,
    hiperwalk, node2vec, torch-geometric, python-louvain, ripser, omegaconf) do not.
    `import qbiocode` and every classical embedding work without the extra.

- **Graph-complexity metrics**: `qbiocode.evaluate_graph(G, name="")` returns a one-row
  `DataFrame` of graph descriptors (spectral gap, von Neumann entropy, modularity,
  density, community and topological metrics, and more). Deliberately separate from the
  embedding app — run it on the same graph to characterize it. Needs nothing beyond
  networkx/scipy/pandas; the persistent-homology metrics are skipped with a warning
  when `ripser` is absent.

- **Actionable errors for optional dependencies**: `qbiocode.apps.quvine._deps` resolves
  every `[quvine]` dependency through `require_module()`, so a missing one raises
  `QuvineDependencyError` naming the method, the extra, the exact pip command and the
  missing distribution, instead of a bare `ModuleNotFoundError` traceback. It subclasses
  `ImportError`, so code probing with `except ImportError` still works.
  `describe_environment()` and `missing_dependencies()` report what an environment has.

- **QuVINE is a first-class embedding.** `qbiocode.get_embeddings("quvine_rwr", X_train,
  X_test, n_components=8)` now works exactly like `"pca"`, `"nmf"` or `"umap"` — same call
  shape, same return of `(Z_train, Z_test)`. Any of the 83 QuVINE method names is accepted;
  a symmetric kNN graph is built over the rows, embedded, and reduced to `n_components`.
  QProfiler reaches them through its existing `embeddings:` config list, and passes
  `quvine_args` through for per-method overrides.
  - `qbiocode.SKLEARN_METHODS`, `qbiocode.QUVINE_HEADLINE_METHODS` and
    `qbiocode.QUVINE_METHODS` name what is available. `QUVINE_METHODS` lists all 83
    names even without the `[quvine]` extra installed — resolving a name is
    stdlib-only, so discovery and the "unknown embedding" error message work in a bare
    environment, and only *running* a method raises `QuvineDependencyError`. It is
    empty only if the QuVINE subpackage itself is unimportable.
  - Methods needing no extra beyond the base install — `netmf`, `appnp`, `gat_*` — work on
    a bare `pip install qbiocode`.

- **`qbiocode.is_transductive(name)`** reports whether a method sees test *features* at
  embed time. `spectral` and every QuVINE method have no out-of-sample `transform`, so they
  are fitted over the stacked train+test rows and sliced. `get_embeddings` now also emits a
  single `UserWarning` per call for those methods, stating plainly that test features
  participate in the geometry while test *labels* never do. Inductive methods are silent.

- **Quantum Backend**: Noisy simulation support based on IBM device noise models
  - New backend format: `noisy_<device_name>` (e.g., `noisy_ibm_cleveland`)
  - Extracts noise model from actual IBM Quantum devices for realistic local simulation
  - Configurable simulation method via `sim_method` parameter
  - Supports all AerSimulator methods: `statevector`, `matrix_product_state`, `tensor_network`, etc.
  - Enables testing quantum algorithms under realistic noise without hardware queue times
  - No IBM Quantum compute credits consumed (simulation runs locally)
  - Updated `qbiocode.utils.qutils.get_backend_session()` to support noisy backends
  - Updated `qbiocode.utils.ibm_account.py` for improved credential handling
  - Documentation updated with comprehensive examples and best practices

- **Data Generation**: New blob dataset generator
  - `generate_blobs_datasets()`: Create isotropic Gaussian blob datasets
  - `generate_default_blobs_datasets()`: Quick generation with default parameters
  - Follows QBioCode data generation patterns
  - Useful for clustering and classification benchmarks

- **Evaluation Metrics**: Added generalized `evaluation_metrics()` function to `qbiocode.evaluation.model_evaluation`
  - Supports multiple metrics: accuracy, brier, f1, precision, recall, auc
  - Configurable via `metrics` parameter (default: ['accuracy', 'brier'])
  - Backward compatible: returns (accuracy, brier) tuple by default
  - Supports both binary and multi-class classification
  - Provides calibration quality assessment via Brier score
  - Handles edge cases (e.g., single class in test set)
  - Now available in main API (previously only in tutorial helpers)

- **Quantum Ensemble Learning**: New unified quantum ensemble classifier
  - `compute_qensemble()`: Quantum ensemble with configurable construction methods
    - Implements quantum ensemble using controlled operations and superposition
    - Two ensemble methods via `ensemble_method` parameter:
      - `"swap"` (default): Fixed controlled-SWAP operations (faster, deterministic)
      - `"random_unitary"`: Haar-random unitaries (more general, potentially better generalization)
    - Three ensemble modes: balanced, unbalanced, and pair_sample
    - Support for configurable ensemble depth (d) and operations per qubit (n_swap)
    - Quantum cosine similarity classifier using SWAP test
  - New utility functions in `qbiocode.utils`:
    - `normalize_data()`: Normalize data for quantum state encoding
    - `label_to_array()`: Convert binary labels to one-hot encoding
    - `prepare_training_set()`: Prepare balanced training subsets
    - `retrieve_probabilities()`: Extract probabilities from measurement counts (generic quantum utility)
    - `execute_circuit()`: General-purpose Aer simulator execution (reusable across quantum algorithms)
  - Based on Macaluso et al., "A variational algorithm for quantum ensemble learning" (2023)
  - Integrated from tutorial/QEnsemble with full API compatibility
  - Code organization: Extracted reusable functions to utils for broader applicability

- **Testing Infrastructure**: Comprehensive test suite for core functionality
  - `tests/test_data_generation.py`: Tests for data generation utilities
  - `tests/test_file_utilities.py`: Tests for file operations and utilities
  - `tests/test_generator_dispatch.py`: Tests for generator dispatch logic
  - `tests/conftest.py`: Pytest configuration and fixtures
  - Test coverage for utility modules and data generation helpers
  
- **Code Quality Tools**: Enhanced development tooling
  - `isort` integration for consistent import ordering
  - Configuration in `pyproject.toml` for isort settings
  - Added to `dev` and `all` dependency groups

- **Documentation**: Testing instructions in README
  - Added "Running Tests" section with pytest usage
  - Instructions for installing development dependencies

- **`qbiocode.tutorial_data_path(filename)`** — one documented resolution path for
  tutorial fixtures, replacing the copy-pasted snippet each notebook carried. It
  searches `$QBC_DATA` (colon-separated) first, then every fixture directory of a
  source checkout, locating that checkout both from the installed package *and* by
  walking up from the current working directory — so it resolves for an editable
  install, for a normal install used inside a clone, and for a notebook run from its
  own subdirectory. On failure it raises a `FileNotFoundError` listing every directory
  it tried plus the three ways to fix it. `tutorial_data_dirs()` exposes the search
  path. When `$QBC_DATA` is set but does not hold the requested file, the file is still
  returned from wherever it was found *and* a `WARNING` is logged naming both
  directories — silently reading a fixture out of a directory the caller did not name
  is how a stale copy gets used for a whole session.

- **Four QuVINE tutorial notebooks**:
  - `tutorial/QuVINE/example_quvine.ipynb` — the QuVINE API end to end on a synthetic
    graph: `list_methods`, `embed`, `evaluate_graph`, node classification.
  - `tutorial/QuVINE/quvine_sc_t_vs_mono.ipynb` — T-cell vs. monocyte embedding from an
    `.h5ad` fixture.
  - `docs/source/tutorials/QuVINE/quvine_sc_cd4_vs_cd8.ipynb` — the published
    single-cell walkthrough (CD4 vs. CD8), linked from `tutorials.md` and
    `apps/quvine.rst`.
  - `tutorial/QProfiler/sc_binary_quvine_2x2_qprofiler.ipynb` (mirrored under
    `docs/source/tutorials/QProfiler/`) — QuVINE driven through
    `qbiocode.get_embeddings` exactly like `pca`/`umap`, benchmarked by QProfiler in a
    2x2 classical/quantum design.

  Each declares `pip install "qbiocode[quvine]"` in its first cell, since a plain
  `pip install qbiocode` no longer pulls QuVINE's dependencies.

- **Three single-cell fixtures** under `tutorial/QuVINE/datasets/`
  (`pbmc5k_graph_t_vs_mono.h5ad`, `pbmc5k_small_t_vs_mono.h5ad`,
  `pbmc5k_small_lymphoid_vs_myeloid.h5ad`, 13 MB total). The latter two are the
  provenance of the `sc_binary/*.csv` matrices QProfiler's single-cell tutorial already
  shipped, which could not otherwise be regenerated without the raw 10x matrix and
  scanpy/leidenalg. `pbmc5k_small_cd4_vs_cd8.h5ad` was deliberately *not* duplicated
  here — the copy under `tutorial/QProfiler/data/` is byte-identical, and
  `tutorial_data_path()` finds it from either tree.

- **`tutorials.md`**: a QuVINE gallery section (§4, with the QProfiler-2x2 notebook as a
  subsection) and toctree entries for both new published notebooks. Sections 4-6
  (Quantum Ensemble, QPL, PQK-OV) renumber to 5-7. `apps/quvine.rst` now links both
  notebooks directly instead of only the gallery page.

- **The last two QuVINE notebooks are now published, not merely present in `tutorial/`.**
  `docs/source/tutorials/QuVINE/` gains copies of `example_quvine.ipynb` (11 cells,
  4 figures: 12 embedding methods x 3 planted-community SBMs x 5 iterations, then the
  graph-complexity-vs-macro-F1 correlation) and `quvine_sc_t_vs_mono.ipynb` (8 cells,
  2 figures: seed->eval node classification and seed->target ranking on the 800-cell
  two-view graph, scored against a no-embedding label-spreading baseline and
  degree-/distance-matched null controls). Both ship with complete committed outputs,
  because `nbsphinx_execute = 'never'` means an unexecuted notebook publishes as a page
  that stops partway. Neither copy contains an absolute filesystem path.

  `tutorials.md` §4 now presents all four QuVINE notebooks as `####` subsections in
  reading order -- synthetic-graph API walkthrough, CD4 vs. CD8, T vs. monocyte, then
  QuVINE-through-QProfiler -- each with its own **What You'll Learn** and **Key
  Concepts** block, and the toctree gains the two new pages.

  Publishing `example_quvine.ipynb` also surfaced 127 KB of dead
  `metadata.widgets` state in it -- 165 tqdm progress-bar widget records with **no**
  widget output to render, since every progress bar had already been captured as plain
  text. nbsphinx warned `nbsphinx_widgets_path not given and ipywidgets module
  unavailable`, which is fatal under `sphinx-build -W`, and the state would otherwise
  have shipped to Pages unrendered. The block is removed from both copies (nothing else
  in either file changed: same 25 cells, same execution counts, same outputs byte for
  byte). Adding `ipywidgets` to the `[docs]` extra would have silenced the warning while
  keeping the dead payload, so it was deliberately not done.

- **`tests/test_docs_structure.py`: two guards on notebooks that become site pages**
  (19 static checks, up from 17). One rejects `metadata.widgets` with no corresponding
  widget output -- the condition above, which would otherwise return the next time
  anyone re-runs a notebook with a progress bar and breaks the `-W` docs build. The
  other rejects any `/Users/...`, `/home/...` or `/private/tmp/` string anywhere in a
  notebook under `tutorial/` or `docs/source/tutorials/`, so the path scrub cannot
  silently regress.

- **`deploy-docs` job** in `.github/workflows/ci.yml`. On a push to `main` it downloads
  the HTML the `docs` job already built and publishes it to the `gh-pages` branch that
  https://qiskit-community.github.io/QBioCode serves, writing `.nojekyll` first so Pages does not
  discard Sphinx's `_static/`, `_images/`, `_sources/` and `_modules/` directories. The
  site was previously updated by committing the rendered output into the repository and
  copying it across by hand. `peaceiris/actions-gh-pages` is used rather than
  `actions/deploy-pages` because the latter requires switching the repository's Pages
  source to "GitHub Actions"; pushing the branch leaves the existing setting working.
  The job is gated on `github.event_name == 'push' && github.ref == 'refs/heads/main'`,
  so a pull request — including one from a fork — cannot publish to the live site.

- **`install-matrix` job** in `.github/workflows/ci.yml`, with a bare `pip install -e .`
  leg and a `pip install -e ".[quvine]"` leg. The `test` job installs `.[dev]`, which
  brings the QuVINE dependencies along, so it can never tell whether the extra is
  actually optional — the one property the extra exists for. Each leg imports the
  package, runs all three console scripts with `--help`, and runs the suite with `-rs`
  so the log shows what skipped: on the bare leg the QuVINE tests must *skip*, not fail.
  An eager `import gensim` added anywhere on the import path turns this job red while
  the `test` job stays green.

- **Extras matrix** in `README.md` and `docs/source/installation.md`. Both documented
  only `[apps]` and `[all]`; `[quvine]`, `[docs]` and `[dev]` were undocumented, and
  nothing said what a bare `pip install qbiocode` includes. `installation.md` gains a
  QuVINE section explaining why the extra is all-or-nothing, what still works without
  it, the actual `QuvineDependencyError` message, the `setuptools<81` pin, and the
  `brew install cmake` prerequisite for `ripser` on macOS.

- **`api_overview.rst`**: a *Graph Embeddings (QuVINE)* section documenting `embed`,
  `list_methods`, `resolve_method`, `is_transductive` and `evaluate_graph` alongside the
  transductivity contract, and a *Preprocessing* subsection for `scale_train_test`. New
  public API added in this release was otherwise reachable only by knowing its name.

- **`tests/test_docs_structure.py`** — 17 static checks on the documentation source, so
  the defects listed under *Documentation build* cannot come back silently. It parses
  the `.rst`/`.md` sources and `ci.yml` rather than running `sphinx-build`, so it works
  without the `[docs]` extra installed: every toctree entry names an existing document,
  no document is orphaned (honouring an explicit `:orphan:`/`orphan: true`), every
  `automodule` target is a real module, `qbiocode.apps` and the other new public modules
  each have a page, `conf.py`'s paths are anchored on `__file__` and its version agrees
  with `qbiocode/version.py`, every optional QuVINE import is mocked, no rendered HTML
  is tracked by git, and the deploy job is gated to pushes on `main` and writes
  `.nojekyll`. The `omegaconf` mock gap above was found by this test, not by review.

- **An integration test tier under `tests/integration/`.** The existing tests call
  functions; these run the package the way a user does — through the console scripts, a
  fresh interpreter, a real `sphinx-build`, a real QProfiler run — because that is the only
  place several of this release's defects were reachable. The `_resolve_scaling`
  `ListConfig` bug under *Fixed* was found by the first of them on its first run.
  - `test_qprofiler_end_to_end.py` — runs the real `qprofiler` entry point on a small
    synthetic dataset with `embeddings=[pca,none]`, `model=[lr,dt]`, `iter=2`, and asserts
    on the `ModelResults.csv` it writes: every embedding × model × iteration produced a
    row, the PCA rows report `n_components` features and the unembedded rows report all
    five, the metrics are finite and in `[0, 1]`, and the classifier actually learned the
    signal. Then the reproducibility contract from the seeding fix: **two runs at the same
    seed are identical frames**, two seeds differ, and the two iterations of one run differ
    from each other — the `split_seed = seed + iter` behaviour, asserted rather than
    assumed.
  - `test_qprofiler_quvine.py` — the same harness with `embeddings=[quvine_rwr]`, so the
    QuVINE embedding path is exercised through the CLI and not only through `embed()`. Also
    asserts the PCA safety net delivers exactly `n_components` columns and that the
    transductivity `UserWarning` actually fires.
  - `test_cli_smoke.py` — every name in `[project.scripts]`, read from `pyproject.toml` so
    a script added later is covered automatically, imports and answers `--help` with exit
    code 0 in a subprocess whose working directory is outside the checkout. This is the
    regression guard for the `cli.py` import defect: a broken entry point fails here.
  - `test_package_surface.py` — `__all__` has no duplicates and every name in it resolves;
    the star-import surface matches it exactly; the four documented top-level helpers are
    importable; and a subprocess asserts that `import qbiocode` leaves every `[quvine]`
    dependency unimported, which is what makes the extra genuinely optional.
  - `test_docs_build.py` — runs Sphinx into a temporary directory and asserts the
    structural warning classes are empty and the expected pages were written, plus that the
    `BUILDDIR` in `docs/Makefile` is the path `ci.yml` uploads. Skips when pandoc or
    nbconvert's `rst` template is unavailable, so a missing docs toolchain reads as a skip
    rather than a documentation failure.
  - `test_notebook_execution.py` — executes the two fast tutorials end to end (`slow`,
    deselected by default) from a copy in a temp directory, so a passing run cannot mutate
    the checkout. Alongside them, cheap always-on checks assert no committed notebook is
    *half* executed: a cell counts as run if it has either an execution count or outputs,
    and every notebook must be entirely blank or entirely complete. The five that are
    genuinely truncated — the two QEnsemble copies, the two 2×2 QuVINE copies, and the
    single-cell CD4/CD8 notebook, which need a quantum backend, a long ensemble sweep, or
    `anndata` — are strict `xfail`s with the reason recorded, so they are visible instead
    of silently tolerated, and a stale entry fails the test.

- **`tests/test_graph_evaluation_edges.py`** — 38 tests putting `evaluate_graph` on the
  degenerate graphs a caller will eventually hand it: empty, single node, self-loop,
  disconnected, complete, star, weighted, directed. It pins the *contract* rather than a
  column schema, since which metrics are computable legitimately varies with the graph: one
  summary row, no infinities, byte-identical output across two calls, and undefined metrics
  absent or `nan` rather than a plausible-looking `0.0`. A handful of known answers are
  checked outright — K5 has density 1.0 and modularity 0.0, two disjoint triangles have
  modularity 0.5 and a zero normalized spectral gap.

- **`tests/test_leakage_contract.py`, `tests/test_split_reproducibility.py` and
  `tests/test_pqk_cache_key.py`** — one module per machine-learning defect fixed in this
  release, asserting the property rather than the implementation. Scaling a fixed train set
  alone and with a wildly shifted test set appended must give *identical* train output;
  identical seeds must give identical split indices and different seeds different ones; and
  two `compute_pqk` calls differing only in `encoding`, `reps` or `entanglement` must write
  different cache files, with a shape-mismatched cache rejected rather than loaded.
  `test_split_reproducibility.py` also covers the estimator half of the seeding fix:
  `model_run` must record `estimator__random_state == seed` for every model that takes one,
  must leave `nb` (which takes none) alone rather than crashing on an unexpected keyword, and
  must let a `random_state` in the config win. It asserts the recorded parameters rather than
  the metrics on purpose — whether an unseeded estimator actually changes its answer depends
  on there being a tie to break, so a metric comparison would pass or fail by luck. The
  end-to-end run in `tests/integration/test_qprofiler_end_to_end.py` is what caught the
  defect, and it caught it intermittently for exactly that reason.
  Alongside the existing unseeded-splitter guard, the module now walks every estimator
  construction in the package with `ast` and fails on any `DecisionTreeClassifier`,
  `RandomForestClassifier`, `MLPClassifier`, `LogisticRegression`, `SVC`, `XGBClassifier`
  (and the ensemble variants) built without a `random_state`. That is the check that scales:
  each `compute_*` module looked correct on its own, with `random_state=None` as a documented
  default that nothing filled in — the defect only appears when all the call sites are read
  together. Replayed against the pre-fix tree it reports eight offenders; against this one,
  none.

- **QuVINE documentation at QProfiler/QSage parity.** The QuVINE app page was 199 lines
  against `profiler.rst`'s 974 and linked 2 of its 4 notebooks.
  - `docs/source/apps/quvine.rst` gains **How QuVINE Works** (the view → walk → corpus →
    SGNS → fusion pipeline, with a stage-by-stage reading of `pipeline.py`), a
    **command-line flag table**, an **Outputs** section naming every file a run writes and
    every key in `embedding_meta.json`, a **Configuration** section, and links to all four
    QuVINE notebooks instead of two.
  - New page `docs/source/apps/quvine_config.md` documents the 314-line shipped config
    section by section — and, first of all, **which of the three entry points reads which
    sections**. `embed()` and the `quvine` CLI read `views`, `walks`, `train`, `min_count`,
    `fusion` and `experiment.base_seed`; `data_path`, `graph`, `disease`, `runtime`,
    `preprocess`, `baselines`, `analysis` and `evaluation` are the Hydra research
    pipeline's alone. It also records the `${now:...}` Hydra-only resolver, the
    `preprocess.subgraph` → `preprocess.subsample` schema migration, and the six supported
    `fusion.method` values (`concatenate` is not one, despite appearing in older configs).
  - New `:ref:`graph-complexity-measures`` section grouping all 88 `evaluate_graph`
    columns by family — Laplacian spectrum, eigenvector localization, diffusion and walk
    spectra, quantum composites, paths, degree/centrality concentration, community,
    Ollivier-Ricci curvature, effective resistance, persistent homology, label-aware —
    the graph analogue of `profiler.rst`'s `:ref:`data-complexity-measures``, and
    cross-linked from `apps.rst` and `tutorials.md`.
  - `README.md` gains a QuVINE Applications subsection, and its tutorial list goes from
    5 entries to the 12 the site actually publishes.

### Changed
- **Code Formatting**: Applied consistent code style across entire codebase
  - Ran `black` formatter on all Python files
  - Ran `isort` for standardized import ordering
  - Fixed invalid escape sequences in visualization module
  - Improved code readability and maintainability

- **CI/CD Improvements**: Stabilized continuous integration pipeline
  - Updated GitHub Actions workflows to Node.js 24
  - Fixed CI code quality checks for import ordering
  - Fixed CI type-check issues
  - Fixed documentation build process
  - Fixed Pandoc compatibility issues
  - Enhanced workflow reliability across all platforms

- **Testing**: Improved test reliability
  - Fixed path-order assumptions in duplicate-file tests
  - Tests now work consistently across different file systems

- **Packaging**: dependencies are now tiered under `requirements/`
  - `requirements/requirements-base.txt` is the single source of truth for the
    package's runtime dependencies; `pyproject.toml` reads it via
    `[tool.setuptools.dynamic]`. It ships in the sdist, so building from an
    sdist no longer risks an empty dependency list.
  - `requirements/requirements-quvine.txt` backs the new `[quvine]` extra,
    `requirements/requirements-docs.txt` backs `[docs]`, and
    `requirements/requirements.txt` installs a complete development environment.
  - The root `requirements.txt` is now a pointer to
    `requirements/requirements.txt`, so the documented
    `pip install -r requirements.txt` keeps working unchanged.
  - `[all]` is now defined as `qbiocode[apps,quvine,docs,dev]` instead of a
    hand-maintained copy of every other extra's contents, which had already
    drifted out of sync.

- **Packaging**: no lock file is committed, and `requirements/requirements-lock.txt`
  is gitignored. The development repository carried a 162-line `pip freeze` labelled as
  "the validated environment"; it was taken on macOS/arm64 under Python 3.12.4, while
  QBioCode is tested on three operating systems × three Python versions, so it was
  wrong on eight of those nine combinations while reading as authoritative on all
  nine. It had also outlived the dependency set it froze — it still pinned
  `tensorflow==2.21.0` and `keras==3.15.0`, and covered neither the docs tier nor the
  single-cell packages (`scanpy`, `anndata`, `igraph`, `leidenalg`). The tiered
  `requirements/*.txt` files are the single source of truth instead, and
  `requirements/requirements.txt` documents the four commands that produce a lock in
  the environment that will actually use it.

- **Packaging**: `setup.py` is now a 33-line shim that passes no arguments, so
  `pyproject.toml` is the only place project metadata is declared. The previous
  189-line version re-declared name, version, dependencies, extras, classifiers
  and entry points, and parsed `requirements.txt` by looking for a
  `# Documentation` comment to split runtime from docs dependencies. Duplicated
  metadata is what allowed the two files to disagree.

- **Documentation**: `docs/build/` is now the single build directory. `docs/Makefile`
  sets `BUILDDIR = build` and CI uploads `docs/build/html/`, but ~250 files of rendered
  HTML were committed under `docs/_build/html/` — a second, hand-maintained copy that
  no tool wrote to. Both directories are now gitignored and the `deploy-docs` job
  publishes the build output instead. See *Documentation build* under Fixed.

- **CI**: the docs job's `Build documentation` step no longer sets
  `continue-on-error: true`, so a docs build that fails now fails CI. The artifact
  upload step is likewise no longer allowed to fail silently, since `deploy-docs`
  consumes that artifact. The `lint` job's `black`/`isort`/`mypy` steps keep
  `continue-on-error` — the tree is not fully formatted, and changing that is a separate
  piece of work.

- **Testing**: `pytest` no longer requires `pytest-cov` to start. `addopts` hardcoded
  `--cov=qbiocode`, so a bare `pytest` failed outright with `unrecognized arguments` in
  any environment that installed the package rather than the `[dev]` extra. CI passes
  `--cov` explicitly instead. `addopts` now carries `-m 'not slow and not
  requires_quantum'`, and both markers are registered: `slow` for the minutes-long
  notebook executions and `requires_quantum` for anything needing a real backend and
  credentials. The default run stays fast; the excluded tiers run with `pytest -m slow`
  and `pytest -m requires_quantum`.

- **Dependencies**: removed `tensorflow` from the dependency set. Nothing in
  `qbiocode/`, the tutorials, or the docs imports TensorFlow or Keras — the only
  remaining reference was a stale entry in `autodoc_mock_imports`. This removes
  roughly 600 MB from a default install. `qbiocode/embeddings/compute_autoencoder.py`,
  the one module that could have needed it, is written against PyTorch, which
  remains a dependency.

- **Tutorials no longer label an empirical score difference "quantum advantage".** The
  term names a complexity-theoretic result; what these notebooks measure is a per-dataset
  score gap between quantum and classical arms on small simulated data, which is not
  evidence of one. `QPL_example.ipynb` renames section 9 to *Quantum vs. Classical
  Comparison*, its output figure to `{tag}_quantum_vs_classical.png`, and its prints to
  say which datasets a quantum model ranked first on; `QEnsemble_example_blobs.ipynb`
  renames its *Quantum Advantage* takeaway to *Quantum Ensembling*, since superposition
  over classifiers is a mechanism rather than a demonstrated advantage;
  `quvine_sc_cd4_vs_cd8.ipynb` renames its `quantum_adv` column, comments, and plot axis
  to *quantum recall gain*, which is what the quantity is — `recall(ctqw,dtqw) −
  recall(node2vec)`. `docs/source/tutorials.md` follows the notebooks it describes.

  The QuVINE notebook was re-executed because its axis label is rasterized into a
  committed figure, and the re-run moved its numbers materially — one seed-budget stratum
  went from p=0.005 to p=0.376, and several bootstrap Spearman correlations flipped sign.
  Nothing about the rename caused that: `qbiocode/apps/quvine/embedding/word2vec.py`
  trains gensim `Word2Vec` with `workers=8` and accepts no `seed`, and
  `baselines/node2vec.py` takes a `seed` but also uses `workers=8`, so no SGNS embedding
  in QuVINE is reproducible run to run however `base_seed` is threaded through the walks.
  The notebook's prose already tells readers to trust only intervals that clear zero, so
  it is not contradicted, but the per-stratum significance stars in §7 are not stable and
  should not be read as findings. Fixing this means `workers=1` plus an explicit `seed`,
  which is roughly an 8x slowdown on every SGNS embedding and would change every committed
  QuVINE number — deliberately left out of a rename.

  Two mentions are deliberately kept. `QEnsemble_example_blobs.ipynb` retains "theoretical
  connections to quantum advantage" and "investigate theoretical quantum advantage" in its
  further-work lists, where the term is used correctly, and `docs/source/background.md`
  retains its discussion of the concept along with the Huang et al. paper title. The
  `evaluate_graph` keys `quantum_advantage_score`/`_arithmetic`/`_geometric`/`_harmonic`
  and `compute_quantum_advantage_metrics` are also unchanged: they are public API, and
  renaming them would break callers and any saved result CSV for a wording change.

### Fixed

#### 🚧 The documentation site had not been rebuilt since 12 July
- **`deploy-docs` was never reached.** The job is correct and correctly gated
  (`push` to `main`, never on a pull request), but it declares `needs: docs` -- and the
  `docs` job had been failing on every push to `main`. GitHub therefore *skipped* the
  deploy rather than failing it, which is the quiet kind of broken: CI showed a red X
  attributed to a docs warning while the published site silently stayed months behind.
  `gh-pages` still carried a build whose commit message predates the job's own message
  template, so nothing that job produced had ever been published.
- **The `docs` job failed on six warnings, not on anything structural.** `docs/Makefile`
  passes `-W --keep-going`, so any warning is fatal: three malformed docstring
  constructs surfacing in generated `api/` pages, and three duplicate link targets in
  `docs/source/workshops/ISMB_2026.rst` where named references (`` `Profile <...>`_ ``)
  repeated the same label. The workshop links are now anonymous (`` `...`__ ``), which is
  what a repeated label requires.
- **The docs job built on a Python the pinned Sphinx does not support.** It requested
  3.10 while the `[docs]` extra pins `sphinx<9`; Sphinx 8.2 requires Python >= 3.11, so
  on 3.10 pip quietly backtracked to Sphinx 8.1.3 -- a toolchain nobody had verified the
  build against, selected by the resolver rather than chosen. That job now uses 3.12,
  matching the other single-version jobs. The supported-Python floor is still asserted
  by the test matrix, which is where it belongs.

#### 🔗 Documentation URLs pointed at a repository that had moved
- **The project moved from `IBM/QBioCode` to `qiskit-community/QBioCode`, and the old
  Pages host stopped answering.** `https://ibm.github.io/QBioCode/` returns 404 -- a
  transferred repository redirects on `github.com`, but its `github.io` Pages site does
  not follow. 81 links across 28 files still named the old host or org, including the
  GitHub button that the theme renders on every page of the site.
- All of them now name `qiskit-community`. IBM remains where it is accurate: Apache
  license headers, author affiliations, `research.ibm.com` profiles and IBM Quantum
  references are untouched.
- Five malformed links in `README.md` are repaired at the same time, all of them
  reachable only by reading the rendered page: a missing `/` produced
  `.../QBioCodeapps/sage.html`; a `[url(url)` typo rendered as literal text instead of a
  link; the issue tracker was pointed at the Pages host, which does not serve `/issues`;
  and a stray blank line split one sentence across a paragraph break.

#### 🩹 `quvine` read an edge list's header row as an edge
- **`quvine --edgelist` now recognizes and skips a header row.** `_load_graph` passed
  `header=None` to `pd.read_csv`, so the `source,target` first line that the tool's own
  `--help` and documentation both show was loaded as an edge between a node named
  "source" and a node named "target". On the 34-node karate graph the run reported 34
  nodes without the header and **36 with it**, embedded the two phantom nodes, and exited
  0 either way — nothing in `embedding_meta.json` or on stderr said which had happened.
  A header is now dropped and the fact reported on stderr.
- Detection matches a **closed set of conventional column names** (`source`/`target`,
  `from`/`to`, `gene1`/`gene2`, `node_a`/`node_b`, ...), both fields required, rather than
  sniffing. Every heuristic considered has a failure mode that is worse than the bug:
  "row 0's endpoints appear nowhere else" deletes the first edge of a sparse matching, and
  "row 0 is non-numeric" deletes the first edge of every string-labelled graph — and a
  deleted real edge leaves no phantom node to notice. Single-letter names (`u`, `v`, `a`,
  `b`) are deliberately excluded as plausible node labels.
- New `--header {auto,yes,no}` forces the decision for anything the set does not cover.
  A file containing only a header is now an error rather than an empty graph.

#### 🩹 A nested `.. toctree::` swallowed the rest of its page
- **`tests/test_docs_structure.py`'s toctree parser now ends a directive body where
  reStructuredText does** — at the first non-blank line indented no further than the
  directive — instead of only at column 0. That held for as long as every toctree in the
  tree sat at column 0. Restoring `better_apidoc` broke it: the pages it generates indent
  the whole body under one top-level `.. automodule::`, so nothing dedents to column 0,
  and the parser ran to the end of the file yielding every `list-table` row as a toctree
  entry — one of them a dataclass `__repr__` long enough to raise
  `OSError: File name too long` when the existence check `stat()`ed it as a filename.

#### 💥 Every XGBoost fit crashed the interpreter on macOS
- **`import qbiocode` no longer loads `torch` before `xgboost`.** `xgboost` and `torch`
  each vendor their own copy of `libomp` under the same install name, so importing both
  maps two independent LLVM OpenMP runtimes into one process. The first to initialise
  claims the process-wide runtime state, and the second dies when it opens a parallel
  region — `EXC_BAD_ACCESS` in `libomp.dylib` at `__kmp_suspend_initialize_thread`, with
  no Python traceback and nothing to catch. `qbiocode/__init__.py` imports `.embeddings`
  before `.learning`, and `.embeddings` eagerly imported `ConvAutoencoder`, whose first
  line is `import torch` — so torch won the race in *every* process that imported the
  package, and every XGBoost fit that followed was a segfault. This took out QPL's
  `qpl_xgb` arm, `compute_xgb`, and so any QProfiler run configured with an XGBoost
  model; from a notebook it surfaced only as `DeadKernelError: Kernel died`, and from a
  script as a silent exit 139.

  Two changes fix it. `ConvAutoencoder` is resolved through a lazy module `__getattr__`,
  so torch is imported only if that class is actually used — nothing in the package uses
  it, so the eager import bought nothing. And `qbiocode/__init__.py` now calls
  `qbiocode.utils._openmp.preload_openmp_libraries()` before any submodule import, which
  initialises xgboost's runtime first and keeps a caller's later `import torch` safe.
  That module records the measurements: `xgboost`-then-`torch` fits fine,
  `torch`-then-`xgboost` segfaults, and turning parallelism down does *not* help —
  neither `n_jobs=1` on the estimator nor on the surrounding search, because the fault is
  inside the OpenMP runtime, below joblib. Guarded by
  `tests/test_openmp_import_order.py`.

#### Grid search over a subset of a model's hyperparameters
- **`compute_*_opt` no longer requires a config to enumerate every hyperparameter.**
  All seven tuned learners (`dt`, `lr`, `mlp`, `nb`, `rf`, `svc`, `xgb`) took one keyword
  per tunable hyperparameter, each defaulting to `[]`, and handed all of them to
  `GridSearchCV` unconditionally. Any config naming a subset therefore died inside
  sklearn on whichever parameter it had left alone:

      ValueError: Parameter grid for parameter 'colsample_bytree' need to be a
      non-empty sequence, got: []

  The message names a parameter the user never wrote and points at sklearn rather than at
  the config, and it made a deliberately small grid inexpressible — trimming a demo
  config was indistinguishable from corrupting it, which is why the QPL tutorial shipped
  a 2430-combination XGBoost grid. `qbiocode.learning._grid.build_param_grid` now keeps
  only the values actually supplied, leaving every unmentioned hyperparameter at the
  estimator's own default, and raises a message naming the config block to add and the
  `grid_search: False` opt-out when *nothing* was supplied. Guarded by
  `tests/test_grid_search_partial.py`.

  Two related fixes came with it. A bare string is wrapped rather than searched
  character by character — `max_features: sqrt` in YAML was previously searched as
  `['s', 'q', 'r', 't']`, four invalid values that produced no error and a meaningless
  `best_params_`. And the `[]` defaults are now `None`: a shared mutable default is a
  hazard whether or not this code happened to mutate it.

- **`compute_xgb_opt` reports a `bootstrap` grid instead of silently doubling.**
  XGBoost has no `bootstrap` parameter, but its sklearn wrapper accepts unknown keyword
  arguments without complaint, so the shipped tutorial config was never an error — it
  just searched twice as many combinations, every duplicate returning the same model. It
  now warns and names `subsample` as the parameter that actually samples rows.

#### ⚠️ Train/test contamination in QProfiler — results change
- **QProfiler no longer scales the test set with test-set statistics.** Previously
  `qprofiler.py` called `scaler_fn` separately on each split, which fit a *fresh*
  `MinMaxScaler` on `X_test`. The test features were therefore normalized using the test
  set's own min/max — both a use of held-out information and a train/test distribution
  mismatch, since the model was trained under a different transform than it was evaluated
  under. One scaler is now fit on `X_train` and applied to `X_test` via the new
  `qbiocode.scale_train_test`.
- **Consequence: metrics produced by QProfiler before this change are not comparable to
  metrics produced after it.** Any accuracy/F1/AUC numbers carried over from an earlier run
  should be regenerated. Every notebook that carries QProfiler numbers now states this at
  the top; see *Notebooks* below for why they are annotated rather than re-executed.
- **`seed` now actually controls the train/test split.** `train_test_split` was called with
  no `random_state`, so iterations were irreproducible and the configured `seed` was silently
  ignored for splitting — despite an inline comment claiming the splits "will be based on the
  seed". Each iteration now uses `random_state = seed + iter`: distinct across iterations,
  deterministic across reruns, and independent of other RNG consumers.
- **`seed` now reaches the estimators too, not just the split.** Fixing the split was only
  half of it. `qprofiler` sets `np.random.seed(seed)` in the parent process, but `model_run`
  fans the models out with joblib, whose loky workers are fresh interpreters seeded from OS
  entropy — so every estimator left at `random_state=None` drew a different random state on
  each run. `DecisionTreeClassifier` permutes the features before choosing a split, so a tie
  between two equally-good splits broke either way, and two runs at seed 7 came back with
  0.889 and 0.944 accuracy on the same row. `model_run` now fills in
  `random_state = seed` for every model whose function accepts one (`dt`, `lr`, `rf`, `mlp`,
  `svc`, `xgb`, and their `_opt` grid-search variants, which had no `random_state` parameter
  at all); a `random_state` set explicitly in the config still wins, and models without one
  (`nb`) are left alone. The worker also re-establishes `np.random.seed` and
  `algorithm_globals.random_seed`, which is what `compute_qnn`'s initial weights read.
  Setting the global seed alone would not have been enough: joblib batches tasks, so how far
  an earlier task advanced a shared stream depends on timing. **Any metric produced before
  this change for `dt`, `rf`, `mlp`, `xgb`, or a grid search is not exactly reproducible.**
- **`create_xgb_model` in the QPL pipeline ignored its own `seed` argument.** Every sibling
  (`create_lr_model`, `create_rf_model`, `create_mlp_model`, `create_svc_model`) passed
  `random_state=seed` to its estimator; the XGBoost one passed the seed to
  `RandomizedSearchCV` and left the classifier unseeded, while the search grid varied
  `subsample` and `colsample_bytree` — both of which sample rows and columns at random. Now
  seeded like the rest.

#### ⚠️ Undefined measurements are now `nan`, not a plausible number — results change
Several functions returned an achievable value when they could not compute anything at
all, which made "undefined" indistinguishable from a real result in an aggregated table.
Every one of these now returns `nan`, and every consumer aggregates nan-aware. **Any
table or ranking built from the metrics below should be regenerated**; affected notebooks
have been re-executed.

- **Link-prediction ranking metrics no longer report `0.5` for an undefined AUC.**
  `reproducibility.method_adapters.evaluate_link_prediction` returned
  `{"auc_roc": 0.5, "auc_pr": 0.5, "f1": 0.0}` whenever the evaluation split held a single
  class, or whenever `roc_auc_score` raised. `0.5` is the score of a random ranker — a real,
  and quite common, outcome — so a method that could not be scored sorted level with
  methods that genuinely failed to learn. It now returns `nan` for all three, logs which
  case occurred, and validates that the supplied node indices exist in the embedding matrix
  rather than raising `IndexError` several frames deeper.
- **`summarize_link_prediction_results` is nan-aware.** It averaged with `np.mean`, so one
  undefined method turned every aggregate into `nan` and erased the methods that had
  succeeded. Aggregates now use `np.nanmean`/`np.nanstd` and each metric carries an
  `n_defined_<metric>` count, so a reader can see how many methods the mean is over. An
  empty result set yields `nan` rather than `0.0`.
- **`evaluate_graph` metrics that cannot be computed are absent or `nan`, never `0.0`.**
  `modularity` was set to `0.0` on the exception path and for an edgeless graph;
  `path_length_ratio` likewise. Modularity is `Q = Σ_c (e_c/m − (d_c/2m)²)`, undefined at
  `m = 0`, and `0.0` states "no community structure beyond chance" — a finding. Both now
  report `nan`. `compute_community_metrics` previously disagreed with itself on this,
  returning `modularity: 0.0` alongside `approx_conductance: nan` on the same line.
  (The `0.0` `path_length_ratio` for a *disconnected* graph is retained deliberately: that
  is a measurement, not a failure.)
- **`compute_quantum_advantage_metrics` on an empty graph returns `nan` for all ten
  metrics**, including the composite `quantum_advantage_score`, instead of `0.0`. A `0.0`
  score is rankable, so an empty graph previously sorted alongside genuinely unpromising
  ones. This function has no callers outside `graph_evaluation.py`, so no existing ranking
  changes order — only the values reported for degenerate inputs.
- **`netmf` no longer synthesizes an embedding from random noise.** When its truncated SVD
  failed to converge, and when the deepwalk-matrix construction failed, it returned
  `np.random.randn(n, dim) * 0.01` — which the registry scored, logged and ranked as a
  result. Both paths now raise `RuntimeError` naming the stage that failed, chained from the
  underlying `ArpackError`/`LinAlgError`. A test asserts the module's source contains no
  random-matrix fallback.
- `evaluate_graph(None)` raises `TypeError` instead of returning a size-only summary
  reporting "0 nodes" — which was indistinguishable from a genuinely empty graph. An empty
  `nx.Graph()` remains a valid input and still yields the size-only summary with a warning.
  The two metric families also degrade independently: if one fails, its columns are simply
  absent from the returned frame and a `UserWarning` names the exception type, rather than
  the frame being padded with fabricated values.

#### Silent no-ops reported as success
- **`get_optimizer("L_BFGS_B")` never worked.** The branch read
  `optimizer == L_BFGS_B(maxiter=max_iter)` — a comparison, not an assignment — so it
  constructed the optimizer, discarded it, and left `optimizer` unbound. Every request
  raised `UnboundLocalError: cannot access local variable 'optimizer'`, despite `L_BFGS_B`
  being advertised in the `Literal` type hints of both `compute_vqc` and `compute_qnn`.
- **`get_feature_map` returned its own argument for an unrecognized name.** The if/elif
  chain had no `else`, so `feature_map` stayed bound to the caller's string and failed
  several frames later with `'str' object has no attribute 'num_qubits'`. A mistyped
  `encoding` — `'zz'` for `'ZZ'` — was therefore reported as a qiskit type error.
  `SUPPORTED_FEATURE_MAPS`, `SUPPORTED_ENTANGLEMENTS` and `SUPPORTED_OPTIMIZERS` are now
  module constants in `qbiocode.utils.qutils`, validated at entry and reusable by callers.
  An unknown `entanglement` previously reached qiskit and came back as
  `ValueError: Something went wrong in Rust space`.
- **`scale_train_test` silently returned unscaled data for a mistyped scaler name.** The
  `else` branch fell through to the `"None"` behaviour, so `scaling="minmaxscaler"`
  disabled scaling and reported success. It now raises `ValueError`.
- **QProfiler's `scaling` config was tested with a substring match.** `'True' in
  args['scaling']` accepted `'MinMaxScalerTrue'`, silently ignored the lowercase `['true']`,
  and raised `TypeError: argument of type 'bool' is not iterable` for the most natural YAML
  of all, `scaling: true`. All spellings — bool, `['True']`, `true`/`yes`/`1`,
  `false`/`no`/`0`/`none`, or a scaler name — are now resolved explicitly, and anything else
  is an error rather than a quietly disabled transform.
  The first version of that fix then rejected the shipped `config.yaml`'s own
  `scaling: ['True']`: it unwrapped single-element sequences behind
  `isinstance(value, list)`, and Hydra hands the config over as
  `omegaconf.ListConfig`, which is a `Sequence` but *not* a `list` subclass. Every
  dict-based unit test passed and the real CLI path raised
  `ValueError: Unrecognized scaling ['True']`. The test is now on `Sequence`, and the
  docstring names the trap. Found by the new end-to-end QProfiler test on its first run —
  no unit test built from plain dicts could have caught it.
- **`compute_pqk` accepted a `primitive` it never used.** `primitive="sampler"` changed only
  the cache fingerprint, not the computation: projected quantum kernels are Pauli expectation
  values and the backend is requested as `"estimator"` unconditionally. That produced two
  cache files holding identical projections and a caller who believed they had measured
  something else. Only `"estimator"` is accepted now, and the message says why.
- **A mistyped `--config` path could produce a successful `quvine` run with the wrong
  settings**, by falling back to the packaged config. The path is now checked with
  `os.path.isfile` before anything else runs.
- **`QSage.__init__` aliased its metric list instead of copying it.**
  `self._available_metrics = self._columns_metrics` followed by an in-place `.sort()` bound
  both names to one list, so sorting the public attribute silently reordered the column list
  used to slice the input frame. It is now `sorted(...)`, which returns a new list.
- A QProfiler run whose `folder_path` matched no dataset, or whose `iter` was `0`, exited
  successfully having produced no output at all — indistinguishable from a run whose models
  all failed. Both are now errors.

#### Errors attributed to the wrong cause
- **Dead sentinel guards removed.** `try: from xgboost import XGBClassifier / except
  ImportError: XGBClassifier = None` (in `qbiocode/__init__.py`, `learning/__init__.py` and
  `learning/compute_pqk.py`) replaced an actionable `ImportError` with
  `TypeError: 'NoneType' object is not callable` at the point of use — a message that names
  neither xgboost nor the fix. xgboost is a declared base dependency, so its absence is a
  broken install. `compute_qpl`'s guard is retained (it holds an optional import) but now
  records the original `ImportError` and quotes it in the message. The PyTorch-Geometric
  guard in `reproducibility/graph_generator.py` does the same.
- **Broad `except Exception` handlers narrowed to the failure they actually handle**, so a
  bug in the handler's own body no longer masquerades as a degenerate graph: graph-metric
  handlers in `baselines/gat.py`, `baselines/graphgps.py` and `evaluation/graph_evaluation.py`
  catch `nx.NetworkXException`; SVD and eigendecomposition handlers in `baselines/netmf.py`,
  `baselines/graphsage.py` and `fusion/fuse_fixes.py` catch
  `(scipy.sparse.linalg.ArpackError, np.linalg.LinAlgError, ValueError)`; the edge-list
  reader in the `quvine` CLI catches the four ways a text table fails to parse. Where a
  handler must stay broad — `evaluate_graph`'s two public metric blocks,
  `reproducibility/validator.py` — it now names `type(e).__name__` in the message and keeps
  the traceback at `DEBUG`, because the alternative is a missing column with nothing to
  trace it back to.
- `evaluation/classification.py` discarded the reason the *requested* label strategy failed
  before falling back to `label_propagation`, so a silently-substituted labelling was
  indistinguishable from the requested one. The substitution is now logged with its cause,
  and if the fallback also fails the warning names both failures rather than only the last.
- `graphgps.py` read precomputed node dictionaries with `d[node]`, raising `KeyError` for a
  node absent from a partial computation; it uses `.get(node, default)` now. `get_nodelist` /
  `_stable_nodelist` validate that a supplied `nodelist` is a permutation of the graph's
  nodes, instead of silently producing misaligned feature rows.
- **Validation moved to the boundary.** `get_embeddings` coerces and shape-checks `X_train`
  and `X_test` up front (a list of lists — the natural thing to pass from a notebook — used
  to fail on `'list' object has no attribute 'shape'`), and the new public
  `qbiocode.embeddings.check_embedding_name` lets a caller validate an entire `embeddings`
  list *before* any work begins, with the identical message; QProfiler uses it for exactly
  that, so a typo in the sixth entry no longer surfaces after the first five have run.
  `compute_pqk` performs eleven argument checks before it creates directories or reads a
  projection cache. `evaluation/model_run.py` rejects unknown model names with the available
  set instead of a bare `KeyError` raised inside a joblib worker, and falls back to estimator
  defaults — with a log line — for a model that has no `<model>_args` config block, which
  `xgb` and `qpl` do not in the shipped `config.yaml`. The `qprofiler`, `qsage` and `quvine`
  CLIs validate every argument before creating an output directory or reading input:
  `--cv 1`, `--test-size 1.0`, `--n-iter 0`, an `--input` that is not a directory, an empty
  `--sep`, and a graph with no nodes are all caught up front.
- `tests/test_error_contracts.py` (43 tests) pins the behaviour above: that undefined
  measurements are `nan`, that a silent no-op is now an error, and that each message names
  the parameter, the value received and the accepted set.

#### Embeddings
- `get_embeddings(..., n_components=None)` — the documented default — raised
  `TypeError: '<=' not supported between instances of 'NoneType' and 'int'` instead of
  defaulting to the feature count. It now defaults to `X_train.shape[1]` as documented.
- `spectral` embedding raised `AttributeError: 'SpectralEmbedding' object has no attribute
  'transform'` on every run. `SpectralEmbedding` has no out-of-sample transform, so it is now
  fit transductively over the combined train+test rows and sliced back. Only feature structure
  participates — no labels — so no label leakage is introduced.

#### Quantum sessions and caching
- Qiskit Runtime sessions in `embed.pqk` and `learning.compute_pqk` are now closed in a
  `finally` block. An exception mid-computation previously leaked the session, leaving jobs
  queued against the user's account.
- **PQK projection caches are now keyed by the feature map that produced them.** The cache
  filename omitted `encoding`, `entanglement`, `reps` and `primitive`, so re-running with a
  different feature map into the same `pqk_projection_dir` silently reloaded the previous
  run's projections and reported them as the new result. A short digest of those parameters
  is now part of the filename.
- Cached projection files are also validated on width (`3 x feat_dimension`), not just row
  count, so a file written with a different feature dimension is rejected rather than reused.

#### QuVINE method resolution and packaging
- **The `quvine` CLI rejected its own default method.** `--method` defaults to
  `quvine_fused`, but the CLI validated it against `list_methods()`, and the five fused
  names (`quvine`, `quvine_fused`, `quvine_sgns`, `quvine_sgns_fused`, `fused`) were
  handled directly inside `embed()` without ever being registered in the alias tables.
  Running `quvine -i edges.csv -o out/` therefore always exited 1 with
  `unknown method 'quvine_fused'`, and `resolve_method("quvine_fused")` raised
  `KeyError`. Fused names are now a first-class kind in
  `qbiocode/apps/quvine/api/aliases.py`: `resolve_method` returns `("fused", "fused")`,
  `list_methods()` includes them (83 names, up from 78), `list_methods("fused")` filters
  to them, and `core.embed` sources its fused-name set from the same table instead of
  duplicating it. The CLI now validates through `resolve_method`, so its message carries
  close-match suggestions and can no longer drift from what `embed()` dispatches.

- **Four QuVINE subpackages were missing from a built wheel.** `walks/`, `corpus/`,
  `utils/` and `configs/` had no `__init__.py`, so `[tool.setuptools.packages.find]`
  skipped them. The modules imported fine from a source checkout — including
  `from qbiocode.apps.quvine.walks.rwr import ...` — which is exactly why this went
  unnoticed; an installed wheel raised `ModuleNotFoundError`. Each directory is now a
  real package, and `tests/test_quvine_packaging.py` fails if any regains modules
  without an `__init__.py`.

- **`import qbiocode` no longer pulls in the `[quvine]` dependency set.**
  `qbiocode/apps/quvine/api/__init__.py` eagerly imported `core`, `config`, `sgns` and
  `targets`, so reaching the stdlib-only `resolve_method` dragged in omegaconf and
  everything behind it. That only appeared to work because `hydra-core` pulls omegaconf
  in transitively. Those attributes are resolved lazily now, which is what lets
  `qbiocode.embeddings` probe method names on a bare install. The `quvine` CLI likewise
  imports `embed` at its point of use, so `--help` and `--list-methods` work without
  the extra.

- **`torch` is declared in the base dependency set only.** It was listed in both the
  base set and the `[quvine]` extra, which implied it was optional. It is declared once,
  in the base set, so a missing torch reads as a broken install rather than a missing
  extra and no wrong `[quvine]` install hint is offered. Note that `import qbiocode` no
  longer *loads* torch — see the OpenMP crash above — but it is still a base dependency,
  because both `ConvAutoencoder` and the QuVINE baselines need it to be present.

- **`QUVINE_METHODS` was documented as empty without the `[quvine]` extra.** Both
  `qbiocode/embeddings/__init__.py` and this changelog said so; the code has always
  listed all 83 names, because resolving a name is stdlib-only and only *running* a
  method needs the extra. The documentation understated what works in a bare
  environment — including the "unknown embedding" error message, which lists the valid
  QuVINE names. Found by running the suite against a simulated bare install rather than
  by reading the code.

- **Stale dependency advice removed.** `embedding/word2vec.py` raised
  `ImportError("... Try: pip install gensim==4.3.0 scipy==1.11.0")`, advising a downgrade
  that conflicts with `qiskit-machine-learning==0.9.0` (which requires numpy>=2.0, while
  gensim<4.4 forces numpy<2.0). It now names the `[quvine]` extra. The unused
  `GENSIM_AVAILABLE` module flag was dropped along with it.

- **Documentation claims corrected.** `qbiocode/apps/quvine/__init__.py` and
  `docs/source/apps/quvine.rst` stated QuVINE was "available on a plain
  `pip install qbiocode` / `git clone` with no extra install step". Both now document the
  `[quvine]` extra and what still works without it. A stale comment about avoiding a
  TensorFlow/Keras import at load time was also removed — no TensorFlow exists in the tree.

- **`qbiocode/apps/quvine/data/` was missing entirely, and with it every registry
  method.** The directory was never committed to the internal repository: an *unanchored*
  `data/` rule in its `.gitignore` matched it at every depth, so it was excluded silently
  and appears nowhere in that repository's history. Because
  `embedding/quantum_filters.py` imports it at module scope — and through it the adapter
  module that builds the method registry — all 69 registry methods died with
  `ModuleNotFoundError: No module named 'qbiocode.apps.quvine.data'`, and
  `qbiocode.apps.quvine.Pipeline` was unimportable. The six modules
  (`data_loader`, `prepare`, `sparsify`, `subgraph`, `random_graphs`,
  `random_graphs_extended`) are recovered from the working tree they were written in and
  are now tracked, restoring the registry, `Pipeline`, and all 15 synthetic graph families
  in `reproducibility.graph_generator`. External's `.gitignore` anchors the rule as
  `/data/`, so this cannot recur here. `graph_complexity.py`, which sat alongside them in
  the standalone QuVINE distribution, is deliberately **not** carried over — nothing
  imports it and `qbiocode.evaluate_graph` supersedes it.
- `data.subgraph.expand_neighborhood` filtered roots absent from the graph only for
  `radius >= 1`; at `radius=0` it returned the roots verbatim and so could hand back nodes
  the graph does not contain. Roots are now filtered before the radius check.
- **Missing optional dependencies were attributed to the wrong feature.**
  `walks/ctqw.py` and `walks/dtqw.py` bound `hiperwalk` at module scope, and `walks/base.py`
  imports both eagerly — so an RWR-only run, which never performs a quantum walk, failed
  reporting *"continuous-time quantum walks (CTQW) requires ... missing: hiperwalk"*.
  `baselines/node2vec.py` had the same shape, and because `baselines/__init__.py` wraps its
  imports in `except ImportError: pass`, that left `run_node2vec` unbound and every registry
  method — netmf, appnp, graphgps — failed with a node2vec error. All three now resolve
  their dependency at call time and each names its own. A test asserts structurally that no
  module under `apps/quvine` calls `require_module` at module scope.
- GraphGPS's missing-`torch_geometric` message told the user to `pip install
  torch-geometric` by hand; it now goes through `require_module` and names the `[quvine]`
  extra like every other one.
- **The method registry printed to stdout.** `run_method` wrote `✓ netmf: 0.00 minutes` /
  `✗ node2vec: FAILED - ...` on every call and logged a full traceback at `ERROR` for a
  failure it then returned to its caller to re-raise. QProfiler calls this once per method
  per iteration. Progress now goes through `logging`, the traceback is available at `DEBUG`,
  and the causing exception travels on `MethodResult.exception` so `api.core` chains it
  (`raise QuvineMethodError(...) from exc`) — the cause reaches the caller through the
  exception rather than a stray log line.
- `get_embeddings` validated nothing. Unknown method names raised `KeyError` or fell through
  to an sklearn assertion; a non-string `embedding`, a non-integer, zero, negative, or
  wider-than-the-input `n_components` all produced errors from deep in sklearn. Each now
  raises `ValueError` at the boundary, naming the parameter, the value received and the
  accepted set, with close-name suggestions for typos (`"pcaa"` → *Did you mean: pca?*).
- `qbiocode.embeddings`' module docstring documented `get_embeddings(X, method='pca',
  n_components=2)` — a signature the function has never had.

#### Plotting no longer reconfigures, blocks, or overwrites
- **`import qbiocode` no longer rewrites the importing program's matplotlib.**
  `visualization/visualize_correlation.py` assigned 27 entries into the global
  `plt.rcParams` at module scope — font family, size, tick geometry, spine visibility,
  and a 600-dpi `savefig` default. Because `qbiocode/__init__.py` imports the module,
  merely importing the package restyled every unrelated figure the caller drew
  afterwards, with nothing in the traceback or the call stack to attribute it to. The
  settings are now `qbiocode.visualization.PUBLICATION_STYLE`, applied for the duration
  of a plotting call through `plt.rc_context` and restored even when plotting raises.
  `qbiocode.publication_style()` returns a copy for callers who want the same look on
  their own figures. `tests/test_plotting_hygiene.py` asserts in a subprocess that
  importing the package leaves `rcParams` untouched, and an AST guard fails if any
  module under `qbiocode/` regains a module-level `rcParams` assignment,
  `matplotlib.use()`, or `sns.set_theme()`.
- **⚠️ `plot_results_correlation` no longer overwrites its own output.** It writes three
  figures, deriving the second and third paths with
  `re.sub(".pdf", "_heatmap.pdf", save_file_path)`. That pattern did not match any
  extension other than `.pdf`, so the derived path *equalled the original*: a caller who
  passed `corr.png` had the scatter plot overwritten by the clustered heatmap and that
  overwritten by the non-clustered one, ending with one file where three were requested
  and nothing to indicate two had been lost. The `.` was also an unescaped regex
  wildcard, so `out/spdf_x.pdf` became `out/_heatmap.pdf_x_heatmap.pdf`. Paths are now
  derived with `os.path.splitext`, any image format works, and a missing output directory
  is created rather than raising. `.pdf` filenames are unchanged.
  `QSage.plot_results` had the same `re.sub('.pdf', '', saveFile)` defect and forced a
  `.pdf` extension regardless of the request, producing files named
  `results.png_f1_score_barplot.pdf`; it now honours the extension it was given.
- **Library code no longer blocks on a window.** `plt.show()` was called unconditionally
  in `visualize_correlation` (three times per call), `apps/quvine/analysis/analyze.py`
  (four sites) and `QSage.plot_results` (two) — hanging a batch QProfiler run under a GUI
  backend, and emitting `UserWarning: FigureCanvasAgg is non-interactive` under `Agg`.
  Every site is now guarded on the backend actually being able to display.
  `plot_results_correlation` keeps `show_plots=True`, since five tutorial notebooks rely
  on it for inline display; `QSage.plot_results` gained `show=None`, meaning "show only
  when not saving", which is what its docstring already claimed. No `matplotlib.use("Agg")`
  guard was added: forcing a backend at import is the same hidden global mutation this
  change removes, and matplotlib already falls back to `Agg` when no GUI toolkit is
  importable.
- **`plot_singular_values` and `spectral_info` saved *after* showing**, so under a GUI
  backend the file was not written until someone closed the window — a batch run stalled
  with nothing on disk. They save first now. `spectral_info` also wrote
  `log_spectrum.png`, `loglog_spectrum.png` and `log_normalized_spectrum.png` into the
  process's current working directory under fixed names, so two runs from the same
  directory silently overwrote each other; it takes an `outdir` argument.
- **Figures no longer accumulate.** Every `plt.close()` — which closes whatever figure is
  *current*, not necessarily the one just drawn — is now `plt.close(fig)` on the specific
  figure, and the touched functions use the explicit `fig, ax = plt.subplots()` API rather
  than the pyplot state machine. This matters because QProfiler plots once per metric per
  embedding per iteration.
- **The plotting functions return their figures.** `plot_results_correlation` returns a
  `CorrelationFigures` named tuple (`scatter`, `scatter_ax`, `clustered_heatmap`,
  `ordered_heatmap`), `QSage.plot_results` returns its list of figures, and
  `plot_singular_values` returns its figure — so a notebook can restyle or re-save
  without recomputing. All are closed before returning, which removes them from pyplot's
  manager while leaving each fully usable for `fig.savefig(...)` and inline display.
- `plot_results_correlation`'s docstring documented `Returns: None`; the module docstring
  of `qbiocode.visualization` documented an `output_dir=` parameter the function has never
  had. Three `print()` calls in it, and one in `QSage.plot_results`, now go through
  `logging`.

#### Packaging and dependency declarations
- Three dependencies were imported by the library but never declared:
  `pyyaml` (`qbiocode/utils/generate_qml_configs.py` and
  `qbiocode/apps/qprofiler/qprofiler_batchmode.py`, which backs the
  `qprofiler-batch` console script), `matplotlib` (five modules, including
  `qbiocode/visualization/visualize_correlation.py`), and `joblib`
  (`qbiocode/evaluation/model_run.py` and `qprofiler_batchmode.py`). Installs
  only worked because `seaborn` and `optuna` happened to pull them in
  transitively. All three are now declared explicitly.
- `MANIFEST.in` referenced `apps/qprofiler/configs`, a path that does not exist
  in this tree — the configs live at `qbiocode/apps/qprofiler/configs`. The
  sdist therefore shipped no Hydra config for QProfiler. Likewise
  `[tool.setuptools.packages.find]` listed an `apps*` include glob that matched
  no package.
- `pandoc` was declared as a pip dependency of the `docs` extra. Pandoc is a
  system binary; the PyPI distribution of that name does not provide it.
  Removed, with the platform install commands documented in
  `requirements/requirements-docs.txt` instead.
- `.gitignore` matched `data/` and `results/` unanchored, so those patterns
  excluded a directory of either name at *any* depth — including
  `docs/source/tutorials/QProfiler/data/`, which holds the committed `.h5ad`
  and `sc_binary/*.csv` fixtures the published notebooks read. Because git
  cannot re-include a file whose parent directory is excluded, those fixtures
  were only addable by virtue of already being tracked. Both patterns are now
  anchored to the repository root.
- Stopped tracking generated QProfiler output under
  `docs/source/tutorials/QProfiler/` (`ModelResults.csv`,
  `RawDataEvaluation.csv`, `results.pkl`, and 500 KB of `pqk_projections/*.npy`).
  The notebooks write these files and read them back within the same run, and
  `nbsphinx_execute = 'never'` means the docs build never executes them. The
  cached projections were additionally unreachable after the PQK cache-key fix
  above, since their filenames predate the feature-map fingerprint.

- **`pqk(..., data_map=True)` could never build its feature map.** The data map in
  `qbiocode.embeddings.embed.pqk` ended in `return float(coeff)`, but qiskit calls a
  `data_map_func` with *symbolic* parameters while it constructs the circuit --
  `PauliFeatureMap.pauli_block` passes a `ParameterVector`, not a data row -- so every
  call raised `TypeError: Parameter expression with unbound parameters {...} is not
  numeric` before a single circuit ran. `data_map=True` is the default, so the function
  was unusable for all three feature maps. `compute_pqk` carried a *fixed* copy of the
  same function, which is why the QProfiler `pqk` model worked while calling `pqk()`
  directly -- what `tutorial/PQK - OV.ipynb` does -- did not. The two copies are now one
  shared `qbiocode.utils.qutils.unit_coefficient_data_map`, which returns a float for
  numeric input and the unevaluated expression for symbolic input, so they cannot drift
  apart again. `tests/test_feature_map_data_map.py` covers the symbolic input directly,
  each feature map built through it, and a `pqk` round trip on the simulator.
- **A missing `seed` surfaced as a bare `KeyError`.** `get_backend_session` indexed
  `args["seed"]` and `args["shots"]` unchecked, so a config missing either failed several
  frames into a notebook with a message naming neither the function nor what the key is
  for. It now validates the keys the requested backend and primitive actually need and
  raises a `ValueError` naming the missing ones, their purpose, and the keys it did get.
- **A stale `dir_home` outranked the directory you were standing in.** `dir_home` is
  computed once, at import time, from `os.getcwd()`; `_resolve_input_folder` then tried
  `dir_home / folder_path` *before* walking the ancestors of the current directory. In a
  short-lived CLI run the two agree, so the defect was invisible there -- but in any
  long-lived process whose working directory moves after import (a notebook kernel that
  `os.chdir`s, a test session, a batch driver looping over checkouts) the frozen root won,
  and a run launched from `QBioCode-main/tutorial/QProfiler` silently read another
  checkout's CSVs and reported them as its own. The candidate order is now: the path as
  given, then each ancestor of the *current* directory, then a checkout root derived from
  the current directory, and only last the import-time `dir_home` -- kept in the list so
  configs that relied on it still resolve, but no longer able to shadow the caller's own
  tree. `tests/test_error_contracts.py::TestFolderPathResolution` covers the case
  directly, `monkeypatch.chdir`-ing into a fake `QBioCode-main` checkout.
- **`dir_home`'s value was published to GitHub Pages.** autodoc renders module-level
  data values, so `api/qbiocode.apps.qprofiler.html` carried the absolute filesystem path
  of whichever machine built the docs. It is now marked `:meta private:`.

#### Notebooks

- **`tutorial/QuVINE/example_quvine.ipynb` killed its own kernel.** The notebook
  benchmarks 12 methods, three of which are torch-backed (`appnp`, `gat_*`,
  `graphgps_*`). On macOS the process ends up with three separate LLVM OpenMP
  runtimes mapped in -- `torch/lib/libomp.dylib`, `qiskit_aer/backends/libomp.dylib`
  and the interpreter's own (a venv built on Anaconda) -- and whichever initialises
  second dies with `SIGSEGV` in `__kmp_fork_barrier` the first time it opens a
  parallel region. Each of the three torch methods reproduced it standalone at exit
  139; there is no Python traceback, so a Jupyter front end can only report that the
  kernel died. `KMP_DUPLICATE_LIB_OK=TRUE` does not help. The notebook now pins
  `OMP_NUM_THREADS=1` before importing `qbiocode`, matching what its sibling
  `quvine_sc_t_vs_mono.ipynb` already did -- that inconsistency was the whole bug.
  Re-executed end to end: 11/11 cells, 4 figures, 180 result rows, no NaN, so all
  three torch methods now genuinely run rather than being skipped.

  Note that `preload_openmp_libraries()` imports xgboost at `import qbiocode` to
  claim the runtime first, which is what a QPL/XGBoost pipeline needs and is exactly
  what makes torch lose the race here. The two orderings are mutually exclusive and
  the preload has no opt-out; `OMP_NUM_THREADS=1` is the only setting that satisfies
  both. Left as-is rather than reordered, because reordering would restore the
  XGBoost-fit crash it was added to prevent.
- **`PQK - OV.ipynb` reported accuracy as F1.** `run_model` returned
  `best_model.score(x_test, y_test)`, which for an `SVC` is accuracy — then named it
  `f1_score`, printed it as "Test F1 score", stored it in `F1_Quantum`/`F1_Classical`,
  wrote it to `PQK_OV_results.csv`, and labelled the plot's y-axis "F1 Score". It also
  contradicted the grid search in the same function, which selects on `f1_weighted`. The
  tell was that every reported value was an exact k/59 fraction of the test set. It now
  scores the held-out set with `f1_score(..., average='weighted')`, the metric the search
  actually optimises, so the reported number and the selection criterion agree.
- **`PQK - OV.ipynb` refit 1,260 identical models.** Its `param_grid` was a single flat
  dict sweeping `gamma` across all four kernels, but `SVC` ignores `gamma` when
  `kernel='linear'` — so 35 C x 37 gamma for the linear kernel was 1,295 fits of 35
  distinct models, 24.3% of the 5,180-combination grid spent on exact duplicates, and a
  linear winner reported a meaningless best `gamma`. `param_grid` is now a list of grids
  applying `gamma` only to the kernels that read it: the same search space in 3,920
  combinations. The cell now also prints the combination count, since this grid is the
  notebook's whole runtime cost.
- **A separator printed 60 lines of one `=`.** `print('\n=' * 60)` repeats
  newline-plus-equals sixty times rather than prefixing a newline to a 60-character rule;
  five occurrences across the download and preprocessing cells turned each section break
  into 60 lines of noise. Now `print('\n' + '=' * 60)`.
- **Four notebooks imported packages that do not exist.** `example_qprofiler.ipynb`
  (docs) used `from apps.qprofiler import qprofiler`, `qsage.ipynb` (docs) used
  `from apps.sage.sage import QuantumSage`, and both copies of `QPL_example.ipynb` used
  `import qprofiler.qprofiler` — there is no top-level `apps` or `qprofiler`
  distribution, so every one of them was an unconditional `ModuleNotFoundError` on a
  clean install. All four now import from `qbiocode.apps.*`. This is the same defect
  already fixed in `qbiocode/apps/qprofiler/cli.py`.
- **Notebook fixture paths only resolved for an editable install.** The single-cell
  notebooks derived a repository root with
  `os.path.dirname(os.path.dirname(os.path.abspath(qbiocode.__file__)))`, which yields
  `site-packages` for a normal install — so the derived fixture path pointed nowhere and
  the failure surfaced as anndata's "file not found", naming neither the `$QBC_DATA`
  override nor the directories that were tried. Each notebook also knew only one of the
  four fixture directories in the tree. They now call `qbiocode.tutorial_data_path()`.
  The QProfiler-2x2 notebook additionally derived its config template from the same
  broken root; it now reads `configs/config.yaml` out of the installed package.
- **Eight notebooks pinned a kernel that does not exist outside their author's machine**
  (`venv`, `.env`, `venv_quvine`, `qbc-pkg`). A notebook whose `kernelspec.name` cannot
  be resolved fails to execute under `nbclient`/`nbsphinx` rather than falling back.
  All notebooks now declare the standard `python3` kernel.
- **`sc_binary_qprofiler.ipynb` carried a stale warning about the PQK cache.** Its
  comment told readers the projection cache ignores `pqk_args` and that stale
  projections silently mask a feature-map change. That was true before this release and
  is no longer — the cache key now includes the feature-map fingerprint. The comment was
  corrected rather than deleted, since the purge it introduces is still worth keeping
  for a self-contained run.
- **Notebooks whose numbers came from the leaky scaling path now say so.** A note at the
  top of `sc_binary_qprofiler.ipynb` (both trees) states that the outputs below predate
  the train/test contamination fix, that they are therefore not comparable to a fresh
  run of the same config, and that the corrected numbers are usually slightly lower.
- **The 2x2 quadrant tutorial published as a truncated page.**
  `sc_binary_quvine_2x2_qprofiler.ipynb` (both trees) had 7 of its 8 code cells run: the
  final cell, which is the entire point of the notebook -- the 2x2 comparison of classical
  vs QuVINE embeddings against classical vs quantum models -- had never been executed, so
  the published page stopped just before its result. It now runs end to end (36 result
  rows: 4 embeddings x 3 models x 3 iterations, 3 figures) and its two
  `KNOWN_TRUNCATED` entries in `tests/integration/test_notebook_execution.py` are gone.
- **`example_qprofiler.ipynb` published as an empty page and could not be run to fill
  it.** The notebook had 0 of 7 cells executed, and its config pointed at
  `folder_path: 'tutorial/QProfiler/data/ld_data'` -- a *repository-relative* path, which
  `_resolve_input_folder` walks up the directory tree to find. From the `docs/` copy that
  resolved to the `tutorial/` tree's data, so the two copies of the notebook silently read
  the same input, and from an installed package it resolved to nothing. The two
  `configs/config.yaml` copies had also drifted apart (different model lists, embeddings,
  `n_jobs`, and credential comments), so the same notebook behaved differently depending
  on which tree you opened. Both copies are now byte-identical and demo-sized
  (`embeddings: ['none','pca']`, 8 models, `n_jobs: 1`, `iter: 2`), `folder_path` is the
  notebook-relative `'data/ld_data'` that the notebook itself generates, and the missing
  `xgb_args` / `gridsearch_xgb_args` blocks are documented rather than silently defaulted.
  The notebook now runs end to end from a clean checkout: 7/7 cells, 7 figures, 80s.
- **A deprecated matplotlib call mislabelled the tutorial's axes.** Every
  `ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')` in
  `example_qprofiler.ipynb` is now `ax.tick_params(axis='x', rotation=45)`. Setting tick
  *labels* on an axis whose tick *locations* are not fixed warns in matplotlib >= 3.5 and
  silently attaches the labels to the wrong ticks if the locator picks a different count.
- **Notebooks executed under a headless matplotlib backend published without their
  plots.** Six notebooks were re-executed earlier in this release with `MPLBACKEND=Agg`
  set. In an ipykernel process the backend that produces inline output is
  `module://matplotlib_inline.backend_inline`, which captures each `plt.show()` as an
  `image/png` display output; forcing `Agg` makes `plt.show()` a silent no-op, so the
  cells ran, reported success, and emitted *no figure at all*. Combined with
  `nbsphinx_execute = 'never'`, the published pages showed code and printed text with
  every plot missing -- and `test_no_notebook_is_half_executed` stayed green, because the
  cells were genuinely executed. `quvine_sc_cd4_vs_cd8.ipynb` alone lost 4 figures this
  way. All six were re-run under the default inline backend and now carry their figures:
  `QPL_example.ipynb` 4 (both trees), `qsage.ipynb` 7 (both trees),
  `quvine_sc_t_vs_mono.ipynb` 2, `quvine_sc_cd4_vs_cd8.ipynb` 4.
- **The Quantum Ensemble tutorial published as a truncated page, and its XGBoost arm
  never ran.** `QEnsemble_example_blobs.ipynb` (both trees) had 10 of its 15 code cells
  executed: the xgb, qcosine, qensemble and random-unitary arms and the post-processing
  that compares them were all unrun, so the page stopped immediately before the
  comparison the tutorial exists to make. Two defects kept it that way:
  - Each arm is guarded by `method not in predictions[dataset_name].keys()` against the
    cache in `experiments/predictions.pkl`. A method that ran but produced *nothing* is
    also a key, so the committed cache's empty `xgb_gs` frame (0 rows, recorded in an
    environment without XGBoost) was treated as done and never retried. The notebook
    printed "XGBoost grid search results are empty ... Skipping XGBoost" on an install
    where XGBoost is a base dependency and present. The guard is now
    `needs_run(predictions, dataset_name, method, rerun)`, which counts an empty cached
    result as absent.
  - `helper_functions.py` used `re.sub('\ ', '_', metric)`, an invalid escape sequence
    that warns today and is a `SyntaxError` in a future Python. Now `re.sub(' ', '_', metric)`,
    which is the same regex.

  The notebook now runs end to end in 41s (15/15 cells, 3 figures): `xgb_gs` fits its 486
  candidates and `xgb` its 90 rows, and the post-processing reports XGBoost against every
  quantum arm with significance tests. `experiments/predictions.pkl` and the three
  `Blob_max_median_*.pdf` figures are refreshed with the completed comparison.
- **`KNOWN_TRUNCATED` is now empty.** Every notebook in the tree is either a clean
  template or fully executed. The mechanism is kept so a future truncation has to be
  recorded deliberately rather than by weakening `test_no_notebook_is_half_executed`.
- **The Quantum Ensemble tutorial link was a 404.** `tutorials.md` linked
  `tutorials/QEnsemble/QEnsemble_example_blobs.html`, but the notebook existed only under
  `tutorial/QEnsemble/` and had no toctree entry, so Sphinx never built that page. The
  notebook (and the `helper_functions.py` it imports) are now mirrored under
  `docs/source/tutorials/QEnsemble/` and the page is in the toctree. Every gallery link
  in `tutorials.md` now resolves to a built page, and every built page is reachable from
  the toctree.

- **Absolute local paths were baked into committed notebook outputs.** Seventeen
  notebooks carried the running machine's own filesystem paths in their stored output --
  a `<env>/lib/python3.12/site-packages/node2vec/edges.py` pkg_resources warning, tqdm's
  `IProgress not found` banner, and `print()` echoes of the working directory and data
  directory. Because `nbsphinx_execute = 'never'`, those strings publish verbatim to
  GitHub Pages, where they are simultaneously useless to a reader (they name a directory
  that does not exist on their machine) and a needless disclosure of the author's home
  directory layout. Every one is now a `<env>/` or `<repo>/` placeholder. Only the output
  text changed; no figure, table, or metric was touched. Commented-out source paths in
  `archive/` are left alone -- they are provenance in code, not stray output, and that
  tree is not part of the published site.

#### Documentation build
- **The entire API reference was orphaned.** Nothing in any toctree pointed at
  `api/modules`, so Sphinx emitted "document isn't included in any toctree" for every
  generated page and none of them was reachable by navigation — the API docs existed
  only for anyone who guessed a URL. `api_overview.rst` now roots the tree with a hidden
  toctree entry.
- **The generated API pages were stale and came from two different generators.** Of the
  23 committed pages, 15 were `better_apidoc --separate` output and 8 were plain
  `sphinx-apidoc`, with no page at all for `qbiocode.apps` — meaning the whole of
  QuVINE, QProfiler and QSage was absent from the API reference, along with
  `qbiocode.evaluation.graph_evaluation`. The directory is now regenerated from the real
  package tree: 26 pages, 128 `automodule` targets covering every one of the 128
  importable modules under `qbiocode/`, with no dangling toctree references and no
  orphans. The `--separate` per-module pages are no
  longer committed; `run_apidoc` regenerates them at build time with `--force`.
- **`conf.py` resolved every path relative to the current working directory.** The
  `sys.path` entries, the `better_apidoc` template directory and its output directory
  were all built from `"."`, so they were correct only when `sphinx-build` was invoked
  from `docs/`. Running `sphinx-build docs/source docs/build/html` from the repository
  root — as most editors and IDE integrations do — silently produced a build with no
  autodoc templates and API pages written to the wrong place. All four are now derived
  from `__file__`.
- **`run_apidoc` swallowed every exception.** A bare `except Exception: pass` meant a
  missing `better_apidoc`, a template error or an import failure produced no output at
  all, and the build continued against whatever pages happened to be committed. A
  missing `better_apidoc` is now logged at info level (the committed pages are the
  intended fallback), and any other failure is logged as a Sphinx warning, so
  `sphinx-build -W` fails on it.
- **The documented version was wrong.** `conf.py` hardcoded `release = '0.0.1'` while
  `qbiocode/version.py` declared `0.1.0`, so the published documentation labelled itself
  a version that was never released. `conf.py` now parses `qbiocode/version.py` with a
  regex rather than importing the package, so the version is correct even when the docs
  build cannot import `qbiocode`.
- **`autodoc_mock_imports` mocked a package that does not exist and missed the ones that
  do.** It listed `tensorflow` — absent from the entire tree — while omitting every
  `[quvine]` dependency (`gensim`, `hiperwalk`, `node2vec`, `torch_geometric`,
  `community`, `ripser`, `omegaconf`), so a docs build without the extra installed
  failed to import the QuVINE modules it was documenting. `omegaconf` matters most:
  it backs QuVINE's config loading, so without it `api/config`, `api/core`, `api/sgns`,
  `main`, `pipeline` and `utils/io` were all unimportable. Base dependencies are deliberately *not*
  mocked: mocking them would hide a genuinely broken install behind a clean docs build.

- **Four broken documentation references, found by running the build rather than reading
  the sources.** Each one rendered as a visible defect on the published site:
  - `workshops/ISMB_2025.rst` embedded `../_static/qml_multiomics.png`, which has never
    been committed to either repository — a broken image on the workshop page. The
    directive is removed; nothing else on the page referred to the figure.
  - `apps/sage.rst` linked `:ref:`Data Complexity Measures <profiler:Data Complexity
    Measures>``. That `docname:Title` form needs `sphinx.ext.autosectionlabel`, which is
    not enabled, so the reference could never resolve. `apps/profiler.rst` now carries an
    explicit `.. _data-complexity-measures:` target and `sage.rst` points at it.
  - `apps/config.md` linked the QProfiler config as
    `../../../apps/qprofiler/configs/config.yaml` — a path that predates the move into
    `qbiocode/` and escapes the docs tree. It now links the file on GitHub.
  - `tutorials/QSage/qsage.ipynb` linked its own section with the GitHub anchor
    `#3b-load-pre-compiled-model-recommended`. nbsphinx slugs headings differently, so the
    link was dead in the rendered HTML; it now uses the nbsphinx form.

- **Thirteen docutils warnings and errors in docstrings and workshop pages.**
  `graph_evaluation.py` wrote graph-theory cardinalities as `|V|`, `|E|` and
  `|N(u) ∩ N(v)|`, which RST reads as substitution references — three ERRORs and no
  rendered formula. They are now inline literals. Ten section underlines in
  `workshops/ISMB_2025.rst` and `workshops/ISMB_2026.rst` were shorter than their titles.

- **Seven more defects that only a real build exposes.** Reading the sources had missed
  all of these; each was found by building the docs and then classifying every warning
  rather than pattern-matching the ones that looked important:
  - `background.md` ended with a `---` transition inside the level-3 *Video Resources*
    subsection, followed only by a closing paragraph. Every other `---` in that file is
    followed by a heading, so docutils reads it as a section separator; this one it
    rejected outright (*Transition must be child of `<document>` or `<section>`*). The
    closing paragraph is a page footer, so it is now a `{seealso}` admonition and needs
    no transition.
  - `installation.md` linked `docs/CONDA_SUBMISSION.md` as `../CONDA_SUBMISSION.md`. The
    file exists, but it sits above `docs/source/` and is not part of the doc set, so MyST
    resolved it as an unknown source document. It now links the file on GitHub.
  - `installation.md` fenced a Google Colab cell as `python`, but `!git clone` and `%cd`
    are IPython magics and the Python lexer failed on them. The `ipython3` lexer would
    handle it and ships with IPython, which is *not* a declared docs dependency, so the
    block is fenced as `text`.
  - `workshops/ISMB_2025.rst` indented two nested bullet lists past their parent item's
    text with no blank line, so docutils opened block quotes and then reported an
    unexpected unindent — four warnings for two lists, and neither rendered as a list.
    `workshops/ISMB_2026.rst` had one paragraph indented a single space too far, with the
    same effect.
  - Google-style `Attributes:` sections collided with `:undoc-members:` on every
    dataclass: the fields were described once from the docstring and once from the
    annotations, which Sphinx reports as *duplicate object description* (14 of them, all
    from `MethodMetadata` and `MethodResult`). `napoleon_use_ivar = True` renders such a
    section as `:ivar:` fields inside the class description, so each field is described
    exactly once.
  - Twelve `docs/source/api/*.rst` pages documented a package's re-exported names twice —
    once in each submodule's section, once again in the trailing *Module contents* block —
    which made the short names ambiguous (*more than one target found for
    cross-reference*). The package-level block now renders the package docstring only.
    A comment in each file records why, so a `better_apidoc` regeneration does not
    silently reintroduce it.

  These took the build from 90 warnings to 43.

  The build is now verified locally: `python -m sphinx -b html docs/source` succeeds and
  emits **no** structural warnings (broken toctree, undefined label, unknown document,
  unreadable image, failed autodoc import, duplicate object, ambiguous cross-reference,
  malformed page structure). Of the 43 that remain, 33 are docstring-formatting nits in
  `qbiocode/` — unindented block quotes, definition lists, one short title underline —
  and 10 are `myst.xref_missing` reports for `module-*` anchors that `.. automodule::`
  creates through the Python domain, which MyST's local-id check cannot see; the anchors
  were confirmed present in the rendered HTML, so those links work. Reformatting every
  docstring in the tree is deliberately out of scope for this release, which is why the
  build is not run under `-W`; `tests/integration/test_docs_build.py` asserts the
  *structural* warning classes are empty instead, which is the contract that matters and
  cannot regress silently.

  That test was also the reason six of the seven defects above survived the first pass: it
  matched warnings by *kind*, and none of them matched a listed kind. It now checks by
  location as well — any docutils or MyST warning reported against a file under
  `docs/source/` fails the test, since a hand-written page that emits a markup warning does
  not render as intended. Docstring warnings keep their exemption (they carry a `qbiocode/`
  location), as does the `myst.xref_missing` class described above, with the verification
  recorded in the code. Replaying the pre-fix build log through the new check produces nine
  offenders and the post-fix log produces none, so the guard is not vacuous.

#### Earlier fixes
- Invalid escape sequence in `qbiocode/visualization/visualize_correlation.py`
- Import ordering issues throughout codebase
- Type-check errors in CI pipeline
- Documentation build failures
- Path handling in cross-platform tests

## [0.1.0] - 2026-04-06

### ⚠️ Breaking Changes
- **Minimum Python version increased to 3.10** (was 3.9)
  - Required for compatibility with latest Qiskit ecosystem (qiskit-ibm-runtime 0.44.0+)
  - Python 3.9 reaches end-of-life in October 2025

### Added

#### Core Features
- **QProfiler**: Automated machine learning benchmarking with data complexity analysis
  - Support for multiple ML models (RF, SVM, LR, DT, NB, MLP, XGBoost)
  - Support for quantum ML models (QSVC, PQK, VQC, QNN)
  - Comprehensive data profiling and complexity metrics
  - Batch processing mode for large-scale experiments
  - CLI interface for easy usage

- **QSage**: Meta-learning framework for intelligent model selection
  - Model recommendation based on dataset characteristics
  - Pre-trained models for quick predictions
  - Integration with QProfiler results

- **Data Generation**: Artificial dataset generation with controlled complexity
  - Multiple dataset types (circles, moons, spirals, S-curve, Swiss roll, spheres)
  - Configurable noise levels and complexity parameters
  - Support for multi-class classification problems

- **Embeddings**: Dimensionality reduction and feature extraction
  - Autoencoder implementations
  - Integration with classical embedding methods

- **Evaluation**: Comprehensive model and dataset evaluation
  - Multiple metrics (accuracy, F1-score, AUC, training time)
  - Cross-validation support
  - Statistical analysis tools

- **Visualization**: Publication-quality plotting functions
  - Correlation analysis plots with scientific styling
  - Heatmaps with customizable colormaps
  - High-resolution output (600 DPI) for publications

#### Documentation
- Complete API documentation with Sphinx
- Tutorials for QProfiler, QSage, and data generation
- Installation guides for multiple platforms
- Galaxy integration documentation
- Example notebooks and workflows

#### Project Infrastructure
- GitHub issue templates (bug report, feature request, documentation, question)
- Pull request template with comprehensive checklists
- Security policy (SECURITY.md)
- Support documentation (SUPPORT.md)
- Contributing guidelines (CONTRIBUTING.md)
- Code of conduct (CODE_OF_CONDUCT.md)
- Citation file (CITATION.cff)
- Zenodo metadata for DOI generation (.zenodo.json)

#### CI/CD
- GitHub Actions workflow for continuous integration
  - Multi-OS testing (Ubuntu, macOS, Windows)
  - Multi-Python version testing (3.10, 3.11)
  - Code quality checks (flake8, black, isort, mypy)
  - Test coverage reporting
  - Documentation building
- GitHub Actions workflow for automated releases
  - PyPI publishing
  - Zenodo archiving
  - Release asset management

#### Command-Line Tools
- `qprofiler`: Run QProfiler experiments
- `qprofiler-batch`: Batch processing mode
- `qsage`: Model recommendation tool

### Changed
- Updated visualization functions for publication quality
  - Enhanced scatter plots with better colormaps
  - Improved heatmaps with professional styling
  - Better legend positioning and labeling
  - Increased DPI for high-quality output

### Fixed
- NaN handling in correlation visualization
- Windows path compatibility issues
- Automated QSage model download

### Dependencies
- Python >= 3.10, < 3.13
- Qiskit for quantum computing functionality
- scikit-learn for classical ML algorithms
- pandas, numpy for data manipulation
- matplotlib, seaborn for visualization
- XGBoost for gradient boosting
- hydra-core for configuration management


---

## Release Notes

### Version 0.1.0 - Initial Public Release

This is the first public release of QBioCode, a comprehensive framework for quantum machine learning applications in healthcare and life sciences. The release includes:

- Complete implementation of QProfiler and QSage applications
- Support for both quantum and classical machine learning models
- Extensive documentation and tutorials
- Professional project infrastructure for open-source collaboration
- Automated CI/CD pipelines
- Ready for PyPI distribution and Zenodo archiving

**Note**: This is an alpha release. APIs may change in future versions. Please report any issues on our [GitHub issue tracker](https://github.com/qiskit-community/QBioCode/issues).

---

[0.2.0]: https://github.com/qiskit-community/QBioCode/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/qiskit-community/QBioCode/releases/tag/v0.1.0