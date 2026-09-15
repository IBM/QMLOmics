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

"""The pyMFE seam beyond the merged row: seeding, plotting, NaN, resume, and p=1.

``tests/integration/test_pymfe_model_integration.py`` covers the assembled row, the CSV
round-trip, QuantumSage, the correlation *numbers*, the winner finder and the combine
step. What is left is everything that only shows up once the block is 115 columns wide
rather than 13, and one boundary that was never a boundary before.

Four of these would fail silently rather than loudly, which is why they are worth
asserting rather than assuming:

* The complexity block is *seed-independent* at the app tier, because both
  ``evaluate()`` call sites in ``qprofiler.py`` take the default ``random_state=0``.
  That is what lets two runs at different ``seed`` values be compared to each other --
  and it would be undone by the natural-looking "improvement" of threading
  ``args['seed']`` through, which nothing currently forbids.
* ``plot_results_correlation`` rewrites feature names for display through a nested chain
  of six ``re.sub`` calls written for the legacy column names -- ``std_var``,
  ``std_co_of_v``, ``Coefficient of Variation %``. It now receives ~125 pyMFE names it
  was never designed for. A chain that is not injective silently *merges two rows of the
  heatmap*, which reads as a plot rather than as an error.
* ``get_low_var_features`` returns ``None`` -- not a number -- when no feature clears the
  variance threshold, which happens for any matrix whose feature variances are all equal
  (the threshold is their 25th percentile, so every feature is at or below it). That
  ``None`` becomes ``NaN`` in the row and has to survive the whole chain.
* ``p == 1`` now fails, and it fails inside ``get_intrinsic_dim`` before pyMFE is
  reached, with an sklearn message about array shapes that names nothing the caller
  wrote.
"""

from __future__ import annotations

import ast
import pathlib
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # before pyplot is touched anywhere below

import qbiocode  # noqa: E402,F401  -- orders the OpenMP runtimes first
from qbiocode.evaluation.dataset_evaluation import (  # noqa: E402
    complexity_feature_columns,
    detect_complexity_schema,
    evaluate,
    get_low_var_features,
)
from qbiocode.evaluation.mfe_features import MFE_COLUMN_PREFIX  # noqa: E402
from qbiocode.evaluation.model_run import model_run  # noqa: E402
from qbiocode.utils.combine_evals_results import combine_results  # noqa: E402
from qbiocode.utils.dataset_checkpoint import checkpoint_restart  # noqa: E402
from qbiocode.visualization.visualize_correlation import (  # noqa: E402
    compute_results_correlation,
    plot_results_correlation,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
QPROFILER_SOURCE = REPO_ROOT / "qbiocode" / "apps" / "qprofiler" / "qprofiler.py"


def _dataset(n=60, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame(X), y


def _row(seed_offset=0):
    X, y = _dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate(X, y, "seam.csv")


@pytest.fixture(scope="module")
def scored_table():
    """A QProfiler-shaped table: the complexity block plus real metrics, several rows."""
    X, y = _dataset()
    from sklearn.model_selection import train_test_split

    rows = []
    for iteration in (1, 2, 3):
        split = train_test_split(
            X.to_numpy(), y, stratify=y, test_size=0.3, random_state=10 + iteration
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The complexity block is measured on the training half, as qprofiler does
            # for the embedded case.
            block = evaluate(pd.DataFrame(split[0]), split[2], "seam.csv")
            out = model_run(*split, f"seam-{iteration}", {
                "model": ["dt", "lr", "nb"], "seed": 7, "n_jobs": 1, "grid_search": False,
            })
        for label in (k[len("results_"):] for k in out if k.startswith("results_")):
            metrics = out[f"results_{label}"][0]
            rows.append({
                **block.iloc[0].to_dict(),
                "model": label,
                "embeddings": "none",
                "iteration": iteration,
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
                "auc": metrics["auc"],
                "time": metrics["time"],
                "Model_Parameters": str(metrics["Model_Parameters"]),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def correlations(scored_table):
    """Spearman correlations over the full 125-column block.

    Module-scoped rather than class-scoped-on-a-method: pytest 8 deprecates the latter,
    because the fixture runs once while each test gets a fresh instance.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, frame = compute_results_correlation(scored_table, thresh=0.5)
    return frame


class TestTheBlockDoesNotDependOnTheRunSeed:
    """Comparability across runs rests on ``evaluate()`` keeping its default seed."""

    def test_the_block_is_identical_whatever_the_ambient_numpy_state(self):
        """Landmarking cross-validates and the clustering measures run k-means.

        Both would happily consume the global RNG. ``evaluate()`` passes an explicit
        ``random_state`` to pyMFE so they do not, which is what makes the block a
        property of the dataset rather than of whatever ran before it in the process.
        """
        np.random.seed(1)
        first = _row()
        np.random.seed(99999)
        second = _row()
        pd.testing.assert_frame_equal(first, second)

    def test_neither_qprofiler_call_site_forwards_the_run_seed(self):
        """A design pin, not a style check -- and the reason the block is comparable.

        ``qprofiler.py`` calls ``evaluate(df, y, file)`` twice, once on the raw matrix and
        once per embedded matrix, and neither passes ``random_state``. So every run uses
        pyMFE seed 0 and two runs at ``seed: 7`` and ``seed: 99`` produce the same
        complexity columns for the same matrix, which is what makes rows from different
        runs comparable in one QSage training table.

        Threading ``args['seed']`` through here looks like an improvement and is not: it
        would make the complexity of a dataset depend on the run that measured it, so
        concatenating two runs would silently mix two different measurements of the same
        thing. If that change is ever wanted, this test is where to argue for it.
        """
        tree = ast.parse(QPROFILER_SOURCE.read_text(encoding="utf-8"))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "evaluate"
        ]
        assert len(calls) == 2, (
            f"expected exactly 2 evaluate() call sites in qprofiler.py, found {len(calls)}"
        )
        for call in calls:
            seeded = [kw.arg for kw in call.keywords if kw.arg in ("random_state", "mfe_features")]
            assert not seeded, (
                f"evaluate() at qprofiler.py:{call.lineno} now passes {seeded}. If the run "
                "seed is being forwarded, the complexity block stops being comparable "
                "between runs -- read this test's docstring before changing it."
            )
            assert len(call.args) == 3, (
                f"evaluate() at qprofiler.py:{call.lineno} takes {len(call.args)} positional "
                "arguments; this test assumed (df, y, file)"
            )


class TestTheDisplayLabelsStayDistinct:
    """The heatmap's label rewriting must not merge two features into one row."""

    def test_the_rewrite_chain_is_injective_over_the_real_feature_set(
        self, correlations, tmp_path
    ):
        """Six nested ``re.sub`` calls, written for 13 legacy names, now see ~125.

        The chain lowercases nothing and anchors only two of its patterns, so it is easy
        for two distinct inputs to land on the same output -- and the consequence is not
        an error. ``plot_results_correlation`` pivots on the rewritten name, so a
        collision silently combines two features into a single heatmap row, showing one
        of their correlation values under a label that names both.

        Asserted on the *real* figure rather than on a reimplementation of the chain. The
        chain is written inline as a list comprehension with no seam to call, so the
        obvious test -- apply the same six substitutions in the test and check the outputs
        are distinct -- would guard a copy and pass forever while the original drifted.
        The pivoted matrix behind the clustermap has one row per surviving label, so
        comparing its height to the number of distinct input features tests the chain
        that actually runs. The local copy is used only to name the culprits when it
        fails, which is worth the duplication for the error message alone.
        """
        import matplotlib.pyplot as plt

        expected = correlations["feature"].nunique()
        assert expected > 100, f"expected the full block, got {expected} features"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            figures = plot_results_correlation(
                correlations, metric="f1_score", show_plots=False
            )
        try:
            pivoted = figures.clustered_heatmap.data2d
        finally:
            plt.close("all")

        if pivoted.shape[0] != expected:
            # Reproduce the chain only to say WHICH features merged.
            import re

            def display(x):
                return re.sub(
                    "std", "Std. dev. of",
                    re.sub("co of v", "coefficient of variation",
                           re.sub("kurt$", "kurtosis",
                                  re.sub("skew$", "skewness",
                                         re.sub("var$", "variation",
                                                re.sub("%", "", re.sub("_", " ", x)))))),
                )

            groups = {}
            for feature in sorted(correlations["feature"].unique()):
                groups.setdefault(display(feature), []).append(feature)
            merged = {k: v for k, v in groups.items() if len(v) > 1}
            explanation = (
                f"the collisions this test can reproduce are {merged}"
                if merged
                else (
                    "the local reproduction of the chain finds no collision, which means "
                    "the substitutions in visualize_correlation.py have changed since "
                    "this test was written -- update the copy in this test so it can name "
                    "the culprits, then fix the chain"
                )
            )
            raise AssertionError(
                f"the heatmap has {pivoted.shape[0]} rows for {expected} distinct "
                f"features, so the display-label rewriting merged some of them; "
                f"{explanation}"
            )

    def test_the_heatmaps_render_at_the_full_block_width(self, correlations, tmp_path):
        """~125 features is ten times what the default figsize was chosen for.

        A clustermap over that many rows is where a linkage on constant or NaN input
        raises, so this is a real path rather than a smoke test. Asserted on the returned
        figures, not on pixels.
        """
        import matplotlib.pyplot as plt

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            figures = plot_results_correlation(
                correlations,
                metric="f1_score",
                save_file_path=str(tmp_path / "corr.pdf"),
                show_plots=False,
            )
        assert figures.scatter is not None
        assert figures.clustered_heatmap is not None
        assert figures.ordered_heatmap is not None
        assert (tmp_path / "corr.pdf").exists()
        plt.close("all")


class TestTheNativeBranchThatReturnsNone:
    """``# Low variance features`` is ``None`` for equal-variance data, hence NaN."""

    @staticmethod
    def _equal_variance_frame():
        """Equal variances across features, but distinct columns.

        ``get_low_var_features`` thresholds at the 25th percentile of the variances and
        ``VarianceThreshold`` drops features at or below it, so when every variance is
        (near) identical it drops all of them, raises, and the helper returns ``None``.

        The columns are shuffled copies of one vector rather than identical copies. That
        matters: four *identical* columns also collapse pyMFE's ``linear_discr``
        landmark, which fails inside sklearn's LDA with
        ``IndexError: index 0 is out of bounds for axis 0 with size 0`` -- so the fixture
        would exercise a second, unrelated degeneracy and never reach the branch this
        class is about.
        """
        rng = np.random.default_rng(3)
        base = np.linspace(-2.0, 2.0, 60)
        columns = {}
        for i in range(6):
            shuffled = base.copy()
            rng.shuffle(shuffled)
            columns[f"f{i}"] = shuffled
        return pd.DataFrame(columns)

    @classmethod
    def _labels(cls, frame):
        return (frame["f0"].to_numpy() > 0).astype(int)

    def test_the_helper_really_returns_none_on_this_input(self):
        """The premise. If this stops being None the tests below prove nothing."""
        frame = self._equal_variance_frame()
        # Shuffling leaves the variances equal only to floating-point noise, which is
        # enough: the threshold is their 25th percentile, so all of them sit at or below
        # it. Asserting the spread rather than exact equality keeps the premise honest.
        assert frame.var().std() < 1e-12, "fixture variances are no longer near-identical"
        assert get_low_var_features(frame, frame.shape[1]) is None

    def test_evaluate_turns_it_into_nan_rather_than_failing(self):
        """``None`` in a numeric column would make the whole row object-dtype.

        ``evaluate`` coerces with ``errors='coerce'`` precisely so this becomes NaN and the
        column stays numeric -- otherwise every downstream ``np.log`` and every regressor
        would reject the frame over one unmeasurable feature count.
        """
        frame = self._equal_variance_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = evaluate(frame, self._labels(frame), "equal-variance.csv")
        value = row["# Low variance features"].iloc[0]
        assert pd.isna(value), f"expected NaN, got {value!r}"
        assert row["# Low variance features"].dtype.kind == "f", "column went object-dtype"

        # One unmeasurable column must not poison the other 125. The three tree-structure
        # standard deviations are the honest exception: this data induces a single-level
        # decision tree, so the spread of nodes across levels genuinely does not exist.
        # They are named rather than tolerated by a threshold, so a *new* NaN fails here.
        allowed = {
            "# Low variance features",
            f"{MFE_COLUMN_PREFIX}nodes_per_level.sd",
            f"{MFE_COLUMN_PREFIX}nodes_repeated.sd",
            f"{MFE_COLUMN_PREFIX}tree_imbalance.sd",
        }
        numeric = row.drop(columns=["Dataset"])
        missing = {c for c in numeric.columns if not np.isfinite(numeric[c].iloc[0])}
        assert missing <= allowed, (
            f"a NaN spread beyond the columns that earned it: {sorted(missing - allowed)}"
        )

    def test_the_schema_is_still_detected_with_a_nan_in_the_native_half(self):
        """detect_complexity_schema's mixed-table check looks for all-NaN pyMFE rows.

        A NaN in the *native* half must not be mistaken for that, or a legitimate
        equal-variance dataset would be refused as a concatenation.
        """
        frame = self._equal_variance_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = evaluate(frame, self._labels(frame), "equal-variance.csv")
        schema, columns = detect_complexity_schema(row)
        assert schema == "pymfe"
        assert "# Low variance features" in columns


class TestResumeAndCompile:
    """``checkpoint_restart`` and ``combine_results`` on 126-column tables."""

    @staticmethod
    def _lay_out(root, datasets):
        """A results directory shaped like a finished QProfiler run.

        Both helpers under test walk *subdirectories*: ``checkpoint_restart`` treats each
        one as a dataset and strips ``prefix_length`` characters off its name to recover
        that dataset's name (default 8, for QProfiler's ``dataset_`` prefix), and
        ``combine_results`` scans each subdirectory for files whose names start with
        ``eval_file_prefix`` / ``results_file_prefix``. A flat directory therefore yields
        either the directory's own name as a phantom dataset or "No objects to
        concatenate" -- so the layout is part of the contract, not incidental.
        """
        root.mkdir(parents=True, exist_ok=True)
        frames = []
        for name in datasets:
            X, y = _dataset(seed=abs(hash(name)) % 1000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                block = evaluate(X, y, name)
            frames.append(block)
            directory = root / f"dataset_{name}"
            directory.mkdir(parents=True, exist_ok=True)
            block.to_csv(directory / "RawDataEvaluation.csv", index=False)
            pd.DataFrame([{
                "Dataset": name, "model": "dt", "embeddings": "none", "iteration": 1,
                "accuracy": 0.8, "f1_score": 0.8, "auc": 0.8, "time": 0.1,
            }]).to_csv(directory / "ModelResults.csv", index=False)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _lay_out_flat(root, datasets):
        """A *recent* run: the two CSVs directly in the directory, as QProfiler writes them."""
        root.mkdir(parents=True, exist_ok=True)
        frames = []
        for name in datasets:
            X, y = _dataset(seed=abs(hash(name)) % 1000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frames.append(evaluate(X, y, name))
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(root / "RawDataEvaluation.csv", index=False)
        pd.DataFrame([{
            "Dataset": datasets[0], "model": "dt", "embeddings": "none", "iteration": 1,
            "accuracy": 0.8, "f1_score": 0.8, "auc": 0.8, "time": 0.1,
        }]).to_csv(root / "ModelResults.csv", index=False)
        return combined

    def test_checkpoint_restart_lists_the_datasets_already_finished(self, tmp_path):
        """The marker file is RawDataEvaluation.csv, which is now 126 columns wide.

        checkpoint_restart reads the *names* out of it, so width should be irrelevant --
        which is exactly the kind of assumption worth pinning after a schema change.
        """
        root = tmp_path / "previous"
        self._lay_out(root, ["class_data-1.csv", "class_data-2.csv"])
        done = checkpoint_restart(str(root), verbose=False)
        assert isinstance(done, list)
        assert set(done) == {"class_data-1.csv", "class_data-2.csv"}, done

    def test_combining_two_pymfe_runs_keeps_the_schema(self, tmp_path):
        """A resumed run is compiled by concatenating two directories.

        Both halves are the current schema here, so the result must still detect as
        'pymfe' with the whole block intact. The mixed-schema case -- one half from before
        the pyMFE change -- is refused by detect_complexity_schema and is covered in
        tests/test_dataset_evaluation.py::TestTheMixedSchemaHazard.
        """
        previous = tmp_path / "prev"
        recent = tmp_path / "recent"
        first = self._lay_out(previous, ["class_data-1.csv"])
        # The two directories are NOT laid out the same way, which is easy to get wrong:
        # combine_results scans SUBDIRECTORIES of the previous run (one per dataset) but
        # reads the recent run's two CSVs directly out of the directory it is given. That
        # asymmetry is also where the known Dataset-column bug lives -- the recent pair is
        # read with index_col=0 while QProfiler writes them with index=False, which is
        # pinned in tests/integration/test_pymfe_model_integration.py.
        self._lay_out_flat(recent, ["class_data-2.csv"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            combined_eval, _ = combine_results(
                str(previous), str(recent), output_eval_file=str(tmp_path / "eval.csv"),
                output_results_file=str(tmp_path / "results.csv"),
                save_intermediate=False, verbose=False,
            )

        mfe_columns = [c for c in combined_eval.columns if str(c).startswith(MFE_COLUMN_PREFIX)]
        assert len(mfe_columns) == len(
            [c for c in first.columns if str(c).startswith(MFE_COLUMN_PREFIX)]
        ), "the combined table lost or gained pyMFE columns"
        assert complexity_feature_columns(combined_eval.columns), "block no longer recognized"


def test_a_single_feature_fails_before_pymfe_is_reached():
    """``p == 1`` is a boundary now, and the error names nothing the caller wrote.

    ``get_intrinsic_dim`` runs first and skdim's lPCA hands the matrix to sklearn, which
    refuses a single column. So the message is about array shapes, with no mention of
    ``Intrinsic_Dimension`` or of the fact that a one-feature dataset cannot be profiled.

    Pinned rather than fixed: whether one feature should be supported at all is a product
    question -- most of the block (correlation, PCA dimensionality, the neighbourhood
    measures) is either degenerate or undefined at p=1. What this test guarantees is that
    the failure stays a loud, immediate ValueError rather than becoming a row of quiet
    NaN that looks like a measurement.
    """
    frame = pd.DataFrame({"only": np.linspace(0.0, 1.0, 40)})
    y = np.array([0, 1] * 20)
    with pytest.raises(ValueError, match="1 feature"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evaluate(frame, y, "one-feature.csv")
