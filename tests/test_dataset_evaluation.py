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

"""``evaluate`` produces a usable complexity row, and the pyMFE subset stays curated.

``dataset_evaluation.py`` had no tests at all before the pyMFE integration, which is
how two things survived in it: ``std_entropy`` was the standard deviation of a
scalar and so was always exactly 0, and ``get_complexity`` accepted
``n_neighbors``/``n_components`` and then ignored both in favour of hardcoded values.

The pyMFE half needs guarding for a different reason. pyMFE reports a measure it
could not compute as ``NaN`` and warns; QBioCode must pass ``suppress_warnings=True``
(a wide matrix otherwise emits thousands of lines per dataset), which discards the
warning. So a broken or degenerate measure does not fail -- it silently occupies a
column of ``NaN`` or of a constant that QSage then trains on. ``MFE_FEATURES`` is
curated against measured behaviour precisely to avoid that, and
``test_excluded_measures_stay_excluded`` keeps it curated: each excluded name is
listed with the reason it was excluded, so re-adding one fails here rather than
quietly shipping a dead column.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qbiocode.evaluation.dataset_evaluation import (
    LEGACY_COMPLEXITY_COLUMNS,
    NATIVE_COMPLEXITY_COLUMNS,
    SAMPLE_COUNT_COLUMN,
    complexity_feature_columns,
    detect_complexity_schema,
    evaluate,
    get_complexity,
    mfe_columns,
)
from qbiocode.evaluation.mfe_features import (
    BINARY_SCALAR_SD_SUPPRESSED,
    MFE_COLUMN_PREFIX,
    MFE_FEATURES,
    get_mfe_features,
    mfe_column_names,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tutorial" / "QProfiler" / "data" / "ld_data" / "class_data-1.csv"


def _fixture():
    """The committed low-dimensional benchmark dataset, as ``evaluate`` wants it."""
    raw = pd.read_csv(FIXTURE)
    X = pd.DataFrame(raw.iloc[:, :-1].to_numpy(dtype=float))
    y = raw.iloc[:, -1].to_numpy().astype(int)
    return X, y


def _wide():
    """A ``p >> n`` frame -- the regime QBioCode's omics data actually lives in."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 500))
    y = rng.integers(0, 2, size=40)
    X[y == 1, :10] += 1.5
    return pd.DataFrame(X), y


@pytest.fixture(scope="module")
def evaluated():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate(*_fixture(), "class_data-1.csv")


class TestSchema:
    """What ``evaluate`` promises its callers about the shape of its output."""

    def test_it_returns_exactly_one_row(self, evaluated):
        assert len(evaluated) == 1

    def test_the_dataset_name_is_carried_through(self, evaluated):
        assert evaluated["Dataset"].iloc[0] == "class_data-1.csv"

    def test_every_native_column_is_present(self, evaluated):
        missing = [c for c in NATIVE_COMPLEXITY_COLUMNS if c not in evaluated.columns]
        assert not missing, f"natively-computed columns absent from evaluate(): {missing}"

    def test_the_pymfe_block_is_present_and_prefixed(self, evaluated):
        block = mfe_columns(evaluated)
        assert block, "no pyMFE columns in evaluate() output"
        assert all(c.startswith(MFE_COLUMN_PREFIX) for c in block)
        # Every emitted name must be one the curated list can account for.
        expected = set(mfe_column_names(binary=True))
        assert set(block) <= expected, f"unexpected pyMFE columns: {sorted(set(block) - expected)}"

    def test_the_measure_columns_are_numeric(self, evaluated):
        """Object dtype here means ``numpy`` ufuncs fail on the result.

        Building the row from a dict that also holds the ``Dataset`` string makes the
        whole frame object-dtype. That was invisible while the only consumer wrote to
        CSV and read it back -- but QSage's ``calculate_SLGH`` calls ``np.log`` on
        these columns, which raises "loop of ufunc does not support argument 0 of type
        numpy.float64" on an object column.
        """
        measures = evaluated.drop(columns=["Dataset"])
        assert all(dtype.kind in "if" for dtype in measures.dtypes)

    def test_landmarking_is_included(self, evaluated):
        """The gap the integration existed to close.

        QBioCode's hand-rolled block had no landmarking features at all, and
        cheap-classifier performance is the strongest known predictor for model
        selection -- QSage's exact task.
        """
        for measure in ("one_nn", "naive_bayes", "linear_discr", "best_node", "elite_nn"):
            assert f"{MFE_COLUMN_PREFIX}{measure}.mean" in evaluated.columns


class TestNoDeadColumns:
    """No column may be structurally empty or non-finite."""

    def test_nothing_is_nan_or_inf(self, evaluated):
        measures = evaluated.drop(columns=["Dataset"])
        bad = [c for c in measures.columns if not np.isfinite(measures[c].iloc[0])]
        assert not bad, f"non-finite complexity columns: {bad}"

    def test_no_structurally_empty_sd_column_is_emitted(self, evaluated):
        """With two classes a one-vs-one measure has a single value, so its ``.sd`` is NaN.

        pyMFE still emits the column. Shipping it would guarantee a NaN column, so
        ``get_mfe_features`` drops it -- but only when the labels really are binary.
        """
        for measure in BINARY_SCALAR_SD_SUPPRESSED:
            assert f"{MFE_COLUMN_PREFIX}{measure}.sd" not in evaluated.columns

    def test_the_sd_columns_return_when_labels_are_multiclass(self):
        """The suppression is conditional on binary labels, not unconditional.

        With three classes there are three one-vs-one pairs, so the standard
        deviation across them is a real quantity and must not be dropped.
        """
        rng = np.random.default_rng(1)
        X = rng.normal(size=(90, 6))
        y = np.repeat([0, 1, 2], 30)
        X += y[:, None] * 0.9
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            block = get_mfe_features(X, y, features=("l2", "f3"))
        assert f"{MFE_COLUMN_PREFIX}l2.sd" in block
        assert f"{MFE_COLUMN_PREFIX}f3.sd" in block


class TestCuration:
    """The excluded pyMFE measures stay excluded, with the reason recorded."""

    #: name -> why it is not in MFE_FEATURES. Every entry is a measurement on
    #: QBioCode-shaped data (all-numeric, binary, frequently p >> n), not a guess.
    EXCLUDED = {
        # Raises ValueError on every dataset, iris included, under NumPy >= 2:
        # pymfe 0.4.4 assigns a (1, 1) array into a scalar slot. QBioCode cannot
        # avoid NumPy 2 (qiskit-machine-learning 0.9.0 requires it).
        "f1v": "crashes under numpy>=2",
        # O(p^2) with no subsampling: 94 s at p=2000, so hours at p=20000.
        "one_itemset": "O(p^2), intractable on omics data",
        "two_itemset": "O(p^2), intractable on omics data",
        # 957 s of a 1007 s run at p=20000, and .mean is bit-identical to var.mean.
        "eigenvalues": "O(p^3) and duplicates var.mean",
        # max_attr_num defaults to 12, so these describe 12 random columns.
        "attr_conc": "silently samples only 12 attributes",
        "class_conc": "silently samples only 12 attributes",
        # Undefined for negative values, i.e. for anything that has been scaled.
        "g_mean": "NaN on scaled data",
        "h_mean": "NaN on scaled data",
        # NaN in every shape probed.
        "sd_ratio": "always NaN",
        "num_to_cat": "always NaN (no categorical attributes)",
        # +inf once p >= n, which propagates silently into QSage's regressors.
        "lh_trace": "inf at p>=n",
        "roy_root": "inf at p>=n",
        # Constant across every shape and separation probed.
        "t1": "saturated at 1.0",
        "sc": "constant 0",
        "nre": "constant ln(2) on balanced binary",
        "sparsity": "constant 0 on continuous data",
        "var_importance": "exactly 1/p, restates nr_attr",
        "attr_ent": "constant under uniform discretization",
        "random_node": "constant across the separation sweep",
        "nr_disc": "always 1 for two classes",
        # Structurally inapplicable to QBioCode's data.
        "nr_cat": "always 0 (all-numeric data)",
        "nr_bin": "always 0 (all-numeric data)",
        "cat_to_num": "always 0 (all-numeric data)",
        "nr_class": "always 2 (qprofiler is binary-only)",
        "freq_class": "0.5 by construction on balanced binary",
        "leaves_per_class": "0.5 by construction on balanced binary",
        # Exact duplicates of measures that are kept.
        "t2": "identical to attr_to_inst (both p/n)",
        "c1": "identical to class_ent for binary labels",
    }

    @pytest.mark.parametrize("measure", sorted(EXCLUDED))
    def test_excluded_measures_stay_excluded(self, measure):
        assert measure not in MFE_FEATURES, (
            f"{measure!r} is back in MFE_FEATURES but was excluded because it "
            f"{self.EXCLUDED[measure]}. See qbiocode/evaluation/mfe_features.py."
        )

    def test_the_curated_list_has_no_duplicates(self):
        assert len(set(MFE_FEATURES)) == len(MFE_FEATURES)

    def test_every_curated_name_is_a_real_pymfe_measure(self):
        """A typo in the list would otherwise be dropped in silence.

        pyMFE ignores an unknown name in ``features=``, so a misspelling costs a
        measure without any error.
        """
        from pymfe.mfe import MFE

        valid = set(MFE.valid_metafeatures())
        unknown = sorted(set(MFE_FEATURES) - valid)
        assert not unknown, f"not pyMFE measure names: {unknown}"

    def test_the_kept_degenerate_measures_are_bounded(self):
        """Measures kept despite going flat at p >= n must not go to inf.

        ``can_cor``, ``w_lambda``, ``p_trace``, ``f2``, ``f4`` and the L-family
        collapse to a constant when p >= n (linear separability is free in high
        dimension) but stay informative at p < n, which is the embedded
        ``evaluate()`` call. They are only safe to keep because they are bounded --
        ``lh_trace`` and ``roy_root`` are the same family and are excluded for
        going to inf.
        """
        X, y = _wide()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            block = get_mfe_features(
                X, y, features=("can_cor", "w_lambda", "p_trace", "f2", "f4", "l1", "l2", "l3")
            )
        assert block, "no columns returned"
        infinite = {k: v for k, v in block.items() if np.isinf(v)}
        assert not infinite, f"kept measures went infinite at p>n: {infinite}"


class TestReproducibility:
    """The pyMFE block cross-validates and clusters, so the seed has to be honoured."""

    def test_two_runs_with_the_same_seed_agree(self):
        X, y = _fixture()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = evaluate(X, y, "class_data-1.csv", random_state=7)
            second = evaluate(X, y, "class_data-1.csv", random_state=7)
        pd.testing.assert_frame_equal(first, second)

    def test_the_seed_reaches_pymfe(self):
        """Guards against ``random_state`` being accepted and then dropped.

        Landmarking cross-validates and the clustering measures run k-means, so two
        different seeds must be able to disagree somewhere in the block. (They are
        not *required* to differ on any particular measure, hence the whole-block
        comparison.)
        """
        X, y = _fixture()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = get_mfe_features(X, y, random_state=0)
            b = get_mfe_features(X, y, random_state=99)
        assert a.keys() == b.keys()
        assert any(a[k] != b[k] for k in a), "random_state had no effect anywhere"


class TestWideData:
    """``p >> n`` is the normal case for omics matrices, not an edge case."""

    def test_evaluate_survives_more_features_than_samples(self):
        X, y = _wide()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = evaluate(X, y, "wide.csv")
        assert len(row) == 1
        measures = row.drop(columns=["Dataset"])
        assert not np.isinf(measures.to_numpy(dtype=float)).any(), "inf in a p>n row"

    def test_the_shape_measures_report_the_real_shape(self):
        X, y = _wide()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = evaluate(X, y, "wide.csv")
        assert row[f"{MFE_COLUMN_PREFIX}nr_inst"].iloc[0] == 40
        assert row[f"{MFE_COLUMN_PREFIX}nr_attr"].iloc[0] == 500
        assert row[f"{MFE_COLUMN_PREFIX}attr_to_inst"].iloc[0] == pytest.approx(500 / 40)


def test_get_complexity_forwards_its_arguments():
    """``n_neighbors``/``n_components`` were accepted and then ignored.

    The body hardcoded ``Isomap(n_neighbors=10, n_components=2)``, so passing
    anything else silently did nothing.
    """
    X, _ = _fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = get_complexity(X)
        changed = get_complexity(X, n_neighbors=25, n_components=4)
    assert default != changed, "n_neighbors/n_components are still being ignored"


class TestSchemaHelpers:
    """The complexity-column schema is owned here, and two callers depend on it.

    ``QuantumSage`` uses the strict :func:`detect_complexity_schema` because training on
    the wrong columns is a correctness problem; ``compute_results_correlation`` uses the
    tolerant :func:`complexity_feature_columns` because a narrower plot is acceptable
    where an exception is not.
    """

    def test_the_current_schema_is_detected_from_evaluate_output(self, evaluated):
        schema, columns = detect_complexity_schema(evaluated.columns)
        assert schema == "pymfe"
        # Both halves: the natively-computed measures and the whole pyMFE block.
        assert set(NATIVE_COMPLEXITY_COLUMNS) <= set(columns)
        assert set(mfe_columns(evaluated)) <= set(columns)
        assert "Dataset" not in columns, "metadata must not be trained on as a feature"

    def test_the_legacy_schema_is_detected(self):
        schema, columns = detect_complexity_schema(
            list(LEGACY_COMPLEXITY_COLUMNS) + ["Dataset", "model", "accuracy"]
        )
        assert schema == "legacy"
        assert columns == list(LEGACY_COMPLEXITY_COLUMNS)

    def test_a_half_migrated_table_is_refused(self):
        """pyMFE columns without the native ones means the table was subset or mixed."""
        columns = ["mfe.nr_inst", "mfe.f1.mean", "Dataset"]
        with pytest.raises(ValueError, match="Intrinsic_Dimension"):
            detect_complexity_schema(columns)

    def test_an_unrecognized_table_is_refused_with_both_options_named(self):
        with pytest.raises(ValueError, match="neither"):
            detect_complexity_schema(["Dataset", "model", "accuracy"])

    def test_the_sample_count_column_is_mapped_for_both_schemas(self, evaluated):
        """QSage's calculate_SLGH looks this up rather than naming one schema's column."""
        assert SAMPLE_COUNT_COLUMN["pymfe"] in evaluated.columns
        assert SAMPLE_COUNT_COLUMN["legacy"] in LEGACY_COMPLEXITY_COLUMNS

    def test_the_tolerant_helper_returns_the_whole_block(self, evaluated):
        columns = complexity_feature_columns(evaluated.columns)
        assert set(NATIVE_COMPLEXITY_COLUMNS) <= set(columns)
        assert set(mfe_columns(evaluated)) <= set(columns)

    def test_the_tolerant_helper_returns_empty_rather_than_raising(self):
        assert complexity_feature_columns(["Dataset", "model", "accuracy"]) == []

    def test_correlation_analysis_sees_the_whole_block(self, evaluated):
        """``compute_results_correlation`` named its 21 feature columns literally.

        Once QProfiler's block became pyMFE-backed, that list would have matched only
        the ten surviving native names and none of the new ones -- silently dropping
        every landmarking and Lorena-complexity feature from the analysis, with nothing
        in the output saying the analysis had narrowed.
        """
        from qbiocode.visualization.visualize_correlation import compute_results_correlation

        rng = np.random.default_rng(0)
        rows = []
        for iteration in (1, 2, 3):
            row = evaluated.iloc[0].copy()
            # Vary the features so a rank correlation is defined at all.
            for name in mfe_columns(evaluated):
                row[name] = float(row[name]) * (1.0 + 0.05 * iteration)
            for model in ("rf", "svc"):
                rows.append(
                    row.to_dict()
                    | dict(
                        model=model,
                        embeddings="none",
                        iteration=iteration,
                        accuracy=float(rng.uniform(0.6, 0.9)),
                        f1_score=float(rng.uniform(0.6, 0.9)),
                        auc=float(rng.uniform(0.6, 0.9)),
                        time=1.0,
                    )
                )
        frame = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, correlations = compute_results_correlation(frame, thresh=0.5)

        correlated = set(correlations["feature"].unique())
        for expected in (
            "Intrinsic_Dimension",
            f"{MFE_COLUMN_PREFIX}one_nn.mean",
            f"{MFE_COLUMN_PREFIX}f1.mean",
            f"{MFE_COLUMN_PREFIX}n1",
        ):
            assert expected in correlated, f"{expected} dropped from correlation analysis"


class TestTheMixedSchemaHazard:
    """A table concatenating both complexity schemas must be refused, not guessed at.

    This is the failure the schema detection originally got wrong, and it is the one
    most likely to happen in practice: append a fresh QProfiler run to an older
    results table and train QSage on the result.

    It is invisible to column-name inspection, structurally. All ten
    ``NATIVE_COMPLEXITY_COLUMNS`` are a subset of ``LEGACY_COMPLEXITY_COLUMNS``, so the
    legacy rows supply every native column the ``'pymfe'`` test looks for, and the
    fresh rows supply the ``mfe.`` prefix -- the union satisfies both halves of the
    check. The original guard only fired when native columns were *missing*, which in
    this concatenation can never happen.

    The consequence was silent rather than loud: read as ``'pymfe'``, the legacy rows
    contribute NaN across all 115 pyMFE columns, ``train_sub_sages`` maps those to zeros
    via ``X.replace([inf, -inf], nan).fillna(0)``, and the 13 legacy-only features that
    the majority of rows actually have are dropped from training. Measured on the
    committed 576-row benchmark table plus fresh rows: 98.6% of rows were entirely NaN
    across the pyMFE block, and nothing warned.
    """

    @staticmethod
    def _mixed():
        """The committed legacy benchmark table with fresh pyMFE rows appended."""
        legacy = pd.read_csv(REPO_ROOT / "tutorial" / "QSage" / "data" / "qprofiler_benchmarks.csv")
        X, y = _fixture()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fresh = evaluate(X, y, "class_data-1.csv")
        fresh = pd.concat(
            [
                fresh.assign(
                    model=model, embeddings="none", iteration=1,
                    accuracy=0.8, f1_score=0.8, auc=0.8, Model_Parameters="{}",
                )
                for model in ("rf", "svc")
            ],
            ignore_index=True,
        )
        return legacy, fresh, pd.concat([legacy, fresh], ignore_index=True)

    def test_a_mixed_table_is_refused_rather_than_read_as_pymfe(self):
        _, _, mixed = self._mixed()
        with pytest.raises(ValueError, match="mixes both complexity schemas"):
            detect_complexity_schema(mixed)

    def test_the_refusal_counts_the_rows_so_the_cause_is_obvious(self):
        """A bare "mixed schemas" would leave the user guessing which rows are wrong."""
        _, _, mixed = self._mixed()
        with pytest.raises(ValueError) as failure:
            detect_complexity_schema(mixed)
        message = str(failure.value)
        assert "576 of 578" in message, message
        assert "concatenation" in message
        # And it says what to do about it.
        assert "one schema at a time" in message or "re-run QProfiler" in message

    def test_quantum_sage_refuses_it_too(self):
        """QSage is the consumer that would silently train on zeros, so it must refuse."""
        from qbiocode.apps.sage.sage import QuantumSage

        _, _, mixed = self._mixed()
        mixed = mixed.assign(datatype="ld", model_embed_datatype="x")
        with pytest.raises(ValueError, match="mixes both complexity schemas"):
            QuantumSage(mixed)

    def test_it_is_the_rows_that_give_it_away_not_the_column_names(self):
        """Pins the reason the frame must be passed rather than just the columns.

        If this ever starts raising on the column list alone, the detection has changed
        shape and the docstring above is stale.
        """
        _, _, mixed = self._mixed()
        schema, _ = detect_complexity_schema(list(mixed.columns))
        assert schema == "pymfe", (
            "column names alone now distinguish a mixed table; the row-level check "
            "may no longer be necessary -- re-read detect_complexity_schema's Note"
        )

    @pytest.mark.parametrize("which", ["legacy", "pymfe"])
    def test_each_schema_alone_is_still_accepted(self, which):
        """The guard must not make a clean single-schema table collateral damage."""
        legacy, fresh, _ = self._mixed()
        frame = legacy.assign(datatype="ld") if which == "legacy" else fresh
        schema, columns = detect_complexity_schema(frame)
        assert schema == which
        assert columns
