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

"""The pyMFE complexity block and the models still compose into one results row.

``evaluate()`` and ``model_run()`` are each well covered on their own, and neither
covers the thing QProfiler actually does with them: it merges one complexity row and
one metrics dict into a *single* record with ``dict.update``, and appends that record
to ``ModelResults.csv`` with ``csv.writer``, writing the header once -- from the keys
of whichever row happened to be first (``qprofiler.py``, the
``if csvfile.tell() == 0`` block). Every consumer downstream then reads that file
rather than either frame. Replacing the 23-column complexity block with a 141-column
pyMFE-backed one moved all three of those seams at once, and each fails silently
rather than loudly:

* ``dict.update`` has no notion of a collision. A pyMFE measure whose name happened
  to equal a metric key, or a metadata key, would not raise -- one of the two values
  would simply never reach the file, and the row would be one column narrower than
  the schema everyone believes in.
* pyMFE emits a bare name for a measure that is scalar on the given dataset and
  ``<name>.<summary>`` when it is vector-valued (see
  ``mfe_features.mfe_column_names``), and which applies depends on the data. QProfiler
  calls ``evaluate()`` twice per iteration -- once on the raw matrix and once per
  embedding, at ``n_components`` width -- so if any measure ever flipped between the
  two shapes the later rows would carry keys the header never had. ``csv.writer``
  writes them anyway, and every field after the divergence point is shifted by one:
  a ModelResults.csv that still parses, still detects as ``pymfe``, and trains QSage
  on scrambled features.
* ``compute_results_correlation`` had a hardcoded list of the 21 pre-pyMFE column
  names. Left alone it would have kept working -- correlating the ten names that
  survived the rewrite and none of the 115 new ones, narrowing the analysis with
  nothing in the output saying so.

So this module assembles the QProfiler-shaped table the way ``qprofiler.py`` does --
real ``evaluate()`` calls at two different widths, real fits through ``model_run()``,
merged by ``dict.update``, written and read back through ``csv`` -- and then hands it
to every consumer in turn: QuantumSage, ``compute_results_correlation``,
``qml_winner`` and ``combine_results``.

Two of those consumers are known to be broken independently of the pyMFE work, and
the tests that name them are ``xfail(strict=True)`` so that a fix cannot land
unnoticed: ``combine_results`` deletes the resumed run's ``Dataset`` column, and
``qml_winner``'s ``winner_eval_score`` concatenates two differently-indexed frames.
The second one is *worse* on the wider table -- the misalignment fabricates NaN in
proportion to the column count -- which is exactly why it belongs here.

Cheapest possible models (``dt``, ``lr``, ``nb``, all under 0.01 s) and four tiny
datasets: this is a test of the seam, not of the learners.
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

# Imported the way qprofiler.py imports them -- from the package root -- so a name that
# stopped being re-exported fails here too.
from qbiocode import evaluate, get_embeddings, model_run, scale_train_test
from qbiocode.apps.sage.sage import QuantumSage, calculate_SLGH
from qbiocode.evaluation.dataset_evaluation import (
    LEGACY_COMPLEXITY_COLUMNS,
    NATIVE_COMPLEXITY_COLUMNS,
    SAMPLE_COUNT_COLUMN,
    complexity_feature_columns,
    detect_complexity_schema,
    mfe_columns,
)
from qbiocode.evaluation.task_spectrum import TASK_COLUMN_PREFIX, task_column_names
from qbiocode.utils.combine_evals_results import combine_results
from qbiocode.utils.qc_winner_finder import qml_winner
from qbiocode.visualization.visualize_correlation import compute_results_correlation

from .conftest import N_FEATURES, write_dataset

#: Named after the committed datasets (``tutorial/QProfiler/data/ld_data/class_data-1.csv``)
#: because ``compute_results_correlation`` derives ``datatype`` by stripping ``-.*``
#: from the file name: these four therefore share one datatype and are correlated
#: against each other, which is what the real analysis does.
DATASETS = [f"class_data-{i}.csv" for i in range(1, 5)]
MODELS = ["dt", "lr", "nb"]
EMBEDDINGS = ["pca", "none"]
N_COMPONENTS = 2
SEED = 7
ITERATION = 1

#: Size of the pyMFE half of a complexity row. Not a free parameter: QSage's feature
#: list, ``compute_results_correlation``'s analysis and every committed
#: ModelResults.csv were all sized against it, so a change here is a schema change.
N_MFE_COLUMNS = 115

#: Size of the target-spectrum block, the third source in a complexity row.
#:
#: Derived from :func:`task_column_names` rather than written out, deliberately, and
#: differently from ``N_MFE_COLUMNS`` above. pyMFE's width is fixed by an external
#: package and a curated list, so a literal there is a schema assertion worth making.
#: This block's width is a function of its own options -- ``n_permutations=0`` drops the
#: eight ``_z`` columns and ``stability=True`` adds five ``_cv_k`` ones -- so a literal
#: would break the moment ``evaluate`` was called with different ``task_kwargs``, which
#: is a supported thing to do. The arithmetic below stays exact either way.
N_TASK_COLUMNS = len(task_column_names())

#: Width of the complexity block ``evaluate()`` emits: three sources, no shared names.
N_COMPLEXITY_COLUMNS = len(NATIVE_COMPLEXITY_COLUMNS) + N_MFE_COLUMNS + N_TASK_COLUMNS

#: The keys ``model_evaluation.modeleval`` puts in every result dict with tuning off.
#: ``time`` is in there too -- it is wall-clock, but it is a column of the results
#: file all the same, so it counts against the schema arithmetic below.
METRIC_KEYS = ("model", "accuracy", "f1_score", "time", "auc", "Model_Parameters")

#: What QProfiler adds around the two halves.
METADATA_KEYS = ("Dataset", "embeddings", "iteration")

#: 'Dataset' is written by QProfiler *and* returned by evaluate(), with the same
#: value both times. It is the one deliberate overlap between the two halves; any
#: other shared name would be silent data loss.
SHARED_KEYS = ("Dataset",)


def _dataset(directory: Path, name: str, seed: int):
    """One tiny learnable dataset, read back from disk as QProfiler reads it."""
    path = write_dataset(directory, name=name, seed=seed)
    frame = pd.read_csv(path)
    y = frame["label"].to_numpy()
    X = frame.drop(columns=["label"]).to_numpy()
    return X, y


def _profiler_rows(tmp_path: Path):
    """Assemble QProfiler's ModelResults rows, by the same steps ``qprofiler.py`` takes.

    Deliberately not a subprocess run of the CLI: the point is to hold the merged
    record in hand and inspect the merge itself, which the CLI only ever shows
    through the file it writes. ``tests/integration/test_qprofiler_end_to_end.py``
    owns the CLI half.
    """
    rows = []
    raw_rows = []
    for offset, name in enumerate(DATASETS):
        X, y = _dataset(tmp_path / "data", name, seed=100 + offset)

        # qprofiler.py:370 -- evaluate() on the raw, unembedded matrix. This is the
        # call site that fills RawDataEvaluation.csv, at the dataset's full width.
        raw_rows.append(evaluate(pd.DataFrame(X), y, name))

        # Distinct-but-reproducible split per iteration, exactly as qprofiler.py does it.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=SEED + ITERATION
        )
        X_train, X_test = scale_train_test(X_train, X_test, scaling="MinMaxScaler")

        for embedding in EMBEDDINGS:
            X_train_emb, X_test_emb = get_embeddings(
                embedding, X_train, X_test, n_neighbors=30,
                n_components=N_COMPONENTS, method=None, quvine_args={},
            )
            # qprofiler.py:432 -- evaluate() again, on the *embedded* training matrix.
            # 'pca' is N_COMPONENTS wide and 'none' is N_FEATURES wide, so the two
            # widths that share one CSV header are both represented.
            complexity = evaluate(pd.DataFrame(X_train_emb), y_train, name).to_dict(
                orient="records"
            )[0]
            args = {
                "model": list(MODELS), "seed": SEED, "n_jobs": 1, "grid_search": False,
            }
            data_key = "_".join([name.removesuffix(".csv"), embedding,
                                 str(N_COMPONENTS), str(ITERATION)])
            results = model_run(
                X_train_emb, X_test_emb, y_train, y_test, data_key, args
            )
            for key, value in results.items():
                if not key.startswith("results_"):
                    continue
                # The merge under test, key for key as qprofiler.py performs it.
                row = {"Dataset": name, "embeddings": embedding}
                row.update(complexity)
                row.update({"iteration": ITERATION})
                row.update(value[0])
                rows.append(row)
    return rows, pd.concat(raw_rows, ignore_index=True)


def _to_csv_text(rows) -> str:
    """Write the records the way ``qprofiler.py`` writes ModelResults.csv.

    Deliberately not ``DataFrame.to_csv``. QProfiler keeps one accumulating
    ``model_results`` dict, ``update``s it per result, writes the header from
    ``.keys()`` only on the first row (``if csvfile.tell() == 0``) and then
    ``.values()`` for every row -- so a record whose key set has drifted is written
    at its own width instead of being reindexed onto the header. Building a
    DataFrame first would paper over exactly the defect this file exists to catch,
    because the constructor unions the keys and pads the gaps with NaN.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    accumulating = {}
    for index, row in enumerate(rows):
        accumulating.update(row)
        if index == 0:
            writer.writerow(accumulating.keys())
        writer.writerow(accumulating.values())
    return buffer.getvalue()


@pytest.fixture(scope="module")
def assembled(tmp_path_factory):
    """The merged results table, the raw-data evaluations, and the CSV text."""
    rows, raw = _profiler_rows(tmp_path_factory.mktemp("profiler"))
    results = pd.DataFrame(rows)
    return {"rows": rows, "results": results, "raw": raw,
            "csv_text": _to_csv_text(rows)}


@pytest.fixture(scope="module")
def from_csv(assembled):
    """The results table as a consumer gets it: read back from ModelResults.csv."""
    return pd.read_csv(io.StringIO(assembled["csv_text"]))


@pytest.fixture(scope="module")
def sage_input(from_csv):
    """The round-tripped table plus the two metadata columns the QSage tutorial adds."""
    frame = from_csv.copy()
    frame["datatype"] = [str(name).split("-")[0] for name in frame["Dataset"]]
    frame["model_embed_datatype"] = (
        frame["model"] + "_" + frame["embeddings"] + "_" + frame["datatype"]
    )
    return frame


@pytest.fixture(scope="module")
def trained(sage_input):
    """One QSage, trained once on the real table. The smallest search that still fits.

    This is a seam test, not a test of surrogate quality: a wildly negative r2 on
    eight held-out rows is expected and is not what any assertion here reads.
    """
    sage = QuantumSage(data_input=sage_input)
    sage.train_sub_sages(test_size=0.3, sage_type="random_forest", n_iter=1, cv=2)
    return sage


@pytest.fixture(scope="module")
def correlations(sage_input):
    _, frame = compute_results_correlation(sage_input.copy())
    return frame


class TestTheMergedRow:
    """One record, assembled from two halves that never used to meet in a test."""

    def test_both_halves_reach_the_record(self, assembled):
        """The complexity block and the metrics are both there, in full.

        Either half could go missing without an exception: the merge is a chain of
        ``dict.update`` calls, so a complexity frame that came back empty, or a
        results dict that lost a key, produces a narrower row and no error at all.
        """
        row = assembled["rows"][0]
        missing_native = [n for n in NATIVE_COMPLEXITY_COLUMNS if n not in row]
        assert not missing_native, (
            f"the natively-computed half did not survive the merge: {missing_native}"
        )
        assert len(mfe_columns(assembled["results"])) == N_MFE_COLUMNS, (
            "the pyMFE half of the record changed size; downstream consumers "
            "(QSage's feature list, the correlation analysis) were sized against 115"
        )
        missing_metrics = [k for k in METRIC_KEYS if k not in row]
        assert not missing_metrics, (
            f"the metrics half did not survive the merge: {missing_metrics}"
        )

    def test_nothing_in_one_half_silently_overwrites_the_other(self, assembled):
        """A shared key would be absorbed by ``dict.update``, losing a column's value.

        This is the collision test the schema arithmetic makes precise: the record
        must be exactly as wide as its parts, minus the one name QProfiler and
        ``evaluate`` deliberately agree on (``Dataset``).
        """
        row = assembled["rows"][0]
        # 'Dataset' plus all three blocks of one evaluate() row.
        one_evaluate_row = 1 + N_COMPLEXITY_COLUMNS

        expected = one_evaluate_row + len(METADATA_KEYS) + len(METRIC_KEYS) - len(SHARED_KEYS)
        assert len(row) == expected, (
            f"the merged record has {len(row)} keys, expected {expected}. A count "
            "below that means two of the three sources share a name and one value "
            "was discarded by dict.update"
        )

        # From the real evaluate() output, so the disjointness is asserted against
        # the column set this run actually produced rather than a remembered list.
        complexity_names = set(assembled["raw"].columns)
        overlap = complexity_names & (set(METRIC_KEYS) | set(METADATA_KEYS))
        assert overlap == set(SHARED_KEYS), (
            f"complexity columns collide with metric/metadata keys: "
            f"{sorted(overlap - set(SHARED_KEYS))}"
        )

    def test_the_prefix_belongs_to_the_pymfe_block_alone(self, assembled):
        """``mfe.`` is what every consumer keys the schema off, so nothing else may wear it.

        ``detect_complexity_schema`` decides ``'pymfe'`` on the presence of a single
        prefixed column, and ``complexity_feature_columns`` hands *every* prefixed
        column to QSage as a feature. A metric or a metadata column that acquired the
        prefix would be trained on as though it were a dataset property.
        """
        prefixed = mfe_columns(assembled["results"])
        for name in (*METRIC_KEYS, *METADATA_KEYS):
            assert name not in prefixed
        assert not set(prefixed) & set(NATIVE_COMPLEXITY_COLUMNS)

        # And the prefix is the whole of what selection adds to the native block: the
        # features QSage would train on are the native columns plus the prefixed ones
        # and nothing else, so no metric or metadata column can leak in as a feature.
        _, features = detect_complexity_schema(assembled["results"].columns)
        task = [c for c in assembled["results"].columns if str(c).startswith(TASK_COLUMN_PREFIX)]
        assert features == list(NATIVE_COMPLEXITY_COLUMNS) + prefixed + task

    def test_every_row_is_as_wide_as_the_header_whichever_width_it_came_from(self, assembled):
        """The invariant QProfiler's append-mode writer silently depends on.

        The header is written once, from the first row's keys, and every later row is
        written positionally. The rows here come from two different ``evaluate()``
        widths -- ``pca`` at ``n_components=2`` and ``none`` at the dataset's own
        five -- and from four datasets, which is the situation in which pyMFE could
        report a measure as scalar for one and vector-valued for another. If it ever
        does, this is where it shows: the record gains a key the header never had and
        every field after it is shifted by one, in a file that still parses cleanly.

        The key sets are compared as well as the field counts, because the two catch
        opposite halves of the drift. A key that *appears* late makes the file ragged.
        A key that *disappears* does not: ``model_results`` is one dict reused across
        rows, so the value from the previous record is carried forward under the
        vanished name and written again -- same width, wrong number, nothing to see in
        the file. Only the records themselves show that one.

        Parsed with ``csv.reader`` rather than by lines on purpose: an sklearn
        estimator repr wraps, so ``Model_Parameters`` legitimately contains newlines
        inside a quoted field and a line count is not a record count.
        """
        key_sets = {frozenset(row) for row in assembled["rows"]}
        assert len(key_sets) == 1, (
            "the records do not agree on their key set, so the header written from "
            "the first one does not describe the rest: "
            f"{sorted(set.union(*map(set, key_sets)) - set.intersection(*map(set, key_sets)))}"
        )

        records = list(csv.reader(io.StringIO(assembled["csv_text"])))
        header, *data = records
        widths = {len(record) for record in data}
        assert widths == {len(header)}, (
            f"header is {len(header)} fields wide but rows are {sorted(widths)}; "
            "ModelResults.csv is ragged and every column after the divergence is shifted"
        )
        assert len(data) == len(DATASETS) * len(EMBEDDINGS) * len(MODELS)

        # And the two call sites really did disagree about width, so the assertion above
        # was not vacuous.
        widths_evaluated = {row["mfe.nr_attr"] for row in assembled["rows"]}
        assert widths_evaluated == {float(N_COMPONENTS), float(N_FEATURES)}


class TestTheCsvRoundTrip:
    """ModelResults.csv is the real handoff to QSage; the schema has to survive it."""

    def test_the_column_names_and_their_order_come_back_unchanged(self, assembled, from_csv):
        assert list(from_csv.columns) == list(assembled["results"].columns)

        # A duplicated name would come back mangled ('mfe.f1.mean.1') instead of
        # raising, and an off-by-one header would leave an 'Unnamed: N' column --
        # both of which a downstream `results_df[features]` would then miss.
        mangled = [c for c in from_csv.columns
                   if str(c).startswith("Unnamed") or str(c).endswith(".1")]
        assert not mangled, f"round trip produced {mangled}"

    def test_every_complexity_column_comes_back_numeric(self, assembled, from_csv):
        """Object dtype here is what makes QSage fail, and only in memory.

        ``evaluate`` coerces its measures precisely because a frame built from a dict
        that also holds the ``Dataset`` string is object-dtype throughout, and
        ``calculate_SLGH`` then raises "loop of ufunc does not support argument 0 of
        type numpy.float64" on it. The CSV round trip normally hides that by
        re-inferring dtypes -- unless a measure starts emitting something
        unparseable, in which case one column comes back as object and only the
        column that lost its dtype fails.
        """
        features = complexity_feature_columns(from_csv.columns)
        assert len(features) == N_COMPLEXITY_COLUMNS

        not_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(from_csv[c])]
        assert not not_numeric, f"complexity columns came back non-numeric: {not_numeric}"

        for metric in ("accuracy", "f1_score", "auc", "time"):
            assert pd.api.types.is_numeric_dtype(from_csv[metric])

        np.testing.assert_allclose(
            from_csv[features].to_numpy(dtype=float),
            assembled["results"][features].to_numpy(dtype=float),
            rtol=1e-6,
            err_msg="a complexity value changed value across the CSV round trip",
        )

    def test_the_schema_is_still_detected_after_the_round_trip(self, from_csv):
        """Detection runs on the file, not on the frame ``evaluate`` returned.

        Worth its own assertion because the two inputs differ: the in-memory frame
        carries the names ``evaluate`` chose, the file carries whatever ``csv``
        wrote and pandas re-read -- a mangled duplicate or a shifted header changes
        the second without touching the first.
        """
        schema, features = detect_complexity_schema(from_csv.columns)
        assert schema == "pymfe"
        assert len(features) == N_COMPLEXITY_COLUMNS

        # The metrics are what QSage regresses *onto*; selecting one as a feature
        # would train each surrogate on its own target.
        leaked = [name for name in features if name in (*METRIC_KEYS, *METADATA_KEYS)]
        assert not leaked, f"metric/metadata columns selected as complexity features: {leaked}"


class TestQuantumSageOnARealTable:
    """QSage trained on values ``evaluate()`` actually produces, not on uniform(1, 10).

    ``tests/test_sage_contract.py`` pins the column handling against a hand-listed
    24-column synthetic frame. That fixture cannot exhibit what the production code
    defends against: the real block is 141 columns wide and carries condition numbers
    in the thousands, negative log-densities and near-zero Lorena measures -- which is
    why ``train_sub_sages`` replaces infinities before fitting and why
    ``calculate_SLGH`` takes logs of two of these columns.
    """

    def test_it_detects_the_pymfe_schema_and_takes_the_whole_block(self, trained):
        assert trained._complexity_schema == "pymfe"
        assert len(trained._columns_data_features) == (
            N_COMPLEXITY_COLUMNS
        )
        assert trained._available_models == sorted(MODELS)

    def test_it_trains_and_predicts_a_finite_score_for_every_model(self, trained, sage_input):
        features = sage_input[trained._columns_data_features].iloc[[0]]
        predictions = trained.predict(features, metric="f1_score")

        assert sorted(predictions["model"]) == sorted(MODELS)
        assert np.isfinite(predictions["f1_score"]).all(), (
            "a surrogate predicted a non-finite score from real complexity values"
        )
        assert np.isfinite(predictions["r2"]).all()
        assert predictions["f1_score*r2"].is_monotonic_decreasing

    def test_the_derived_feature_reads_the_pymfe_sample_count(self, trained, sage_input):
        """SLGH survived the schema change only because the sample count is looked up.

        ``# Samples`` is gone from this table; the count now lives in
        ``mfe.nr_inst``. If ``calculate_SLGH`` ever went back to naming a column, it
        would raise here -- and the value is checked against the formula rather than
        merely for finiteness, because reading the *wrong* numeric column would still
        produce a plausible number.
        """
        assert "# Samples" not in sage_input.columns
        assert SAMPLE_COUNT_COLUMN["pymfe"] in sage_input.columns

        row = sage_input[trained._columns_data_features].iloc[[0]]
        slgh = calculate_SLGH(row)["SLGH"].iloc[0]

        n_train = math.ceil(row["mfe.nr_inst"].iloc[0] * 0.7)
        expected = -math.log(row["Intrinsic_Dimension"].iloc[0] + 1e-8) - math.log(
            1.0 + row["Fisher Discriminant Ratio"].iloc[0] * n_train
        )
        assert slgh == pytest.approx(expected)
        assert math.isfinite(slgh)


class TestTheCorrelationAnalysis:
    """The regression that already happened once: a hardcoded complexity list."""

    def test_the_whole_block_is_correlated_and_not_just_the_native_survivors(
        self, correlations, sage_input
    ):
        """Exactly the table's complexity columns, all 141 of them.

        The pre-pyMFE implementation named 21 columns and skipped any that were
        absent, so on this table it would have silently correlated the ten native
        names that survived the rewrite. Set *equality* is the assertion that catches
        that: a subset check passes just as happily on ten features as on 141.
        """
        expected = complexity_feature_columns(sage_input.columns)
        assert set(correlations["feature"]) == set(expected)
        assert len(expected) == N_COMPLEXITY_COLUMNS

        groups = sage_input["model_embed_datatype"].nunique()
        metrics = ["accuracy", "f1_score", "time", "auc"]
        assert len(correlations) == groups * len(metrics) * len(expected)
        assert set(correlations["metric"]) == set(metrics)

    def test_the_pymfe_features_are_really_correlated_not_merely_listed(self, correlations):
        """Listing a feature and computing its correlation are different things.

        ``spearmanr`` returns NaN for a column that is constant across the datasets
        in a group, which several pyMFE measures legitimately are, so this cannot ask
        for all of them. It can ask that the pyMFE block contributes far more real
        coefficients than the ten native columns could on their own -- which is the
        difference between the analysis being widened and merely appearing to be.
        """
        computed = correlations[correlations["correlation"].notna()]
        prefixed = {f for f in computed["feature"] if str(f).startswith("mfe.")}
        assert len(prefixed) > 50, (
            f"only {len(prefixed)} pyMFE features produced a correlation coefficient; "
            "the block is listed in the output but is not being correlated"
        )
        assert correlations["median_metric"].notna().all()
        assert ((correlations["frac_gt_thresh"] >= 0) & (correlations["frac_gt_thresh"] <= 1)).all()


def _scored_table(from_csv, winners, qml_name):
    """The round-tripped table with ``qml_name`` winning on ``winners`` and nowhere else.

    ``qml_winner`` groups by the parameter column, so it needs the *file* -- the
    in-memory table holds a real dict under ``Model_Parameters`` and grouping on it
    raises ``TypeError: unhashable type: 'dict'``. QProfiler always writes the CSV
    first, so reading it back is the faithful input.

    The QML model is named the way ``model_run`` names it: the ``compute_ml_dict`` key
    it dispatched on, passed straight through as ``model=method`` and therefore lower
    case, plus whatever the computing function appends to it for a tuned twin or for
    QPL's per-head fan-out (``TestTheWinnerFinder.QML_LABELS`` sweeps all four shapes).
    This fixture used to spell it ``QSVC`` instead, as a workaround --
    ``qml_winner`` selected its quantum rows with
    ``.isin(["QSVC", "QNN", "VQC", "PQK"])`` against a column every writer of it spells
    in lower case, so the branch was dead on genuine output (``qml_winners.csv`` came
    back empty from every real quantum run) and upper-casing the label here was the
    only way to reach it. That defect is fixed: the label's first ``_``-token is
    lower-cased before it is matched against ``{qsvc, qnn, vqc, pqk, qpl}``. So the
    table handed to the winner finder below now carries the label a real run carries,
    and the branch is exercised end to end instead of on a spelling nothing produces.
    """
    frame = from_csv.copy()
    frame.loc[frame["model"] == "dt", "model"] = qml_name
    is_qml = frame["model"] == qml_name
    # Deterministic scores: the QML model wins outright on `winners` and loses
    # outright everywhere else, so the winner set is exact rather than tie-broken.
    frame.loc[:, "f1_score"] = 0.60
    frame.loc[is_qml & frame["Dataset"].isin(winners), "f1_score"] = 0.90
    frame.loc[is_qml & ~frame["Dataset"].isin(winners), "f1_score"] = 0.30
    return frame


def _legacy_rawevals(datasets):
    """A pre-pyMFE RawDataEvaluation, 23 complexity columns wide."""
    frame = pd.DataFrame(
        {name: np.arange(1.0, 1.0 + len(datasets)) for name in LEGACY_COMPLEXITY_COLUMNS}
    )
    frame.insert(0, "Dataset", list(datasets))
    return frame


class TestTheWinnerFinder:
    """``qml_winner`` slices its output positionally, and the table got five times wider."""

    WINNERS = DATASETS[:2]

    #: Every shape of label a real quantum row wears, and the reason the finder matches
    #: on the lower-cased first ``_``-token rather than on a list of names.
    #: ``model_run`` dispatches ``compute_ml_dict[method]`` and passes ``model=method``,
    #: so an untuned run writes the bare key and the ``_opt`` twin writes ``<key>_opt``;
    #: ``compute_qpl`` then fans out one row per classical head as ``f"{model}_{head}"``,
    #: which keeps the head as the suffix and puts the tuning marker in the *middle* of
    #: ``qpl_opt_rf``. No list of exact spellings covers the last three at any casing.
    QML_LABELS = ("qsvc", "qsvc_opt", "qpl_rf", "qpl_opt_rf")

    #: A *classical* dispatch key that is a substring of a quantum one (``svc`` inside
    #: ``qsvc``), so a matcher written as a containment test in the wrong direction
    #: admits it while still passing every case above. ``qml_winners.csv`` is read as
    #: evidence that a quantum method beat the classical ones on a dataset, so a
    #: classical row admitted to it is a false claim of quantum advantage.
    CLASSICAL_LOOKALIKE = "svc"

    #: The bare, untuned QSVC label, for the tests that are not about the name matching.
    #: ``compute_qsvc`` defaults to this same lower-case spelling, so it is the string
    #: that lands in the ``model`` column of ModelResults.csv -- not a spelling picked to
    #: satisfy the winner finder, which is what ``_scored_table`` explains.
    QML_NAME = "qsvc"

    @pytest.mark.parametrize("qml_name", QML_LABELS)
    def test_the_winning_datasets_are_found_on_a_pymfe_wide_evaluation(
        self, from_csv, assembled, tmp_path, qml_name
    ):
        """The quantum branch runs on every label shape a run writes, and only on those.

        ``qml_winner`` used to select quantum rows with
        ``.isin(["QSVC", "QNN", "VQC", "PQK"])`` against a column every writer of it
        spells in lower case, so no genuine table ever entered this branch and
        ``qml_winners.csv`` came back empty from every real quantum run. This module
        reached the branch by upper-casing the label in ``_scored_table``, which meant
        the widened table was judged on a spelling no run produces. Matching is now done
        on the lower-cased first ``_``-token against ``{qsvc, qnn, vqc, pqk, qpl}``.

        The parametrization is what pins that whole matcher rather than only its casing
        half. Lower-casing the old list of four names -- the obvious half-fix, and one
        that passes on a table whose only quantum label is a bare ``qsvc`` -- still
        admits neither the tuned twin nor either QPL shape, so ``qsvc_opt``,
        ``qpl_rf`` and ``qpl_opt_rf`` are the cases that go red on it. The two
        ``qpl_*`` cases carry a second defect of their own: ``qpl`` was missing from the
        family list entirely, at any casing.

        Both sides are pinned. Too narrow a matcher finds no quantum winner at all and
        ``qml_winner`` returns None; too loose a one -- anything that also claims
        ``lr`` or ``nb`` -- makes the winner set all four datasets instead of the two
        where the quantum model actually wins, which the equality below rejects.
        ``test_a_classical_label_that_merely_looks_quantum_is_not_a_winner`` covers the
        one loose matcher that survives both of those. ``tests/test_quantum_remainder.py``
        sweeps the same matcher across the rest of the family and a wider set of
        classical labels; what this module adds is that it happens on the 126-column
        pyMFE table, assembled by the steps QProfiler takes.
        """
        scored = _scored_table(from_csv, self.WINNERS, qml_name)
        result = qml_winner(scored, assembled["raw"], str(tmp_path), "tag")

        assert result is not None, (
            f"qml_winner found no QML winner on a table where {qml_name!r} wins "
            "outright, so a real quantum run of that kind writes an empty "
            "qml_winners.csv"
        )
        winners, _, _ = result
        assert sorted(winners["Dataset"].unique()) == sorted(self.WINNERS)

    def test_a_classical_label_that_merely_looks_quantum_is_not_a_winner(
        self, from_csv, assembled, tmp_path
    ):
        """Widening an exact comparison is how over-matching gets in, so pin that too.

        The fix to the dead upper-case ``.isin`` had to loosen the comparison, and the
        sibling test above cannot tell a first-``_``-token match from a containment test
        written the wrong way round (``label in family``): both accept every label a
        quantum run writes. They differ on ``svc``, a classical dispatch key that is a
        substring of ``qsvc`` -- and admitting it is the expensive direction of the bug,
        because ``qml_winners.csv`` is read as evidence of quantum advantage on a
        dataset, so a classical row in it is a claim the run does not support.

        ``best_across_split.csv`` is asserted to exist so that a ``None`` here is known
        to come from the name matching rather than from the function failing before it
        ever reached the quantum branch -- a refusal to write is exactly what the
        original defect looked like, so the two must not be conflated.
        """
        scored = _scored_table(from_csv, self.WINNERS, self.CLASSICAL_LOOKALIKE)
        result = qml_winner(scored, assembled["raw"], str(tmp_path), "lookalike")

        admitted = None if result is None else sorted(set(result[0]["model"]))
        assert result is None, (
            f"{self.CLASSICAL_LOOKALIKE!r} is a classical model, but qml_winner "
            f"reported it as a quantum winner: {admitted}"
        )
        assert (tmp_path / "lookalike_best_across_split.csv").exists(), (
            "qml_winner returned before reaching its quantum branch at all, so the "
            "None above says nothing about the name matching"
        )
        assert not (tmp_path / "lookalike_qml_winners.csv").exists(), (
            "a table with no quantum model in it still produced a qml_winners.csv"
        )

    @pytest.mark.parametrize("width", ["legacy", "pymfe"])
    def test_the_score_triple_survives_however_wide_the_evaluation_is(
        self, from_csv, assembled, tmp_path, width
    ):
        """``qc_method_and_score.iloc[:, -3:]`` is a positional slice, and that reads
        as a hazard on a 126-column table.

        It is not: the slice is taken from the five-column groupby result, not from
        the evaluation frame, so the three columns it keeps are the parameter column,
        the model and the F1 score at either width. Pinning it both ways is what
        keeps a future change to the groupby keys -- one more grouping column and the
        slice silently drops ``model`` -- from becoming a wrong ``_winner_score.csv``
        that nobody reads closely.
        """
        raw = (
            assembled["raw"] if width == "pymfe"
            else _legacy_rawevals(assembled["raw"]["Dataset"])
        )
        scored = _scored_table(from_csv, self.WINNERS, self.QML_NAME)
        qml_winner(scored, raw, str(tmp_path), width)

        written = pd.read_csv(tmp_path / f"{width}_winner_score.csv")
        assert list(written.columns) == ["model", "Model_Parameters", "f1_score"]
        assert set(written["model"]) == {self.QML_NAME}
        assert len(written) == len(self.WINNERS)

    @pytest.mark.xfail(
        strict=True,
        reason="qc_winner_finder.py:128, `pd.concat([winner_evals_df, "
        "winner_scores_df], axis=1)`, concatenates winner_evals_df (indexed by "
        "RawDataEvaluation row) with winner_scores_df (indexed by df_across_split "
        "row), so _winner_eval_score.csv gains a row per non-overlapping "
        "index and fills the rest with NaN. Pre-existing, but the fabricated NaN "
        "scales with the complexity block: 126 columns make it far worse than 23.",
    )
    def test_each_winning_dataset_gets_exactly_one_row_of_evaluations(
        self, from_csv, assembled, tmp_path
    ):
        scored = _scored_table(from_csv, self.WINNERS, self.QML_NAME)
        _, winner_eval_score, _ = qml_winner(
            scored, assembled["raw"], str(tmp_path), "tag"
        )

        assert len(winner_eval_score) == len(self.WINNERS)
        assert winner_eval_score["Dataset"].notna().all(), (
            "a row of _winner_eval_score.csv names no dataset"
        )
        assert winner_eval_score["f1_score"].notna().all()


class TestCombiningAResumedRun:
    """``combine_results`` is the documented restart path, and it reads these files."""

    @pytest.fixture()
    def laid_out(self, tmp_path, from_csv, assembled):
        """A previous run (one dataset, in a subdirectory) and a resumed one (the rest)."""
        previous, recent = tmp_path / "prev", tmp_path / "recent"
        (previous / "dataset_class_data-1").mkdir(parents=True)
        recent.mkdir()

        first, rest = DATASETS[0], DATASETS[1:]
        from_csv[from_csv["Dataset"] == first].to_csv(
            previous / "dataset_class_data-1" / "ModelResults.csv", index=False
        )
        assembled["raw"][assembled["raw"]["Dataset"] == first].to_csv(
            previous / "dataset_class_data-1" / "RawDataEvaluation.csv", index=False
        )
        from_csv[from_csv["Dataset"].isin(rest)].to_csv(
            recent / "ModelResults.csv", index=False
        )
        assembled["raw"][assembled["raw"]["Dataset"].isin(rest)].to_csv(
            recent / "RawDataEvaluation.csv", index=False
        )
        return previous, recent, tmp_path

    @staticmethod
    def _combine(previous, recent, output_dir):
        # Absolute output paths and no intermediates: the defaults are relative, so a
        # bare call litters the repository root with four CSVs.
        return combine_results(
            str(previous), str(recent),
            output_eval_file=str(output_dir / "RawDataEvaluation_Combined.csv"),
            output_results_file=str(output_dir / "ModelResults_Combined.csv"),
            save_intermediate=False, verbose=False,
        )

    def test_the_combined_table_keeps_the_pymfe_schema_intact(self, laid_out, from_csv):
        """Widening the block did not break the concatenation itself.

        The failure this rules out is the mixed-schema one: a legacy half and a pyMFE
        half concatenated into one table give a frame that still detects as
        ``'pymfe'`` while most of its rows are entirely NaN across the block, because
        every native column is also a legacy column. Same-schema halves must produce
        no such column, and must keep the column order the append-mode writer set.
        """
        previous, recent, output_dir = laid_out
        combined_eval, combined_results = self._combine(previous, recent, output_dir)

        assert list(combined_results.columns) == list(from_csv.columns)
        assert detect_complexity_schema(combined_results.columns)[0] == "pymfe"
        assert detect_complexity_schema(combined_eval.columns)[0] == "pymfe"

        features = complexity_feature_columns(combined_results.columns)
        dead = [c for c in features if combined_results[c].isna().all()]
        assert not dead, f"complexity columns arrived empty from the combine: {dead}"

        rows_per_dataset = len(EMBEDDINGS) * len(MODELS)
        assert len(combined_results) == len(DATASETS) * rows_per_dataset
        assert len(combined_eval) == len(DATASETS)

    @pytest.mark.xfail(
        strict=True,
        reason="combine_evals_results.py:228 and :232 read the resumed run's CSVs "
        "with index_col=0 and then reset_index(drop=True). QProfiler writes both "
        "files with index=False and 'Dataset' first, so the resumed rows lose their "
        "dataset name in both combined outputs -- and 'Dataset' is required "
        "metadata for QuantumSage and the join key for qml_winner.",
    )
    def test_the_resumed_rows_keep_their_dataset_name(self, laid_out):
        previous, recent, output_dir = laid_out
        combined_eval, combined_results = self._combine(previous, recent, output_dir)

        assert combined_results["Dataset"].notna().all()
        assert combined_eval["Dataset"].notna().all()
        assert set(combined_results["Dataset"]) == set(DATASETS)
