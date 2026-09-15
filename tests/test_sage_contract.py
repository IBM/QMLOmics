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

"""QSage can be built from QProfiler's output, and predicted from.

Both halves of that sentence were false, and each failed in a way that pointed
away from itself.

``QuantumSage.__init__`` sliced a metadata list naming *both*
``BestParams_GridSearch`` and ``Model_Parameters``. QProfiler writes exactly one
of them -- ``model_evaluation.py`` branches on ``args["grid_search"]``, and
``qc_winner_finder.py`` documents the same fact -- so construction raised
``KeyError: "['BestParams_GridSearch'] not in index"`` for a table produced with
grid search off, and the mirror-image error with it on. There was no
configuration in which QSage could read its own documented input, and the error
named a column the user had never heard of rather than the mismatch.

``predict`` then forwarded the caller's frame straight to a fitted estimator,
but ``train_sub_sages`` appends a derived ``SLGH`` column after splitting. So
passing exactly the features named in ``_columns_data_features`` -- what the
docstring asks for -- produced sklearn's "Feature names seen at fit time, yet
now missing: - SLGH", blaming the caller for a column the class derives itself.

A third contract joined those two when QProfiler's complexity block became
pyMFE-backed. The 576-row table the QSage tutorial trains on
(``tutorial/QSage/data/qprofiler_benchmarks.csv``) is in the *previous* schema and
cannot be regenerated from this repository -- it covers ``class_data-{1..16}.csv``
and only three of those are committed. So ``QuantumSage`` detects the schema it is
handed instead of naming one, and both must keep working: the tests below run the
whole construct-train-predict path twice, once per schema.

The fixture below is deliberately synthetic and tiny: these are contract tests
about column handling, and the committed benchmark table is exercised by the
QSage notebook instead.
"""

from importlib import import_module
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_sage = import_module("qbiocode.apps.sage.sage")
_evaluation = import_module("qbiocode.evaluation.dataset_evaluation")

LEGACY_FEATURES = [
    "# Features", "# Samples", "Feature_Samples_ratio", "Intrinsic_Dimension",
    "Condition number", "Fisher Discriminant Ratio", "Total Correlations",
    "Mutual information", "# Non-zero entries", "# Low variance features",
    "Variation", "std_var", "Coefficient of Variation %", "std_co_of_v",
    "Skewness", "std_skew", "Kurtosis", "std_kurt", "Mean Log Kernel Density",
    "Isomap Reconstruction Error", "Fractal dimension", "Entropy", "std_entropy",
]

# Enough of the current schema to exercise column handling, not the whole block:
# every natively-computed column (detect_complexity_schema requires all of them,
# since both halves come from one evaluate() call) plus a representative slice of
# the pyMFE block. `mfe.nr_inst` is not optional here -- calculate_SLGH reads it.
PYMFE_FEATURES = list(_evaluation.NATIVE_COMPLEXITY_COLUMNS) + [
    "mfe.nr_inst", "mfe.nr_attr", "mfe.attr_to_inst", "mfe.var.mean", "mfe.var.sd",
    "mfe.skewness.mean", "mfe.kurtosis.mean", "mfe.cor.mean", "mfe.mut_inf.mean",
    "mfe.class_ent", "mfe.one_nn.mean", "mfe.naive_bayes.mean",
    "mfe.linear_discr.mean", "mfe.f1.mean", "mfe.f3.mean", "mfe.n1", "mfe.l2.mean",
    "mfe.t3", "mfe.t4", "mfe.leaves", "mfe.nodes", "mfe.sil", "mfe.ch",
    "mfe.conceptvar.mean",
]

SCHEMAS = {"legacy": LEGACY_FEATURES, "pymfe": PYMFE_FEATURES}

#: Column holding the sample count in each schema. calculate_SLGH takes its log, so
#: the fixture has to set it to something sane rather than a uniform random draw.
SAMPLES_COLUMN = {"legacy": "# Samples", "pymfe": "mfe.nr_inst"}

#: A feature column present in each schema, for the missing-column assertions.
PROBE_COLUMN = {"legacy": "Entropy", "pymfe": "mfe.one_nn.mean"}

METRICS = ["accuracy", "f1_score", "auc"]
MODELS = ["rf", "svc"]


def results_table(parameter_column="Model_Parameters", n_datasets=6, schema="legacy"):
    """A QProfiler-shaped results table with one parameter column, as QProfiler writes."""
    rng = np.random.default_rng(0)
    rows = []
    for dataset in range(n_datasets):
        # Complexity features are a property of the dataset, so they repeat across
        # every (model, embedding) row for it -- exactly as in a real table.
        features = {name: float(rng.uniform(1, 10)) for name in SCHEMAS[schema]}
        features[SAMPLES_COLUMN[schema]] = 100.0
        for model in MODELS:
            for embedding in ("pca", "none"):
                row = dict(features)
                row.update(
                    Dataset=f"class_data-{dataset + 1}",
                    embeddings=embedding,
                    model=model,
                    iteration=1,
                )
                row.update({metric: float(rng.uniform(0.5, 1.0)) for metric in METRICS})
                row[parameter_column] = "{}"
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame["datatype"] = frame["Dataset"]
    frame["model_embed_datatype"] = (
        frame["model"] + "_" + frame["embeddings"] + "_" + frame["datatype"]
    )
    return frame


class TestSchemaDetection:
    """QSage reads whichever complexity schema it is handed."""

    def test_the_legacy_schema_is_recognized(self):
        sage = _sage.QuantumSage(data_input=results_table(schema="legacy"))
        assert sage._complexity_schema == "legacy"
        assert sage._columns_data_features == LEGACY_FEATURES

    def test_the_pymfe_schema_is_recognized(self):
        sage = _sage.QuantumSage(data_input=results_table(schema="pymfe"))
        assert sage._complexity_schema == "pymfe"
        # Every pyMFE column is picked up, not just a hardcoded subset -- that is the
        # point of detecting rather than naming.
        assert set(PYMFE_FEATURES) <= set(sage._columns_data_features)
        assert any(c.startswith("mfe.") for c in sage._columns_data_features)

    def test_the_committed_benchmark_table_is_still_trainable(self):
        """The table the QSage tutorial trains on is legacy-schema and unregenerable.

        ``tutorial/QSage/data/qprofiler_benchmarks.csv`` covers
        ``class_data-{1..16}.csv``; only ``-1``, ``-2`` and ``-3`` are committed, and
        the sweep that produced the other 13 is recorded nowhere. If QSage stopped
        reading this schema, the tutorial would have no training data at all.
        """
        path = REPO_ROOT / "tutorial" / "QSage" / "data" / "qprofiler_benchmarks.csv"
        frame = pd.read_csv(path)
        frame = frame.assign(datatype=frame["Dataset"], model_embed_datatype="x")
        sage = _sage.QuantumSage(data_input=frame)
        assert sage._complexity_schema == "legacy"

    def test_a_half_migrated_table_is_refused_with_the_reason(self):
        """pyMFE columns without the native ones means the table was subset or mixed."""
        frame = results_table(schema="pymfe").drop(columns=["Fisher Discriminant Ratio"])
        with pytest.raises(ValueError, match="Fisher Discriminant Ratio"):
            _sage.QuantumSage(data_input=frame)

    def test_a_table_in_neither_schema_says_so(self):
        frame = results_table(schema="legacy").drop(columns=["Entropy", "Skewness"])
        with pytest.raises(ValueError) as failure:
            _sage.QuantumSage(data_input=frame)
        message = str(failure.value)
        assert "neither" in message
        assert "'Entropy'" in message and "'Skewness'" in message


@pytest.mark.parametrize("schema", sorted(SCHEMAS))
@pytest.mark.parametrize(
    "parameter_column",
    # 'BestParams_GridSearch' is the pre-Optuna name for the tuned column; older
    # ModelResults.csv files still carry it, so QuantumSage must still read them.
    ["Model_Parameters", "BestParams_Tuned", "BestParams_GridSearch"],
)
def test_it_accepts_whichever_parameter_column_qprofiler_wrote(parameter_column, schema):
    """QProfiler writes one or the other; requiring both rejected every real table."""
    sage = _sage.QuantumSage(data_input=results_table(parameter_column, schema=schema))

    assert sage._columns_parameters == [parameter_column]
    assert sage._available_models == sorted(MODELS)
    assert sage._available_metrics == sorted(METRICS)


@pytest.mark.parametrize("schema", sorted(SCHEMAS))
def test_it_accepts_a_table_recording_no_parameters_at_all(schema):
    """Neither column is trained on, so their absence is not a reason to refuse."""
    frame = results_table(schema=schema).drop(columns=["Model_Parameters"])
    sage = _sage.QuantumSage(data_input=frame)
    assert sage._columns_parameters == []


@pytest.mark.parametrize("schema", sorted(SCHEMAS))
def test_a_genuinely_missing_metadata_column_is_named_with_what_to_do(schema):
    frame = results_table(schema=schema).drop(columns=["iteration"])
    with pytest.raises(ValueError) as failure:
        _sage.QuantumSage(data_input=frame)
    message = str(failure.value)
    assert "'iteration'" in message
    assert "ModelResults.csv" in message


@pytest.mark.parametrize("schema", sorted(SCHEMAS))
class TestPredictTakesTheDocumentedColumns:
    """``predict`` must accept ``_columns_data_features`` -- what its docstring asks for."""

    @pytest.fixture()
    def trained(self, schema):
        sage = _sage.QuantumSage(data_input=results_table(schema=schema))
        # The smallest search that still fits: this is a column-handling test, not
        # a test of surrogate quality.
        sage.train_sub_sages(test_size=0.3, sage_type="random_forest", n_iter=1, cv=2)
        return sage

    def test_the_feature_columns_alone_are_enough(self, trained, schema):
        features = results_table(schema=schema)[trained._columns_data_features].iloc[[0]]
        predictions = trained.predict(features, metric="accuracy")

        # Every model gets a row, but the *order* is a result, not an echo of the
        # input: rows are ranked by metric * r2, so a confident prediction from a
        # poorly-fitted surrogate cannot top the list on its point value alone.
        assert sorted(predictions["model"]) == sorted(MODELS)
        assert predictions["accuracy"].notna().all()
        assert predictions["accuracy*r2"].is_monotonic_decreasing

    def test_slgh_is_derived_not_demanded(self, trained, schema):
        """The caller must not have to know about a column training invented."""
        assert "SLGH" not in trained._columns_data_features

        features = results_table(schema=schema)[trained._columns_data_features].iloc[[0]]
        # Passing it explicitly is equally fine: it is recomputed, so a stale value
        # cannot reach the estimator.
        with_stale = features.assign(SLGH=-999.0)
        assert trained.predict(with_stale, metric="accuracy").equals(
            trained.predict(features, metric="accuracy")
        )

    def test_several_rows_are_refused_rather_than_silently_reduced(self, trained, schema):
        """The ranking is one row per model, so a multi-row input had nowhere to go.

        ``.predict(...)[0]`` ranked on whichever row sorted first and dropped the
        rest, with nothing in the output saying so. The frame that triggers it is the
        obvious one to build: complexity features are measured on the *embedded*
        data, so ``results_df[features].drop_duplicates()`` for a single dataset
        yields one row per (embedding, iteration), not one row.
        """
        several = results_table(schema=schema)[trained._columns_data_features].iloc[:3]
        with pytest.raises(ValueError, match="exactly one row"):
            trained.predict(several, metric="accuracy")

        # And the message says how to get to one row.
        with pytest.raises(ValueError, match=r"iloc\[\[0\]\]"):
            trained.predict(several, metric="accuracy")

    def test_a_missing_feature_names_itself_and_the_count(self, trained, schema):
        probe = PROBE_COLUMN[schema]
        features = results_table(schema=schema)[trained._columns_data_features].iloc[[0]]
        with pytest.raises(ValueError, match=re.escape(probe)):
            trained.predict(features.drop(columns=[probe]), metric="accuracy")

    def test_an_untrained_metric_is_refused_by_name(self, trained, schema):
        features = results_table(schema=schema)[trained._columns_data_features].iloc[[0]]
        with pytest.raises(ValueError, match="nope"):
            trained.predict(features, metric="nope")


@pytest.mark.parametrize("schema", sorted(SCHEMAS))
def test_predicting_before_training_says_to_train(schema):
    """Used to be a bare KeyError on an empty dict, naming the metric, not the cause."""
    sage = _sage.QuantumSage(data_input=results_table(schema=schema))
    features = results_table(schema=schema)[sage._columns_data_features].iloc[[0]]
    with pytest.raises(RuntimeError, match="train_sub_sages"):
        sage.predict(features, metric="accuracy")
