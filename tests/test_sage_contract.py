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

The fixture below is deliberately synthetic and tiny: these are contract tests
about column handling, and the committed benchmark table is exercised by the
QSage notebook instead.
"""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

_sage = import_module("qbiocode.apps.sage.sage")

FEATURES = [
    "# Features", "# Samples", "Feature_Samples_ratio", "Intrinsic_Dimension",
    "Condition number", "Fisher Discriminant Ratio", "Total Correlations",
    "Mutual information", "# Non-zero entries", "# Low variance features",
    "Variation", "std_var", "Coefficient of Variation %", "std_co_of_v",
    "Skewness", "std_skew", "Kurtosis", "std_kurt", "Mean Log Kernel Density",
    "Isomap Reconstruction Error", "Fractal dimension", "Entropy", "std_entropy",
]
METRICS = ["accuracy", "f1_score", "auc"]
MODELS = ["rf", "svc"]


def results_table(parameter_column="Model_Parameters", n_datasets=6):
    """A QProfiler-shaped results table with one parameter column, as QProfiler writes."""
    rng = np.random.default_rng(0)
    rows = []
    for dataset in range(n_datasets):
        # Complexity features are a property of the dataset, so they repeat across
        # every (model, embedding) row for it -- exactly as in a real table.
        features = {name: float(rng.uniform(1, 10)) for name in FEATURES}
        features["# Samples"] = 100.0
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


@pytest.mark.parametrize(
    "parameter_column",
    # 'BestParams_GridSearch' is the pre-Optuna name for the tuned column; older
    # ModelResults.csv files still carry it, so QuantumSage must still read them.
    ["Model_Parameters", "BestParams_Tuned", "BestParams_GridSearch"],
)
def test_it_accepts_whichever_parameter_column_qprofiler_wrote(parameter_column):
    """QProfiler writes one or the other; requiring both rejected every real table."""
    sage = _sage.QuantumSage(data_input=results_table(parameter_column))

    assert sage._columns_parameters == [parameter_column]
    assert sage._available_models == sorted(MODELS)
    assert sage._available_metrics == sorted(METRICS)


def test_it_accepts_a_table_recording_no_parameters_at_all():
    """Neither column is trained on, so their absence is not a reason to refuse."""
    frame = results_table().drop(columns=["Model_Parameters"])
    sage = _sage.QuantumSage(data_input=frame)
    assert sage._columns_parameters == []


def test_a_genuinely_missing_column_is_named_with_what_to_do():
    frame = results_table().drop(columns=["iteration", "Entropy"])
    with pytest.raises(ValueError) as failure:
        _sage.QuantumSage(data_input=frame)
    message = str(failure.value)
    assert "'iteration'" in message and "'Entropy'" in message
    assert "ModelResults.csv" in message


class TestPredictTakesTheDocumentedColumns:
    """``predict`` must accept ``_columns_data_features`` -- what its docstring asks for."""

    @pytest.fixture(scope="class")
    def trained(self):
        sage = _sage.QuantumSage(data_input=results_table())
        # The smallest search that still fits: this is a column-handling test, not
        # a test of surrogate quality.
        sage.train_sub_sages(test_size=0.3, sage_type="random_forest", n_iter=1, cv=2)
        return sage

    def test_the_feature_columns_alone_are_enough(self, trained):
        features = results_table()[trained._columns_data_features].iloc[[0]]
        predictions = trained.predict(features, metric="accuracy")

        # Every model gets a row, but the *order* is a result, not an echo of the
        # input: rows are ranked by metric * r2, so a confident prediction from a
        # poorly-fitted surrogate cannot top the list on its point value alone.
        assert sorted(predictions["model"]) == sorted(MODELS)
        assert predictions["accuracy"].notna().all()
        assert predictions["accuracy*r2"].is_monotonic_decreasing

    def test_slgh_is_derived_not_demanded(self, trained):
        """The caller must not have to know about a column training invented."""
        assert "SLGH" not in trained._columns_data_features

        features = results_table()[trained._columns_data_features].iloc[[0]]
        # Passing it explicitly is equally fine: it is recomputed, so a stale value
        # cannot reach the estimator.
        with_stale = features.assign(SLGH=-999.0)
        assert trained.predict(with_stale, metric="accuracy").equals(
            trained.predict(features, metric="accuracy")
        )

    def test_several_rows_are_refused_rather_than_silently_reduced(self, trained):
        """The ranking is one row per model, so a multi-row input had nowhere to go.

        ``.predict(...)[0]`` ranked on whichever row sorted first and dropped the
        rest, with nothing in the output saying so. The frame that triggers it is the
        obvious one to build: complexity features are measured on the *embedded*
        data, so ``results_df[features].drop_duplicates()`` for a single dataset
        yields one row per (embedding, iteration), not one row.
        """
        several = results_table()[trained._columns_data_features].iloc[:3]
        with pytest.raises(ValueError, match="exactly one row"):
            trained.predict(several, metric="accuracy")

        # And the message says how to get to one row.
        with pytest.raises(ValueError, match=r"iloc\[\[0\]\]"):
            trained.predict(several, metric="accuracy")

    def test_a_missing_feature_names_itself_and_the_count(self, trained):
        features = results_table()[trained._columns_data_features].iloc[[0]]
        with pytest.raises(ValueError, match="Entropy"):
            trained.predict(features.drop(columns=["Entropy"]), metric="accuracy")

    def test_an_untrained_metric_is_refused_by_name(self, trained):
        features = results_table()[trained._columns_data_features].iloc[[0]]
        with pytest.raises(ValueError, match="nope"):
            trained.predict(features, metric="nope")


def test_predicting_before_training_says_to_train():
    """Used to be a bare KeyError on an empty dict, naming the metric, not the cause."""
    sage = _sage.QuantumSage(data_input=results_table())
    features = results_table()[sage._columns_data_features].iloc[[0]]
    with pytest.raises(RuntimeError, match="train_sub_sages"):
        sage.predict(features, metric="accuracy")
