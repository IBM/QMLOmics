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

"""``ModelResults.csv`` stays rectangular when models disagree about their columns.

QProfiler writes one row per model as each model finishes, so a run that dies half
way still leaves usable output. Doing that with ``csv.writer`` and a header written
when the file is empty holds only while every model contributes the same keys, and
two of QProfiler's do not: a **tuned** model reports ``BestParams_Tuned`` and an
**untuned** one reports ``Model_Parameters``. ``grid_search: True`` with
``tune_quantum: False`` -- the ordinary way to sweep, because a quantum fit per
Optuna trial is expensive -- puts both in one run.

The file that came out had a 150-column header over 151-column rows, which
``pandas.read_csv`` refuses::

    ParserError: Expected 150 fields in line 7, saw 151

so no consumer could read the results at all -- not QSage, not
``compute_results_correlation``, not a notebook. And because the writer merged each
model's keys into one shared dict, values leaked between rows: the ``pqk`` row
carried the *preceding* model's ``BestParams_Tuned``, reporting a naive-Bayes
``var_smoothing`` as pqk's tuned hyperparameters.

Both are pinned here. The raggedness is the loud failure; the leak is the dangerous
one, because it produced a readable file with wrong values in it.
"""

from __future__ import annotations

import csv

import pandas as pd
import pytest

from qbiocode.apps.qprofiler.qprofiler import _append_model_row


def _read(path):
    with open(path, newline="") as handle:
        return list(csv.reader(handle))


class TestTheHappyPath:
    """Rows that agree about their columns append as they always did."""

    def test_the_first_row_writes_a_header(self, tmp_path):
        target = tmp_path / "ModelResults.csv"
        _append_model_row(str(target), {"model": "lr", "accuracy": 0.9})
        assert _read(target) == [["model", "accuracy"], ["lr", "0.9"]]

    def test_later_rows_append_without_repeating_it(self, tmp_path):
        target = tmp_path / "ModelResults.csv"
        _append_model_row(str(target), {"model": "lr", "accuracy": 0.9})
        _append_model_row(str(target), {"model": "dt", "accuracy": 0.8})
        rows = _read(target)
        assert rows[0] == ["model", "accuracy"]
        assert len(rows) == 3

    def test_an_empty_file_is_treated_as_absent(self, tmp_path):
        """QProfiler opens in append mode, so a zero-byte file is a real state."""
        target = tmp_path / "ModelResults.csv"
        target.touch()
        _append_model_row(str(target), {"model": "lr", "accuracy": 0.9})
        assert _read(target)[0] == ["model", "accuracy"]


class TestColumnSetsThatDisagree:
    """The regression. Neither direction may shift a value into the wrong column."""

    def test_a_new_column_widens_the_file_and_pads_earlier_rows(self, tmp_path):
        """The tuned-then-untuned order: ``Model_Parameters`` arrives after the header."""
        target = tmp_path / "ModelResults.csv"
        _append_model_row(str(target), {"model": "nb", "BestParams_Tuned": "{'vs': 1e-8}"})
        _append_model_row(str(target), {"model": "pqk", "Model_Parameters": "{'reps': 4}"})

        frame = pd.read_csv(target)  # this is the call that used to raise ParserError
        assert list(frame.columns) == ["model", "BestParams_Tuned", "Model_Parameters"]
        assert len(frame) == 2
        # Each row carries its own parameter column and is blank in the other.
        nb, pqk = frame.iloc[0], frame.iloc[1]
        assert nb["BestParams_Tuned"] == "{'vs': 1e-8}" and pd.isna(nb["Model_Parameters"])
        assert pqk["Model_Parameters"] == "{'reps': 4}" and pd.isna(pqk["BestParams_Tuned"])

    def test_a_missing_column_writes_a_blank_rather_than_shifting_left(self, tmp_path):
        """The untuned-then-tuned order, and the subtler half of the bug.

        A plain ``csv.writer`` emits a row's values positionally. If the row lacks a
        column the header has, every value after the gap lands one column early --
        producing a *readable* file with an accuracy in the f1_score column. Nothing
        would flag that.
        """
        target = tmp_path / "ModelResults.csv"
        _append_model_row(
            str(target), {"model": "pqk", "Model_Parameters": "{'reps': 4}", "auc": 0.7}
        )
        _append_model_row(str(target), {"model": "nb", "auc": 0.8})

        frame = pd.read_csv(target)
        assert list(frame.columns) == ["model", "Model_Parameters", "auc"]
        assert pd.isna(frame.iloc[1]["Model_Parameters"])
        assert frame.iloc[1]["auc"] == 0.8, "value shifted into the wrong column"

    def test_the_file_stays_rectangular_however_the_key_sets_interleave(self, tmp_path):
        target = tmp_path / "ModelResults.csv"
        for row in (
            {"model": "lr", "BestParams_Tuned": "a"},
            {"model": "pqk", "Model_Parameters": "b"},
            {"model": "dt", "BestParams_Tuned": "c"},
            {"model": "qsvc", "Model_Parameters": "d", "extra": 1},
            {"model": "rf", "BestParams_Tuned": "e"},
        ):
            _append_model_row(str(target), row)

        rows = _read(target)
        widths = {len(r) for r in rows}
        assert len(widths) == 1, f"ragged file, row widths {widths}"
        frame = pd.read_csv(target)
        assert len(frame) == 5
        assert list(frame["model"]) == ["lr", "pqk", "dt", "qsvc", "rf"]

    def test_widening_preserves_every_earlier_value(self, tmp_path):
        """Rewriting the file to widen it must not lose or reorder what is already there."""
        target = tmp_path / "ModelResults.csv"
        _append_model_row(str(target), {"model": "lr", "accuracy": 0.91, "auc": 0.95})
        _append_model_row(str(target), {"model": "dt", "accuracy": 0.82, "auc": 0.88})
        _append_model_row(str(target), {"model": "pqk", "accuracy": 0.75, "auc": 0.80,
                                        "Model_Parameters": "{'reps': 4}"})
        frame = pd.read_csv(target)
        assert list(frame["accuracy"]) == [0.91, 0.82, 0.75]
        assert list(frame["auc"]) == [0.95, 0.88, 0.80]


class TestValuesDoNotLeakBetweenRows:
    """Each row is one model's observation, so building rows independently is required.

    The writer used to merge each model's results into a single dict reused across
    models, so a column one model reported persisted into every later row that did
    not report it. Under ``grid_search: True`` with ``tune_quantum: False`` that put a
    naive-Bayes ``var_smoothing`` in pqk's ``BestParams_Tuned`` cell -- a wrong value
    in a readable file, attributed to the wrong model.
    """

    def test_a_column_from_an_earlier_row_does_not_reappear(self, tmp_path):
        target = tmp_path / "ModelResults.csv"
        shared = {"Dataset": "d.csv", "iteration": 1, "mfe.nr_inst": 80}
        _append_model_row(str(target), {**shared, "model": "nb",
                                        "BestParams_Tuned": "{'var_smoothing': 1e-08}"})
        _append_model_row(str(target), {**shared, "model": "pqk",
                                        "Model_Parameters": "{'reps': 4}"})
        frame = pd.read_csv(target)
        pqk = frame[frame["model"] == "pqk"].iloc[0]
        assert pd.isna(pqk["BestParams_Tuned"]), (
            "pqk inherited the previous model's tuned hyperparameters -- the shared-dict "
            "leak is back"
        )

    def test_the_shared_columns_still_repeat_on_every_row(self, tmp_path):
        """Only the per-model columns are independent; the complexity block is shared."""
        target = tmp_path / "ModelResults.csv"
        shared = {"Dataset": "d.csv", "iteration": 1, "task.graph_dirichlet": 0.25}
        for model in ("nb", "pqk", "rf"):
            _append_model_row(str(target), {**shared, "model": model})
        frame = pd.read_csv(target)
        assert (frame["task.graph_dirichlet"] == 0.25).all()
        assert (frame["Dataset"] == "d.csv").all()


class TestTheCallerContract:
    """What ``main`` relies on, spelled out so a refactor cannot quietly change it."""

    def test_the_row_order_on_disk_is_the_write_order(self, tmp_path):
        target = tmp_path / "ModelResults.csv"
        for i in range(6):
            _append_model_row(str(target), {"model": f"m{i}", "n": i})
        assert list(pd.read_csv(target)["n"]) == list(range(6))

    @pytest.mark.parametrize("value", ["a,b", 'say "hi"', "line\nbreak"])
    def test_values_needing_quoting_survive_a_round_trip(self, value):
        """``Model_Parameters`` is a dict repr, so it contains commas as a matter of course."""
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "ModelResults.csv"
            _append_model_row(str(target), {"model": "lr", "Model_Parameters": value})
            _append_model_row(str(target), {"model": "dt", "Model_Parameters": "plain"})
            frame = pd.read_csv(target)
            assert frame.iloc[0]["Model_Parameters"] == value
