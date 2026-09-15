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

"""One results table, many models -- the invariants a single-model test cannot see.

``model_run`` fans the requested models out through joblib and folds the frames
back together with ``pd.melt(...).dropna().pivot(...)``. Both halves of that fail
quietly. ``dropna()`` *discards* a model whose frame came back short rather than
raising, so a learner that stopped producing a row leaves a smaller dictionary,
not an error -- and every existing test reads one key at a time, so a lost model
looks exactly like a test that asked for less. The one multi-model call in the
suite (``model=['catboost', 'rf']`` in tests/integration) asserts only
``"results_catboost" in result``: dropping ``results_rf`` entirely leaves it green.

The other half is the schema. ``qprofiler`` writes ModelResults.csv with
``csv.writer`` in append mode and emits the header only for the first row, from
that row's own keys; every later row is written positionally. One model
contributing an extra or renamed inner key therefore produces rows wider than the
header -- a CSV that still parses, still detects as a complexity schema and still
trains a QuantumSage, with every column after the divergence point shifted by
one. Nothing was asserting that all models agree on those keys. The agreement
comes from a single ``if args['grid_search']`` branch in
:func:`qbiocode.evaluation.model_evaluation.modeleval`, so any compute function
that ever built its own frame instead would break the schema silently; the last
test here is the structural guard that keeps that construction in one place.

These tests also pin what the return value *is*, because the docstring is wrong
about it: ``model_run`` returns three keys per model label -- ``results_<label>``,
``y_predicted_<label>`` and ``y_test_<label>``, not "keys as model names" -- each
mapping to ``{0: value}`` after the pivot, and each row carries six inner keys
with ``'time'`` among them. Because the predictions are filed next to the metrics
computed from them, recomputing accuracy, F1 and AUC from
``y_predicted_<label>`` is the cheapest available guard against a metric that has
stopped describing the fit it is attached to -- a constant, a swapped pair or a
degenerate average all survive a finite-and-in-range check.

Nothing here hardcodes the model list. It is read out of ``compute_ml_dict``
itself, so a learner added to the dispatch table is covered by every invariant
below on the day it lands rather than being silently skipped.
"""

from __future__ import annotations

import ast
import inspect
from importlib import import_module
import importlib
import pathlib
import warnings

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Imported at module scope, and deliberately before anything torch-backed runs:
# qbiocode's own __init__ imports xgboost, which installs its OpenMP runtime first
# and keeps a later ``import torch`` (TabPFN) from crashing the interpreter. See
# tests/test_openmp_import_order.py.
import qbiocode
from qbiocode import learning
from qbiocode.evaluation.model_run import model_run

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_RUN_SOURCE = REPO_ROOT / "qbiocode" / "evaluation" / "model_run.py"
PACKAGE_ROOT = REPO_ROOT / "qbiocode"

#: The inner keys ``modeleval`` writes for an untuned run. Six, not five: ``'time'``
#: is there too. It is excluded from the reproducibility contract on purpose (see
#: tests/integration/conftest.py) but it is part of the *schema*, and the schema is
#: what the append-mode CSV writer is sized against.
RESULT_KEYS = frozenset(
    {"model", "accuracy", "f1_score", "time", "auc", "Model_Parameters"}
)

#: The tuned branch swaps exactly one key. Everything else must stay put, or a run
#: with grid_search on writes a differently-shaped table from one without.
TUNED_RESULT_KEYS = (RESULT_KEYS - {"Model_Parameters"}) | {"BestParams_Tuned"}

#: ``model_run`` files three columns per model label.
LABEL_PREFIXES = ("results", "y_test", "y_predicted")


# ----------------------------------------------------------------------------------
# The dispatch table, read out of model_run rather than restated here
# ----------------------------------------------------------------------------------
#
# ``compute_ml_dict`` is built from lazy imports *inside* model_run's body, so there
# is no importable object to inspect and nothing about it is checked at import time.
# The existing checks on it are string greps of the source text (test_docs_structure
# regexes the keys out; test_catboost_tabpfn asserts a literal substring is present),
# which a typo'd import or a key wired to the wrong function passes cleanly. Reading
# the table structurally and then *resolving* every entry is what closes that.


def _read_dispatch_source():
    """``({key: (function name, module)}, quantum keys)`` as written in model_run."""
    tree = ast.parse(MODEL_RUN_SOURCE.read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "model_run"]
    assert body, f"model_run is no longer a module-level function in {MODEL_RUN_SOURCE}"

    imported, table_node, quantum = {}, None, None
    for node in ast.walk(body[0]):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported[alias.asname or alias.name] = node.module
        elif isinstance(node, ast.Assign):
            targets = {getattr(t, "id", None) for t in node.targets}
            if "compute_ml_dict" in targets:
                table_node = node.value
            elif "quantum_models" in targets:
                quantum = node.value
    assert isinstance(table_node, ast.Dict), "could not read compute_ml_dict out of model_run"
    assert isinstance(quantum, ast.Set), "could not read quantum_models out of model_run"

    table = {}
    for key_node, value_node in zip(table_node.keys, table_node.values):
        assert isinstance(key_node, ast.Constant), "compute_ml_dict has a non-literal key"
        assert isinstance(value_node, ast.Name), (
            f"compute_ml_dict[{key_node.value!r}] is not a plain imported name"
        )
        table[key_node.value] = (value_node.id, imported.get(value_node.id))
    # Non-vacuity: several tests below iterate this table, and an empty one would
    # make every one of them pass by looping over nothing.
    assert len(table) > 1, "compute_ml_dict parsed as empty; the reader above is stale"
    return table, frozenset(element.value for element in quantum.elts)


DISPATCH, QUANTUM_KEYS = _read_dispatch_source()

#: The classical learners, derived: everything that is neither a tuned twin nor
#: quantum. Nine today (svc dt lr nb rf xgb catboost tabpfn mlp).
CLASSICAL_KEYS = sorted(
    k for k in DISPATCH if not k.endswith("_opt") and k not in QUANTUM_KEYS
)

#: TabPFN is excluded from the ``n_jobs > 1`` run only. Its first fit imports torch,
#: and in a fresh loky worker torch's vendored libomp claims the process-wide OpenMP
#: state and the worker dies with SIGSEGV -- no traceback, just a TerminatedWorkerError.
#: That is a property of the three vendored libomp copies on macOS, not of model_run's
#: fan-out, and the sequential run below covers TabPFN's schema anyway.
PARALLEL_KEYS = [k for k in CLASSICAL_KEYS if k != "tabpfn"]


def _labels(result):
    """The model labels present in a ``model_run`` return value."""
    return sorted(k[len("results_"):] for k in result if k.startswith("results_"))


def _row(result, label):
    """The metrics dict filed under ``results_<label>``.

    The pivot leaves ``{0: {...}}``, not a list -- indexing ``[0]`` happens to work
    for both, which is why the wrong shape has never been noticed.
    """
    return result[f"results_{label}"][0]


# ----------------------------------------------------------------------------------
# Fixtures: one model_run call per shape, shared across the assertions on it
# ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset():
    """Small, separable and split deterministically -- the same fixture shape as
    tests/test_split_reproducibility.py, so runtimes stay in the fractions of a
    second these invariants are worth."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = (X[:, 0] + 0.3 * rng.normal(size=60) > 0).astype(int)
    return train_test_split(X, y, stratify=y, test_size=0.3, random_state=9)


def _run(dataset, **overrides):
    X_train, X_test, y_train, y_test = dataset
    args = {"seed": 7, "n_jobs": 1, "grid_search": False, **overrides}
    with warnings.catch_warnings():
        # sklearn deprecation chatter and CatBoost's overfitting notes are not what
        # these tests are about; suppressed here so a real failure is readable.
        warnings.simplefilter("ignore")
        return model_run(X_train, X_test, y_train, y_test, "matrix", args)


@pytest.fixture(scope="module")
def sequential(dataset):
    """Every classical dispatch key, one call, one worker."""
    return _run(dataset, model=list(CLASSICAL_KEYS))


@pytest.fixture(scope="module")
def parallel(dataset):
    """The same models again, fanned out over four workers."""
    return _run(dataset, model=list(PARALLEL_KEYS), n_jobs=4)


#: One classical learner, one quantum learner, and QPL's fan-out -- the mixture a real
#: config produces. Hoisted out of the fixture because the label assertions need the
#: requested dispatch keys, and a second hardcoded copy could drift from the first.
ONE_OF_EACH_FAMILY_KEYS = ("svc", "qsvc", "qpl")


@pytest.fixture(scope="module")
def quantum_dataset():
    """Scaled to the unit interval, because a feature map encodes magnitudes as
    rotation angles. Same construction as tests/test_quantum_tuning.py's fixture."""
    from qbiocode import scale_train_test

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test = scale_train_test(X[:30], X[30:], scaling="MinMaxScaler")
    return X_train, X_test, y[:30], y[30:]


@pytest.fixture(scope="module")
def one_of_each_family(quantum_dataset, tmp_path_factory):
    """A classical, a quantum and QPL's fan-out, in a single model_run call.

    Mixing the families in *one* call is the point: whether the two branches of
    ``modeleval`` agree is only interesting across the seam, and a table
    concatenated from a real config contains exactly this mixture. The projection
    directories point into tmp_path so no cache lands in the repository root --
    PQK's and QPL's defaults are CWD-relative.
    """
    tmp = tmp_path_factory.mktemp("projections")
    X_train, X_test, y_train, y_test = quantum_dataset
    args = {
        "model": list(ONE_OF_EACH_FAMILY_KEYS),
        "seed": 7,
        "q_seed": 7,
        "n_jobs": 1,
        "grid_search": False,
        "backend": "simulator",
        "shots": 64,
        "pqk_projection_dir": str(tmp / "pqk"),
        "qpl_projection_dir": str(tmp / "qpl"),
        # One head keeps this to a couple of seconds; the default six fit a
        # RandomizedSearchCV each.
        "qpl_args": {"classical_models": ["lr"]},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model_run(X_train, X_test, y_train, y_test, "matrixq", args)


# ----------------------------------------------------------------------------------
# The table itself
# ----------------------------------------------------------------------------------


class TestTheDispatchTable:
    """28 entries wired by hand, none of them checked at import time."""

    def test_every_entry_resolves_to_the_function_its_key_names(self):
        """A typo'd import or a mis-wired key is otherwise a KeyError in a worker.

        Both halves matter. Importing the module and fetching the name proves the
        entry is a real callable rather than a name that happens to appear in the
        file; comparing that name against the key proves ``"nb": compute_svc`` --
        which would run the wrong learner under an honest-looking label -- cannot
        pass. ``modeleval`` interpolates the label into the column name, so a
        mis-wired entry produces a plausible results table, not a crash.
        """
        broken = {}
        for key, (name, module) in sorted(DISPATCH.items()):
            if module is None:
                broken[key] = f"{name} is not imported anywhere in model_run"
                continue
            try:
                resolved = getattr(importlib.import_module(module), name)
            except (ImportError, AttributeError) as exc:
                broken[key] = f"{module}.{name} does not resolve: {exc}"
                continue
            if not callable(resolved):
                broken[key] = f"{module}.{name} is not callable"
            elif resolved.__name__ != f"compute_{key}":
                broken[key] = (
                    f"dispatch key {key!r} runs {resolved.__name__}, not compute_{key}"
                )
        assert not broken, f"compute_ml_dict entries that do not hold up: {broken}"

    def test_every_learner_has_both_a_base_and_a_tuned_entry(self):
        """The invariant that keeps model_run's ``missing_opt`` refusal dead code.

        ``model_run`` refuses ``grid_search`` for a classical model with no ``_opt``
        twin, but that branch is unreachable today: an unknown name is rejected
        earlier, and every base key here has a twin. It is reachable the moment a
        learner is added with a ``compute_<m>`` entry and no ``compute_<m>_opt`` --
        such a model would be accepted by ``model: [m]`` and then refused the
        instant anyone turned tuning on. Asserting the pairing is what makes the
        dead branch legitimately dead, and it fails here first, where the cause is
        named, rather than in a config months later.
        """
        base = {k for k in DISPATCH if not k.endswith("_opt")}
        tuned = {k[: -len("_opt")] for k in DISPATCH if k.endswith("_opt")}
        assert base == tuned, (
            f"learners with no '_opt' twin: {sorted(base - tuned)}; "
            f"'_opt' twins with no base entry: {sorted(tuned - base)}"
        )
        assert len(DISPATCH) == 2 * len(base) == 28, (
            f"compute_ml_dict holds {len(DISPATCH)} entries for {len(base)} learners; "
            f"the documented surface is 28 entries for 14 learners"
        )

    @pytest.mark.parametrize(
        "requested", [pytest.param([], id="empty"), pytest.param(["svm"], id="unknown")]
    )
    def test_both_refusals_advertise_every_name_the_table_holds(self, dataset, requested):
        """The menu in the message is where a user finds the name they meant.

        This is also the bridge between the source text read above and the
        dictionary that actually exists at runtime: the two messages interpolate
        ``sorted(compute_ml_dict)``, so a table that gained or lost an entry
        without the source literal changing shows up here. The existing check on
        these errors (tests/test_error_contracts.py) matches only the offending
        name, so the menu itself could be dropped from either message silently.
        """
        X_train, X_test, y_train, y_test = dataset
        with pytest.raises(ValueError) as excinfo:
            model_run(
                X_train, X_test, y_train, y_test, "matrix",
                {"model": requested, "seed": 7, "n_jobs": 1, "grid_search": False},
            )
        message = str(excinfo.value)
        missing = sorted(key for key in DISPATCH if repr(key) not in message)
        assert not missing, (
            f"the refusal for model={requested!r} does not name {missing}, so a user "
            f"cannot read the valid options off it: {message}"
        )


class TestTheExportSurface:
    """A dispatchable learner a user cannot import is documented but unreachable."""

    def test_every_dispatchable_learner_is_exported_from_the_learning_subpackage(self):
        """``qbiocode.learning`` is where the docs' API table points."""
        missing = sorted(
            f"compute_{key}"
            for key in DISPATCH
            if not callable(getattr(learning, f"compute_{key}", None))
            or f"compute_{key}" not in learning.__all__
        )
        assert not missing, (
            f"dispatchable but not exported from qbiocode.learning: {missing}"
        )

    def test_every_dispatchable_learner_is_reachable_from_the_package_root(self):
        """``from qbiocode import compute_<model>`` is what every example uses.

        The convention the package sets for itself is both twins at the root: all
        nine classical learners export ``compute_<m>`` *and* ``compute_<m>_opt``
        there. The quantum half was not updated alongside the dispatch table, so
        names the docs present as first-class -- ``qpl`` is in the model table
        tests/test_docs_structure.py polices, and in profiler.rst -- raise
        ImportError. tests/integration/test_package_surface.py cannot see it: it
        checks that every name in ``__all__`` resolves, which says nothing about a
        name that never got added.
        """
        missing = sorted(
            f"compute_{key}"
            for key in DISPATCH
            if not callable(getattr(qbiocode, f"compute_{key}", None))
            or f"compute_{key}" not in qbiocode.__all__
        )
        assert not missing, (
            f"dispatchable through model_run but absent from the package root: "
            f"{missing}. Add them to the imports and __all__ in qbiocode/__init__.py."
        )


# ----------------------------------------------------------------------------------
# One call, many models
# ----------------------------------------------------------------------------------


class TestTheFanOut:
    """What comes back for N models, stated as an equality rather than a lookup."""

    def test_each_requested_model_contributes_exactly_three_columns(self, sequential):
        """Equality, because ``dropna()`` turns a lost model into a shorter dict.

        Asserting containment -- which is all the suite did -- cannot see a model
        that vanished. Asserting the whole set also pins the documented return
        contract, which the docstring gets wrong: three columns per label, not one.
        """
        expected = {
            f"{prefix}_{key}" for key in CLASSICAL_KEYS for prefix in LABEL_PREFIXES
        }
        assert set(sequential) == expected

    def test_the_label_in_the_column_name_is_the_label_inside_the_row(
        self, sequential, one_of_each_family
    ):
        """The label is the DISPATCH KEY the caller asked for, not merely self-consistent.

        Every compute function defaults ``model`` to a display name -- 'Decision Tree',
        'Naive Bayes', 'Multi-layer Perceptron' -- and only ``model_run``'s
        ``model=method`` keyword overrides it. Drop that override and ``modeleval`` builds
        ``results_Decision Tree``, so a reader grouping ``ModelResults.csv`` by the model
        column finds a name no config ever named.

        Asserting ``row['model'] == label`` alone cannot catch that, and this test used to
        do only that. ``modeleval`` builds the column name and the row's ``model`` field
        from the *same* argument (model_evaluation.py, ``"y_test_" + model`` and
        ``"model": model``), so the two agree by construction whatever that argument is --
        including 'Decision Tree'. The self-consistency is worth keeping as a cheap
        invariant, but the assertion with something to lose is against the requested key.
        """
        for result, requested in (
            (sequential, CLASSICAL_KEYS),
            (one_of_each_family, ONE_OF_EACH_FAMILY_KEYS),
        ):
            labels = set(_labels(result))
            # The dispatch key itself must appear as a label. QPL is the one learner that
            # fans out (one column per classical head, 'qpl_<head>'), so it is matched by
            # prefix; every other key must be present verbatim.
            for key in requested:
                if key == "qpl":
                    assert any(label.startswith("qpl_") for label in labels), (
                        f"requested {key!r} but no qpl_<head> column came back: {sorted(labels)}"
                    )
                else:
                    assert key in labels, (
                        f"requested dispatch key {key!r} but the results columns are "
                        f"{sorted(labels)} -- if one of those is a display name like "
                        "'Decision Tree', model_run stopped passing model=method"
                    )
            # And the cheap invariant, which a half-dropped override would still break.
            for label in labels:
                assert _row(result, label)["model"] == label

    def test_the_columns_do_not_depend_on_how_many_workers_ran(self, parallel):
        """The joblib fan-out is the seam where a model goes missing unnoticed."""
        expected = {
            f"{prefix}_{key}" for key in PARALLEL_KEYS for prefix in LABEL_PREFIXES
        }
        assert set(parallel) == expected

    def test_the_answer_does_not_depend_on_how_many_workers_ran(self, sequential, parallel):
        """Four workers must not change a number, only the wall clock.

        loky starts fresh interpreters, seeded from OS entropy, and batches tasks
        in an order that depends on timing -- which is why ``_call_with_global_seeds``
        re-establishes the seeds inside the worker and ``_seeded_kwargs`` sets
        ``random_state`` explicitly. Nothing was comparing a fanned-out run against
        a sequential one, so a result that quietly depended on ``n_jobs`` would have
        looked like ordinary run-to-run noise.
        """
        for key in PARALLEL_KEYS:
            sequential_row, parallel_row = _row(sequential, key), _row(parallel, key)
            for metric in ("accuracy", "f1_score", "auc"):
                assert sequential_row[metric] == parallel_row[metric], (
                    f"{key} reported {metric}={parallel_row[metric]!r} at n_jobs=4 but "
                    f"{sequential_row[metric]!r} at n_jobs=1"
                )
            assert np.array_equal(
                np.asarray(parallel[f"y_predicted_{key}"][0]),
                np.asarray(sequential[f"y_predicted_{key}"][0]),
            ), f"{key} predicted different labels under the joblib fan-out"

    def test_qpl_is_the_one_model_that_fans_out_into_several_columns(
        self, one_of_each_family
    ):
        """'One results key per requested model' is false for exactly one model.

        ``compute_qpl`` concatenates one ``modeleval`` frame per classical head, so
        ``model: [qpl]`` produces ``results_qpl_<head>`` -- six columns at the
        default head list -- and no bare ``results_qpl`` at all. Every consumer
        that assumes one column per requested model is wrong for it:
        ``_metric_dicts`` exists to cope, and its docstring records that a helper
        which assumed otherwise 'scored only the first head'. qprofiler writes one
        CSV row per results column, so naming qpl multiplies the row count.
        """
        assert "results_qpl" not in one_of_each_family, (
            "results_qpl appeared; the head fan-out is what downstream readers and "
            "the row-count invariants are written against"
        )
        assert "results_qpl_lr" in one_of_each_family, sorted(one_of_each_family)


class TestNoResultsTableComesOutRagged:
    """The property qprofiler's append-mode CSV writer silently depends on."""

    def test_every_classical_model_reports_the_same_result_keys(self, sequential):
        """One header, written from the first row, used for all of them."""
        shapes = {label: frozenset(_row(sequential, label)) for label in _labels(sequential)}
        assert set(shapes.values()) == {RESULT_KEYS}, (
            f"models disagree on their result keys: "
            f"{ {label: sorted(keys) for label, keys in shapes.items()} }"
        )

    def test_a_quantum_model_reports_the_same_keys_as_a_classical_one(
        self, one_of_each_family
    ):
        """The seam that a single-family test can never reach.

        The quantum learners take an entirely different route to ``modeleval`` --
        no OneVsOne wrapper, a projection cache, and for QPL a concatenation of
        per-head frames -- so they are the likeliest to drift from the schema, and
        they land in the same ModelResults.csv as the classical rows.
        """
        shapes = {
            label: frozenset(_row(one_of_each_family, label))
            for label in _labels(one_of_each_family)
        }
        assert set(shapes.values()) == {RESULT_KEYS}, (
            f"a family disagrees on its result keys: "
            f"{ {label: sorted(keys) for label, keys in shapes.items()} }"
        )

    def test_no_untuned_row_claims_it_was_tuned(self, sequential):
        """``BestParams_Tuned`` on a grid_search-off row would mean the branch in
        ``modeleval`` picked wrong -- and QuantumSage and qc_winner_finder both
        *prefer* that column over ``Model_Parameters`` when it is present."""
        for label in _labels(sequential):
            assert "BestParams_Tuned" not in _row(sequential, label)

    def test_the_tuned_branch_swaps_one_key_and_leaves_the_rest_alone(self, dataset):
        """Same table, one column renamed -- otherwise the two runs cannot be concatenated.

        A tuned run and an untuned run of the same models end up in the same
        results file often enough (the documented restart workflow appends one to
        the other), so the tuned schema is not free to differ by more than the
        parameter column. The ``_opt`` label is asserted alongside because it is
        the only thing that distinguishes the two rows once they are in one table.
        """
        result = _run(
            dataset,
            model=["dt", "rf"],
            grid_search=True,
            tuner="optuna",
            n_trials=2,
            cross_validation=2,
            gridsearch_dt_args={"max_depth": [2, 3]},
            gridsearch_rf_args={"n_estimators": [5, 7]},
        )
        assert _labels(result) == ["dt_opt", "rf_opt"]
        shapes = {label: frozenset(_row(result, label)) for label in _labels(result)}
        assert set(shapes.values()) == {TUNED_RESULT_KEYS}, (
            f"the tuned rows are not the untuned schema with one key swapped: "
            f"{ {label: sorted(keys) for label, keys in shapes.items()} }"
        )

    def test_only_model_evaluation_builds_a_results_column(self):
        """Uniformity holds because exactly one function names those columns.

        Every assertion above is a consequence of that. A compute function that
        assembled its own frame -- to add a column, or to report two rows -- would
        break the schema for everything downstream while its own tests passed, so
        the guard is on the construction rather than on the outcome.
        """
        offenders = {}
        for path in sorted(PACKAGE_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                pieces = []
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    pieces = [node.left, node.right]
                elif isinstance(node, ast.JoinedStr):
                    pieces = list(node.values)
                for piece in pieces:
                    if (
                        isinstance(piece, ast.Constant)
                        and isinstance(piece.value, str)
                        and piece.value.startswith("results_")
                    ):
                        offenders.setdefault(
                            str(path.relative_to(REPO_ROOT)), []
                        ).append(node.lineno)
        assert set(offenders) == {"qbiocode/evaluation/model_evaluation.py"}, (
            f"a 'results_<model>' column is built outside modeleval: {offenders}. "
            f"That is how a results table becomes ragged."
        )


class TestTheReturnValueIsSelfConsistent:
    """The metrics and the predictions they were computed from travel together."""

    def test_every_label_carries_the_truth_and_the_predictions_it_was_scored_on(
        self, sequential, dataset
    ):
        """Two thirds of the return value had no test at all.

        Nothing in the suite read ``y_test_<label>`` or ``y_predicted_<label>``, so
        their presence, length and content were unpinned -- and the next test needs
        them to be the real arrays for its recomputation to mean anything.
        """
        y_test = dataset[3]
        for label in _labels(sequential):
            for prefix in LABEL_PREFIXES:
                cell = sequential[f"{prefix}_{label}"]
                # The pivot indexes by a single constant row; ``[0]`` is the value,
                # and this is where the "it is a list" reading of the contract dies.
                assert set(cell) == {0}, f"{prefix}_{label} is shaped {cell!r}"
            recorded = np.asarray(sequential[f"y_test_{label}"][0])
            predicted = np.asarray(sequential[f"y_predicted_{label}"][0])
            assert np.array_equal(recorded, np.asarray(y_test)), (
                f"y_test_{label} is not the y_test handed to model_run"
            )
            assert predicted.shape == recorded.shape, (
                f"{label} predicted {predicted.shape} labels for {recorded.shape} rows"
            )

    @pytest.mark.parametrize("fixture_name", ["sequential", "one_of_each_family"])
    def test_the_recorded_metrics_are_the_metrics_of_the_recorded_predictions(
        self, fixture_name, request
    ):
        """The cheapest guard there is against a metric that stopped measuring the fit.

        A finite-and-in-range check -- the natural thing to write -- passes for a
        constant, for a swapped pair, and for the 0.0 that
        ``f1_score(average='weighted')`` returns *with a warning rather than an
        exception* when a model collapses to one predicted class. Recomputing from
        the predictions filed alongside the row does not.

        ``auc`` is the one column this test cannot reproduce, and that is the point of
        the fix that made it so. It used to be ``roc_auc_score`` of the hard predicted
        labels -- balanced accuracy under an AUC's name -- and was reproduced here from
        ``y_predicted`` for exactly that reason. It is now the AUC of the estimator's own
        ``predict_proba``/``decision_function`` scores, which are not recorded in the
        frame, so there is nothing here to recompute it from; what is asserted instead is
        that it is a real number in range. The identity is pinned where the scores are
        reachable, in ``tests/test_classical_models.py::TestTheAucColumnIsARankingAuc``.
        """
        result = request.getfixturevalue(fixture_name)
        for label in _labels(result):
            row = _row(result, label)
            y_true = np.asarray(result[f"y_test_{label}"][0])
            y_pred = np.asarray(result[f"y_predicted_{label}"][0])
            assert row["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred)), (
                f"{label} recorded accuracy={row['accuracy']!r}, which is not the "
                f"accuracy of the predictions recorded next to it"
            )
            assert row["f1_score"] == pytest.approx(
                f1_score(y_true, y_pred, average="weighted")
            ), f"{label} recorded f1_score={row['f1_score']!r}"
            assert np.isfinite(row["auc"]) and 0.0 <= row["auc"] <= 1.0, (
                f"{label} recorded auc={row['auc']!r}. NaN means modeleval was handed "
                f"no y_score for this learner -- either its call site dropped the "
                f"keyword or its estimator stopped offering predict_proba and "
                f"decision_function."
            )
            # No "and it differs from roc_auc_score(y_true, y_pred)" assertion here:
            # a learner whose scores are genuinely two-valued -- ``dt``, grown to pure
            # leaves -- coincides with the old label-based number honestly, and telling
            # the two cases apart needs the score array, which only the file named above
            # can reach.


class TestTheQuantumLabelDefaultsMatchTheDispatchKeys:
    """A direct call must file its results under a name a config could actually name.

    All ten of these defaults were upper case -- ``model="QSVC"``, ``model="PQK"`` and so
    on, one per base function and one per ``_opt`` twin. That was invisible for as long as
    the label was either hardcoded in the body or always supplied by ``model_run``, and it
    became load-bearing in two ways at once:

    * ``compute_pqk`` and ``compute_qpl`` used to discard the label entirely (they rebound
      ``model`` to the fitted estimator), so a tuned run was indistinguishable from an
      untuned one. Fixing that made the default reachable -- and an upper-case default
      would then have produced ``results_PQK``, a column no ``model:`` list can name.
    * ``modeleval`` infers whether a row was tuned from this very string (it ends in
      ``_opt`` or it does not). A wrong default therefore mislabels the row *and* files its
      hyperparameters under the wrong column. That is not hypothetical: leaving the
      ``_opt`` twins defaulting to the bare name broke six tests in
      ``tests/test_quantum_tuning.py`` with ``KeyError: 'BestParams_Tuned'``, because they
      call ``compute_<m>_opt`` directly and rely on the default.

    ``qc_winner_finder`` was written against the upper-case spelling while every real
    results table carried the lower-case one, which is how its quantum branch came to be
    dead on all genuine output -- so this is the spelling that has to stay pinned.
    """

    QUANTUM = ("qsvc", "vqc", "qnn", "pqk", "qpl")

    @pytest.mark.parametrize("name", QUANTUM)
    def test_the_base_default_is_the_bare_dispatch_key(self, name):
        module = import_module(f"qbiocode.learning.compute_{name}")
        default = inspect.signature(getattr(module, f"compute_{name}")).parameters["model"].default
        assert default == name, (
            f"compute_{name}'s model default is {default!r}; it must be {name!r} -- the "
            "dispatch key exactly, so a direct call and a dispatched one agree and "
            "qc_winner_finder's family match still sees it"
        )

    @pytest.mark.parametrize("name", QUANTUM)
    def test_the_opt_default_carries_the_opt_marker(self, name):
        module = import_module(f"qbiocode.learning.compute_{name}")
        default = inspect.signature(
            getattr(module, f"compute_{name}_opt")
        ).parameters["model"].default
        assert default == f"{name}_opt", (
            f"compute_{name}_opt's model default is {default!r}; it must be "
            f"{name + '_opt'!r}. modeleval decides the parameter column from this string, "
            "so a bare default would report a tuned run's hyperparameters under "
            "'Model_Parameters' as though no search had run"
        )

    @pytest.mark.parametrize("name", QUANTUM)
    def test_no_default_is_upper_case(self, name):
        """The specific regression: these were 'QSVC', 'VQC', 'QNN', 'PQK', 'QPL'."""
        module = import_module(f"qbiocode.learning.compute_{name}")
        for suffix in ("", "_opt"):
            fn = getattr(module, f"compute_{name}{suffix}")
            default = inspect.signature(fn).parameters["model"].default
            assert default == default.lower(), (
                f"compute_{name}{suffix} defaults to {default!r}; an upper-case label is "
                "what made qc_winner_finder's quantum list unmatchable on real output"
            )
