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

"""The five quantum models on their default path: a real fit, and the row it produces.

Nothing in this suite used to fit a quantum model at its configured hyperparameters --
which is the path every ordinary config takes, because ``grid_search`` is off by
default. ``compute_vqc`` and ``compute_qnn`` were never called at all outside their
``_opt`` wrappers; ``compute_qsvc`` was only ever handed to ``run_function_study`` as a
function object; ``compute_pqk`` and ``compute_qpl`` were called, but only their cache
files on disk were ever asserted on. So the untuned branch of ``modeleval`` -- the one
that writes ``Model_Parameters`` rather than ``BestParams_Tuned`` -- had no quantum
coverage whatsoever.

That branch *is* the ModelResults.csv schema. A quantum model that stopped reporting
``auc``, or renamed ``Model_Parameters``, or (for PQK) started answering to a different
results label, would have left every test green while every results file quietly lost a
column, because the only assertions near this code find the ``results_`` columns by
prefix instead of naming them. The same goes for QPL's fan-out: it is the single
exception to "one results column per requested model", and no test said so.

Two traps are pinned here as well, because both are easy to hit and neither is obvious
from the config:

* ``shots`` is required by QSVC, VQC and QNN and ignored by PQK and QPL. The first three
  ask ``get_backend_session`` for the *sampler* primitive, which needs a shot count; the
  other two ask for the *estimator*, which does not. Omitting it therefore fails for
  three models out of five, and only at the moment the backend is opened.
* ``args['seed']`` and ``args['q_seed']`` are the whole reproducibility story for these
  models -- none of the five ``compute_q*`` functions takes a ``random_state``, so
  ``_seeded_kwargs`` is a no-op for them and the global RNG is all there is. One half of
  that is currently broken; see ``TestTheSeedReachesTheQuantumStack``.

Everything here runs on the statevector simulator, on 24 rows and 2 features, and is
firmly in the default tier. ``requires_quantum`` is for tests that need a real device
and credentials, which none of these do.
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pytest

# Imported at module scope, deliberately: `qbiocode` orders the OpenMP runtimes (see
# tests/test_openmp_import_order.py), and tests/test_suite_hygiene.py forbids reaching
# first-party or base-requirement modules through pytest.importorskip.
import qbiocode  # noqa: F401
from qbiocode import scale_train_test
from qbiocode.evaluation.model_run import _call_with_global_seeds, model_run

# QSVC's kernel warnings and the QPL head chatter are noise here, not the subject.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

#: The keys ``modeleval`` puts in a row when ``grid_search`` is off. Six, not five:
#: ``time`` is a wall clock and no reproducibility test compares it, but it is part of
#: the dict every downstream reader parses, so a rename has to fail somewhere.
DOCUMENTED_KEYS = frozenset({"model", "accuracy", "f1_score", "time", "auc", "Model_Parameters"})

#: The three scores QuantumSage and qc_winner_finder select on.
METRICS = ("accuracy", "f1_score", "auc")

#: Two cheap heads. QPL fits one classical model per name on the same projection, and
#: `rf` alone costs more than the whole rest of this file (a 40-candidate randomized
#: search), so the default list is deliberately not used -- two heads are enough to
#: prove the fan-out.
QPL_HEADS = ("lr", "svc")

#: Dispatch key -> the results labels ``model_run`` must produce for it. QPL is the one
#: entry with more than one, and it is spelt ``qpl_<head>`` rather than ``qpl``.
EXPECTED_LABELS = {
    "pqk": ("pqk",),
    "qnn": ("qnn",),
    "qpl": tuple(f"qpl_{head}" for head in QPL_HEADS),
    "qsvc": ("qsvc",),
    "vqc": ("vqc",),
}

#: These ask ``get_backend_session`` for the sampler primitive, so ``args['shots']`` is
#: required. PQK and QPL ask for the estimator and never read it.
SAMPLER_MODELS = ("qnn", "qsvc", "vqc")

#: ``(model, per-model args)`` for the two models that project onto a feature map and
#: then fit a classical head. Both use the estimator primitive.
PROJECTION_MODELS = (("pqk", {}), ("qpl", {"classical_models": ["lr"]}))


@pytest.fixture(scope="module")
def data():
    """24 rows, 2 features, split 18/6 and MinMax-scaled on the training rows only.

    Scaling is not cosmetic: the feature maps encode magnitudes as rotation angles, so
    unscaled features land anywhere on the circle and all three sampler models collapse
    to chance -- at which point a finite-and-in-range assertion cannot tell a working
    fit from a broken one. Fitting the scaler on the training rows only mirrors
    ``scale_train_test``'s own protocol; a fixture is example code.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test = scale_train_test(X[:18], X[18:], scaling="MinMaxScaler")
    return X_train, X_test, y[:18], y[18:]


@pytest.fixture(scope="module")
def sim_args(tmp_path_factory):
    """The keys the quantum stack reads, with both projection caches under tmp_path.

    ``compute_pqk`` and ``compute_qpl`` default their projection directories to
    CWD-relative ``pqk_projections``/``qpl_projections``. A test that accepted the
    default would write into the repository root and -- because the PQK cache key covers
    the feature map but not the row count -- could later be *served* by a stale file left
    there by an unrelated run.
    """
    projections = tmp_path_factory.mktemp("projections")
    return {
        "backend": "simulator",
        "shots": 64,
        "seed": 7,
        "q_seed": 7,
        "n_jobs": 1,
        "grid_search": False,
        "pqk_projection_dir": str(projections / "pqk"),
        "qpl_projection_dir": str(projections / "qpl"),
    }


@pytest.fixture(scope="module")
def fit_once(data, sim_args):
    """``model_run(model=[m])`` per quantum model, computed on first use and reused.

    Returns a callable rather than a dict of results so that a model which fails to fit
    breaks only its own parametrizations: a module-scoped dict would turn one broken
    learner into an error on every test in the file and hide which one it was.
    """
    X_train, X_test, y_train, y_test = data
    cache: dict[str, dict] = {}

    def run(model):
        if model not in cache:
            args = {**sim_args, "model": [model]}
            if model == "qpl":
                args["qpl_args"] = {"classical_models": list(QPL_HEADS)}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cache[model] = model_run(X_train, X_test, y_train, y_test, "quantum-contract", args)
        return cache[model]

    return run


def _row(out, label):
    """The metrics dict for one results label.

    ``model_run`` ends in ``DataFrame.to_dict()``, so the value under a ``results_``
    column is ``{0: {...}}`` -- a dict keyed by the row index, not a list. ``[0]``
    reads the same either way, which is why the docstrings say "list", but ``len()``
    and iteration do not.
    """
    return out[f"results_{label}"][0]


class TestEveryQuantumModelFitsAndReportsTheSameRow:
    """One real fit per model, then the shape of what came back."""

    @pytest.mark.parametrize("model,labels", sorted(EXPECTED_LABELS.items()))
    def test_the_results_columns_are_the_ones_the_readers_look_for(self, model, labels, fit_once):
        """Equality, not containment: an extra or a missing column both matter.

        qprofiler writes one ModelResults.csv row per ``results_`` column, so QPL's
        fan-out multiplies the row count for a single named model -- and there is no
        ``results_qpl`` column at all, which the documented "keyed results_<model>"
        contract would lead a reader to expect. PQK is the opposite trap: it hardcodes
        its own label, so it is the one model whose column name cannot drift with the
        ``model=`` keyword ``model_run`` passes it.
        """
        out = fit_once(model)
        assert {key for key in out if key.startswith("results_")} == {
            f"results_{label}" for label in labels
        }

    @pytest.mark.parametrize("model", sorted(EXPECTED_LABELS))
    def test_the_untuned_row_carries_exactly_the_documented_keys(self, model, fit_once):
        """The published key set, asserted as a set so a rename cannot slip through.

        ``qprofiler`` accumulates every inner key into one dict and writes the CSV
        header from the first row only, so a model contributing an extra or renamed key
        makes every later row wider than the header -- a file that still parses, with
        every column after the divergence shifted.
        """
        out = fit_once(model)
        for label in EXPECTED_LABELS[model]:
            row = _row(out, label)
            assert set(row) == set(DOCUMENTED_KEYS), (
                f"results_{label} does not carry the documented keys; "
                f"missing {sorted(set(DOCUMENTED_KEYS) - set(row))}, "
                f"unexpected {sorted(set(row) - set(DOCUMENTED_KEYS))}"
            )
            # Named separately from the set comparison because the two failures have
            # different causes: this one means modeleval took the grid_search branch
            # for a run that never searched anything, and sage.py and
            # qc_winner_finder.py both prefer that column when it is present.
            assert "BestParams_Tuned" not in row, (
                f"results_{label} claims tuned parameters, but grid_search is off "
                f"and no search ran"
            )

    @pytest.mark.parametrize("model", sorted(EXPECTED_LABELS))
    def test_the_recorded_model_name_matches_the_column_it_arrived_in(self, model, fit_once):
        """``modeleval`` interpolates the same string into both, so they cannot disagree
        unless a compute function overrode the label it was handed -- which is exactly
        what PQK and QPL do. Their ``model`` fields are 'pqk' and 'qpl_<head>' rather
        than the ``model=`` keyword, and ``qc_winner_finder`` groups on this field.
        """
        out = fit_once(model)
        for label in EXPECTED_LABELS[model]:
            assert _row(out, label)["model"] == label

    @pytest.mark.parametrize("metric", METRICS)
    @pytest.mark.parametrize("model", sorted(EXPECTED_LABELS))
    def test_the_metrics_are_finite_and_inside_the_unit_interval(self, model, metric, fit_once):
        """The cheapest possible smoke test for a fit that silently produced nothing.

        ``f1_score(average='weighted')`` returns 0.0 with a warning rather than raising
        on a degenerate prediction, and a NaN survives ``to_dict()`` and the CSV writer
        unremarked, so a collapsed quantum fit still hands back a plausible row.

        This assertion is deliberately weak -- a range this wide admits a hardwired
        constant -- and on its own it does not do what the paragraph above claims. The
        collapse it describes is caught by
        ``test_the_fit_predicted_both_classes_and_not_just_one`` below, which is the
        assertion this test used to decline to make: the stated reason was that VQC and
        QNN did not return the same answer twice, so no floor could be non-flaky. They
        are reproducible now (see ``TestTheSeedReachesTheQuantumStack``), so the reason
        has expired.
        """
        out = fit_once(model)
        for label in EXPECTED_LABELS[model]:
            value = _row(out, label)[metric]
            assert np.isfinite(value), f"results_{label}[{metric!r}] is {value!r}"
            assert 0.0 <= value <= 1.0, f"results_{label}[{metric!r}] is {value!r}"

    @pytest.mark.parametrize("model", sorted(EXPECTED_LABELS))
    def test_the_parameter_column_describes_the_circuit_that_produced_the_row(
        self, model, fit_once
    ):
        """``Model_Parameters`` is the only record of *which* quantum model this was.

        Two runs of the same dispatch key differing only in ``encoding`` or ``reps``
        produce identical column names, identical labels and identical metrics keys; the
        feature map recorded here is the sole discriminator in the results file. An
        empty dict would leave a results table in which two different feature maps are
        indistinguishable after the fact.
        """
        out = fit_once(model)
        for label in EXPECTED_LABELS[model]:
            params = _row(out, label)["Model_Parameters"]
            assert (
                isinstance(params, dict) and params
            ), f"results_{label} recorded no parameters: {params!r}"
            assert params.get("feature_map"), (
                f"results_{label} does not say which feature map it used: " f"{sorted(params)}"
            )

    @pytest.mark.parametrize("model", sorted(EXPECTED_LABELS))
    def test_the_labels_and_predictions_come_back_beside_every_row(self, model, fit_once, data):
        """``model_run`` returns three keys per label, not one.

        The docstring promises "keys as model names"; the truth is
        ``results_``/``y_test_``/``y_predicted_`` per label, and the other two thirds of
        the return value have never been asserted on. They are what ``qml_winner`` and
        any post-hoc metric recomputation read, so their length and content matter as
        much as the metrics do.
        """
        _, _, _, y_test = data
        out = fit_once(model)
        labels = EXPECTED_LABELS[model]
        assert set(out) == {
            f"{prefix}_{label}"
            for label in labels
            for prefix in ("results", "y_test", "y_predicted")
        }
        for label in labels:
            returned = np.asarray(out[f"y_test_{label}"][0])
            predicted = np.asarray(out[f"y_predicted_{label}"][0])
            assert np.array_equal(
                returned, y_test
            ), f"y_test_{label} is not the test labels it was given"
            assert predicted.shape == y_test.shape
            # A model that predicted a class it never saw would make every metric
            # above meaningless while staying finite and in range.
            assert set(np.unique(predicted)) <= set(np.unique(y_test))

    def test_qpl_reports_one_row_per_head_and_they_are_schema_identical(self, fit_once):
        """Each QPL head is a full row, not a variation on one.

        A helper that assumed a single ``results_`` column for QPL already got this
        wrong once -- it scored the first head and left the rest with an uncorrected
        time and parameter dict. Every head must therefore be a complete row on its own.

        The untuned path times each head from the start of the whole run rather than
        from its own fit, so the numbers are cumulative and include the shared
        projection: the second head's ``time`` covers the first head's search as well.
        That is the opposite of the tuned path, where ``_metric_dicts`` overwrites every
        head with one whole-search clock, so neither convention can be assumed from the
        other.
        """
        out = fit_once("qpl")
        rows = [_row(out, label) for label in EXPECTED_LABELS["qpl"]]
        assert len(rows) == len(QPL_HEADS)
        assert {frozenset(row) for row in rows} == {frozenset(DOCUMENTED_KEYS)}
        assert {row["model"] for row in rows} == {f"qpl_{head}" for head in QPL_HEADS}
        times = [row["time"] for row in rows]  # in the order the heads were configured
        assert all(value > 0 for value in times)
        assert times == sorted(times), (
            f"the heads are timed from one shared start, so their clocks cannot run "
            f"backwards in configuration order: {times}"
        )


    @pytest.mark.parametrize("model", sorted(EXPECTED_LABELS))
    def test_the_fit_predicted_both_classes_and_not_just_one(self, model, fit_once):
        """A quantum fit that collapses to one class still reports a plausible row.

        This is the assertion the range check above cannot make. On a balanced binary
        target, predicting a single class everywhere yields accuracy near 0.5 and --
        because ``f1_score(average='weighted')`` returns 0.0 with a warning rather than
        raising -- an f1 of 0.0, all of which sit inside [0, 1] and survive ``to_dict()``
        and the CSV writer unremarked. A collapsed circuit is the characteristic quantum
        failure (an unscaled feature landing anywhere on the circle, an optimizer that
        never moved off its initial point), so it is the one worth naming.

        Asserted on the predictions rather than on an accuracy floor on purpose: a floor
        is a number that has to be retuned whenever qiskit changes its optimizer
        defaults, whereas "it distinguished the classes at all" is the property actually
        meant and does not drift. Verified non-vacuous: all five models predict both
        classes on this fixture, so the assertion has something to lose.
        """
        out = fit_once(model)
        for label in EXPECTED_LABELS[model]:
            predictions = np.asarray(out[f"y_predicted_{label}"][0])
            assert len(set(predictions.tolist())) == 2, (
                f"{label} predicted only {set(predictions.tolist())} across every test "
                "row -- the fit collapsed to a single class, which accuracy and "
                "weighted f1 both report as a plausible-looking number"
            )

class TestTheShotsContract:
    """``args['shots']`` is required by three of the five models and ignored by two."""

    @pytest.mark.parametrize("model", SAMPLER_MODELS)
    def test_a_sampler_model_names_the_missing_key(self, model, data, sim_args):
        """This is the trap: the omission is only detectable at the backend, several
        frames into a worker, and the bare ``KeyError: 'shots'`` it used to raise named
        neither the key's purpose nor which of the two primitives wanted it. The
        message has to survive being read by someone who never chose a primitive at
        all -- ``primitive`` is a compute-function default, not a config key.
        """
        args = {key: value for key, value in sim_args.items() if key != "shots"}
        args["model"] = [model]
        X_train, X_test, y_train, y_test = data
        with pytest.raises(ValueError, match="shots") as raised:
            model_run(X_train, X_test, y_train, y_test, "no-shots", args)
        message = str(raised.value)
        assert (
            "sampler" in message
        ), f"the message does not say which primitive needed shots: {message}"
        assert (
            "simulator" in message
        ), f"the message does not say which backend was being opened: {message}"

    @pytest.mark.parametrize("model,model_args", PROJECTION_MODELS)
    def test_a_projection_model_does_not_need_shots(
        self, model, model_args, data, sim_args, tmp_path
    ):
        """The other half of the asymmetry, worth pinning in both directions.

        PQK and QPL request the estimator primitive, which returns expectation values
        rather than samples and takes no shot count -- so a config that runs only these
        two never needs ``shots``, and adding a validation that demanded it everywhere
        would break them. A fresh projection directory *and* a fresh ``data_key`` keep
        this cold: served from a warm cache the backend is never opened, and the test
        would prove nothing.
        """
        args = {key: value for key, value in sim_args.items() if key != "shots"}
        args["model"] = [model]
        args[f"{model}_args"] = dict(model_args)
        args["pqk_projection_dir"] = str(tmp_path / "pqk")
        args["qpl_projection_dir"] = str(tmp_path / "qpl")
        X_train, X_test, y_train, y_test = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = model_run(X_train, X_test, y_train, y_test, "cold-no-shots", args)
        results = sorted(key for key in out if key.startswith("results_"))
        assert results, f"{model} produced no results without 'shots': {sorted(out)}"
        for key in results:
            assert np.isfinite(out[key][0]["accuracy"])


class TestTheSeedReachesTheQuantumStack:
    """None of the five compute functions takes a ``random_state``.

    ``_seeded_kwargs`` inspects the signature and finds nothing, so for the quantum
    models the entire reproducibility mechanism is ``_call_with_global_seeds``:
    ``np.random.seed(args['seed'])`` plus a global quantum seed from
    ``args['q_seed']``. If that is wrong, a published quantum number cannot be
    reproduced -- and nothing else in the run would say so.
    """

    @pytest.mark.parametrize("model", ("pqk", "qsvc", "vqc", "qnn"))
    def test_two_runs_at_the_same_seed_agree(self, model, data, sim_args):
        """Same input, same seed, same answer -- for every quantum model.

        QSVC's randomness is the sampler's (seeded from ``args['seed']``) and PQK's is
        the randomized search over its projection (seeded the same way).

        VQC and QNN were originally excluded here, on the grounds that they were not
        reproducible. That was true, and it was a bug rather than a property:
        ``_call_with_global_seeds`` seeded ``qiskit_algorithms.utils.algorithm_globals``
        only, while both variational models draw their initial point through
        ``TrainableModel``, which reads the *separate*
        ``qiskit_machine_learning.utils.algorithm_globals`` singleton -- so both started
        their optimizer from OS entropy no matter what ``q_seed`` said. ``model_run`` now
        seeds both, and these two are the models this test exists for: they are the only
        ones whose answer the fix changed, so excluding them left it unguarded. Confirmed
        stable over repeated runs on this fixture before being added here.
        """
        X_train, X_test, y_train, y_test = data
        args = {**sim_args, "model": [model]}

        def once():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = model_run(X_train, X_test, y_train, y_test, "repeat", args)
            return _row(out, model)["accuracy"], np.asarray(out[f"y_predicted_{model}"][0])

        first_accuracy, first_predictions = once()
        second_accuracy, second_predictions = once()
        assert first_accuracy == second_accuracy
        assert np.array_equal(
            first_predictions, second_predictions
        ), "two runs at the same seed predicted different labels"

    def test_the_worker_reseeds_numpy_from_the_run_seed(self):
        """The half of ``_call_with_global_seeds`` that works, pinned so it stays.

        Asserted on the recorded RNG rather than on a metric: whether an unseeded run
        actually changes its answer depends on there being a tie to break, so a
        metric-level test can pass while the seeding is gone.
        """
        np.random.seed(7)
        expected = np.random.random()

        recorded = {}

        def recorder(*_args, **_kwargs):
            recorded["draw"] = np.random.random()
            return "sentinel"

        assert _call_with_global_seeds(recorder, 7, None) == "sentinel"
        assert recorded["draw"] == expected

    def test_a_run_without_a_quantum_seed_leaves_the_global_alone(self):
        """``q_seed`` is optional, and absent must mean untouched rather than reset.

        A caller that seeded the quantum globals itself -- which is what a notebook
        following the tutorials does -- would otherwise have that overwritten by a
        model_run whose config simply omits the key.
        """
        candidates = _quantum_global_rngs()
        previous = {name: rng.random_seed for name, rng in candidates.items()}
        try:
            for rng in candidates.values():
                rng.random_seed = 5
            _call_with_global_seeds(lambda *a, **k: None, 7, None)
            assert {name: rng.random_seed for name, rng in candidates.items()} == {
                name: 5 for name in candidates
            }
        finally:
            for name, rng in candidates.items():
                rng.random_seed = previous[name]

    def test_the_worker_seeds_every_global_rng_the_quantum_models_read(self):
        """Every ``algorithm_globals`` the installed stack exposes gets the seed.

        This used to fail, and it is why VQC and QNN were once excluded from the
        same-seed test above. ``_call_with_global_seeds`` set
        ``qiskit_algorithms.utils.algorithm_globals`` alone, but with
        qiskit-machine-learning 0.9 installed ``VQC`` and ``NeuralNetworkClassifier``
        draw their initial point through ``TrainableModel``, which reads
        ``qiskit_machine_learning.utils.algorithm_globals`` -- a *different* singleton
        with separate state, so setting the first left the second at ``None`` and both
        variational models started their optimizer from OS entropy regardless of
        ``q_seed``.

        Discovering that took measuring it: the two singletons are indistinguishable by
        name and only ``a is b`` shows they are not the same object. So this test asserts
        the general property rather than naming today's two modules -- it enumerates
        whatever singletons the stack exposes and requires the seed to reach *all* of
        them, which is what keeps a future qiskit that adds a third from silently
        reintroducing the bug.
        """
        candidates = _quantum_global_rngs()
        assert candidates, (
            "no algorithm_globals singleton found -- if the quantum stack no longer "
            "has one, _call_with_global_seeds' q_seed branch is dead code"
        )
        previous = {name: rng.random_seed for name, rng in candidates.items()}
        seen = {}
        try:
            for rng in candidates.values():
                rng.random_seed = None

            def recorder(*_args, **_kwargs):
                seen.update({name: rng.random_seed for name, rng in candidates.items()})
                return "sentinel"

            assert _call_with_global_seeds(recorder, 7, 11) == "sentinel"
        finally:
            for name, rng in candidates.items():
                rng.random_seed = previous[name]

        unseeded = sorted(name for name, seed in seen.items() if seed != 11)
        assert not unseeded, (
            "args['q_seed'] never reached these global RNGs, so the models that read "
            f"them start from OS entropy and cannot be reproduced: {unseeded}"
        )


def _quantum_global_rngs():
    """Every ``algorithm_globals`` singleton the quantum path actually reads, by module.

    Discovered from the code that consumes them rather than hardcoded: the initial
    point of a variational classifier comes from whichever singleton is in scope in
    ``TrainableModel``'s module, and ``compute_qnn``'s warm-up forward pass uses the one
    it imported itself. Splitting the qiskit algorithms helpers out of Qiskit and then
    into qiskit-machine-learning is how there came to be more than one.
    """
    from qiskit_machine_learning.algorithms.trainable_model import TrainableModel

    module_names = (TrainableModel.__module__, "qbiocode.learning.compute_qnn")
    found = {}
    seen_ids = set()
    for module_name in module_names:
        # Imported rather than read out of sys.modules: a module that happened not to be
        # loaded yet would silently narrow what this checks.
        rng = getattr(importlib.import_module(module_name), "algorithm_globals", None)
        if rng is None or id(rng) in seen_ids:
            continue
        seen_ids.add(id(rng))
        found[f"{module_name}.algorithm_globals"] = rng
    return found


def test_the_unknown_model_error_names_the_quantum_models_and_the_opt_rule(data, sim_args):
    """The message is where a user learns that ``_opt`` is not a model name.

    ``tests/test_error_contracts.py`` pins the classical half of this message. The
    quantum listing and the sentence explaining that the tuned variants are selected
    with ``args['grid_search']`` are the confusable part -- the ``_opt`` keys really are
    in the dispatch table, so naming one directly is accepted -- and both could be
    deleted today without a test noticing.
    """
    X_train, X_test, y_train, y_test = data
    args = {**sim_args, "model": ["qsvm"]}
    with pytest.raises(ValueError, match="Unknown model") as raised:
        model_run(X_train, X_test, y_train, y_test, "unknown", args)
    message = str(raised.value)
    assert "quantum" in message
    for name in sorted(EXPECTED_LABELS):
        assert f"'{name}'" in message, f"{name} is missing from: {message}"
    assert (
        "grid_search" in message
    ), f"the message does not say how the '_opt' variants are selected: {message}"
