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

"""Every ``_opt`` twin must be reachable *through the dispatcher*, not only by hand.

``args['grid_search'] = True`` is the only way a user ever selects a tuned learner:
the ``_opt`` names are rejected outright in ``args['model']``. So the wiring inside
:func:`qbiocode.evaluation.model_run.model_run` -- the ``method + "_opt"`` lookup,
``cv=args['cross_validation']``, ``tuner=``, ``n_trials=`` and the
``**_seeded_kwargs(fn, args['gridsearch_<model>_args'])`` splat -- *is* the tuned path
as anyone actually runs it.

That wiring was almost entirely untested. Twelve of the fourteen ``_opt`` twins were
reached only by calling ``compute_<model>_opt`` directly: ``test_grid_search_partial.py``
resolves the function out of ``sys.modules`` and ``test_quantum_tuning.py`` imports it,
which exercises the learner and skips the dispatcher completely. Only ``dt`` and ``rf``
(``test_split_reproducibility.py``, ``test_classical_models.py``) and one incidental
``qsvc`` assertion ever went through ``model_run`` with tuning on.

A single-branch mistake in ``model_run`` would therefore have left every one of those
direct-call tests green: a dropped ``cv=``, a ``gridsearch_`` prefix spelt
``grid_search_``, an ``_opt`` suffix forgotten for one model, a dropped ``tuner=``, or
the ``**_seeded_kwargs`` splat replaced by nothing. It is easy to miss because a
mis-wired tuned run still returns a complete, plausible results row -- nothing in the
row states that a search ran, and the row's *label* does not state it either. That is
the trap this file has to work around rather than rely on: ``model_run`` builds the
label from ``model=method + "_opt"`` (``model_run.py:419``), an expression independent
of the ``compute_ml_dict[method + "_opt"]`` lookup one line above it, so the ``_opt``
suffix survives a branch that dispatched the untuned function -- an untuned run under a
tuned name passes every naming assertion there is. ``tuner`` leaves even less behind:
both engines are handed the same candidates and answer with a value out of them, so a
frame produced by Optuna behind ``tuner: grid`` is indistinguishable from the grid's
own.

So the evidence is taken from the call the dispatcher made, not from the frame alone.
The ``_opt`` twin and the two search entry points (``run_study`` for Optuna,
``GridSearchCV`` for the grid) are each wrapped in a pass-through delegate that records
the keywords the dispatcher chose and then does the real work on real data; nothing
about any model is faked. The label is then asserted against a call that demonstrably
reached the tuned function, the fold count and the engine against the search that
received them, and the reported hyperparameters -- for all fourteen twins -- against the
config block that is the only place they could have come from.

Two defects were found while writing this file and pinned here with strict xfails, and
both were a label that lies. Both are fixed in source now, so both pins have become
positive assertions -- the history is kept because those two regressions are precisely
what the converted tests exist to catch:

* ``compute_pqk`` and ``compute_qpl`` threw the ``model=`` keyword away. The root cause
  was shadowing: each rebound the ``model`` *parameter* to its fitted estimator
  (``model = create_svc_model(...)``), so the label was gone before it could be used, and
  the hardcoded ``method_pqk = "pqk"`` and ``method_qpl = "qpl_" + head`` were the
  workarounds that kept the loss invisible. A tuned PQK or QPL was therefore reported
  under exactly the key an untuned one uses, so two runs differing by a whole
  hyperparameter search were indistinguishable in the results frame. The fitted object is
  ``estimator`` in both now and the label is honoured: ``results_pqk_opt``, and
  ``results_qpl_opt_<head>`` for each of QPL's six heads.
* ``BestParams_Tuned`` did not mean "tuned". ``modeleval`` chose the column name from
  ``args['grid_search']`` alone -- a run-wide flag -- so in a ``grid_search: True`` run an
  *untuned* quantum model, which is the documented default because ``tune_quantum`` is
  off unless asked for, filed its feature-map defaults under a column claiming a search.
  ``qbiocode.apps.sage`` and ``qbiocode.utils.qc_winner_finder`` both read that column
  before ``Model_Parameters``. The name is decided per row now, by ``modeleval``'s
  ``tuned=`` parameter and its ``_was_tuned`` helper, so one frame can carry both columns
  and each row carries the one its own provenance earns.
"""

import functools
import importlib
import math
import sys
from contextlib import contextmanager

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Imported at module scope, before anything fits: qbiocode/__init__.py is what orders
# xgboost's vendored libomp ahead of torch's, and TabPFN below drags torch in.
import qbiocode  # noqa: F401
from qbiocode import scale_train_test
from qbiocode.evaluation.model_run import model_run

# The dispatcher narrates the untuned-quantum case and Optuna's samplers warn about
# choices they cannot persist; neither is what any test here is about.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

def _classical_dataset():
    """60 rows, 4 features, a binary target with enough noise to be learnable but not
    perfectly separable -- a perfect split would make every hyperparameter tie, and a
    tie is exactly when Optuna's choice stops being pinned by the data."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + X[:, 1] + 0.3 * rng.normal(size=60) > 0).astype(int)
    return train_test_split(X, y, stratify=y, test_size=0.3, random_state=9)


def _quantum_dataset():
    """24 rows, 2 features -- two qubits, MinMax-scaled.

    Scaling is not cosmetic here: the feature maps encode magnitudes as rotation
    angles, so unscaled features land anywhere on the circle. Fitted on the training
    rows only, matching ``scale_train_test``, which is the protocol the library uses.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test = scale_train_test(X[:18], X[18:], scaling="MinMaxScaler")
    return X_train, X_test, y[:18], y[18:]


#: Everything ``model_run`` needs for a tuned classical run, minus the model and its
#: block. ``n_jobs: 1`` keeps joblib in this process, which is what lets the two tests
#: below observe the tuner call. ``cross_validation`` -- not ``cv`` -- is the config key
#: the dispatcher reads and forwards as the ``cv=`` argument.
CLASSICAL_ARGS = {
    "seed": 7,
    "n_jobs": 1,
    "grid_search": True,
    "cross_validation": 2,
    "n_trials": 2,
}


@pytest.fixture
def classical_data():
    return _classical_dataset()


@pytest.fixture
def quantum_data():
    return _quantum_dataset()


@pytest.fixture
def sim_args(tmp_path):
    """The keys the quantum stack reads, with tuning on.

    ``shots`` is required for the sampler. The projection directories point into
    ``tmp_path`` so PQK's and QPL's caches never land in the repository root and so one
    test's projections cannot be mistaken for another's: the cache name is built from
    ``data_key`` and a feature-map fingerprint but not the row count, and a stale file
    from a differently-sized split trips PQK's row-count guard.
    """
    return {
        "backend": "simulator",
        "shots": 64,
        "seed": 7,
        "q_seed": 7,
        "n_jobs": 1,
        "grid_search": True,
        "tune_quantum": True,
        "n_trials_quantum": 2,
        "pqk_projection_dir": str(tmp_path / "pqk_projections"),
        "qpl_projection_dir": str(tmp_path / "qpl_projections"),
    }


def learner_module(model):
    """The ``compute_<model>`` *module*, whose ``compute_<model>_opt`` attribute is the
    one ``model_run`` resolves.

    ``model_run`` imports the learners lazily, from inside the call, so that attribute
    is read afresh on every run -- which is what makes a delegate installed here visible
    to the dispatcher. ``qbiocode.learning`` re-exports each function under its module's
    own name, so the module has to be taken out of ``sys.modules`` rather than off the
    package.
    """
    importlib.import_module(f"qbiocode.learning.compute_{model}")
    return sys.modules[f"qbiocode.learning.compute_{model}"]


@contextmanager
def watched_opt_twin(model):
    """Record the keywords the dispatcher hands ``compute_<model>_opt``.

    The delegate passes everything straight to the real twin and returns its frame, so
    the search still runs on real data; ``functools.wraps`` keeps the wrapped signature
    visible because ``_seeded_kwargs`` reads it to decide whether this learner takes a
    ``random_state`` to fill in.

    This is the only way to see *which* function the dispatcher picked. The frame cannot
    say: every name in it is built from the ``model=`` keyword, and that keyword is a
    separate expression from the ``compute_ml_dict`` lookup it travels beside.
    """
    module = learner_module(model)
    attribute = f"compute_{model}_opt"
    real_opt = getattr(module, attribute)
    calls = []

    @functools.wraps(real_opt)
    def recording_opt(*call_args, **call_kwargs):
        calls.append(call_kwargs)
        return real_opt(*call_args, **call_kwargs)

    patched = pytest.MonkeyPatch()
    try:
        patched.setattr(module, attribute, recording_opt)
        yield calls
    finally:
        patched.undo()


@contextmanager
def watched_engines(model):
    """Record which of the two search engines ``compute_<model>_opt`` entered, with what.

    Every classical ``_opt`` learner branches on ``if tuner == "grid":`` to
    ``GridSearchCV`` and otherwise to ``run_study``, and both are module globals of
    ``compute_<model>``, so wrapping the two names says which branch ran and what ``cv``
    it was given. Nothing else does: an engine substitution is silent, and the sklearn
    complaint about an impossible fold count is phrased identically whether it surfaces
    from ``cross_val_score`` inside the study or from ``GridSearchCV.fit``.
    """
    module = learner_module(model)
    real = {"optuna": module.run_study, "grid": module.GridSearchCV}
    seen = {"optuna": [], "grid": []}

    def recorder(engine):
        def recording(*call_args, **call_kwargs):
            seen[engine].append(call_kwargs)
            return real[engine](*call_args, **call_kwargs)

        return recording

    patched = pytest.MonkeyPatch()
    try:
        patched.setattr(module, "run_study", recorder("optuna"))
        patched.setattr(module, "GridSearchCV", recorder("grid"))
        yield seen
    finally:
        patched.undo()


def results_rows(raw):
    """The ``results_*`` metrics rows of a ``model_run`` return, keyed by column name.

    Every value in that dict is a one-entry mapping keyed by the integer 0, because
    ``model_run`` pivots onto a single row index before ``to_dict()``. The column name
    is looked up rather than assumed: which label a learner reports is the subject of the
    two label tests below -- strict xfails until the labels were fixed -- so a helper that
    spelled it out would beg the question.
    """
    return {name: value[0] for name, value in raw.items() if name.startswith("results_")}


# --------------------------------------------------------------------------------------
# The nine classical twins
# --------------------------------------------------------------------------------------
#
# One cheap block per model, every offered value chosen so the estimator's own default
# is *not* among them. That is what makes "the block reached the search" observable at
# all: a dispatcher that dropped the `**_seeded_kwargs(...)` splat would either raise
# from build_search_space (nothing to search) or report a default, and both are visible.
# Two values per model, so the reported choice is a real selection rather than the only
# possibility -- the one-value case is pinned separately below.
CLASSICAL_BLOCKS = [
    ("svc", {"kernel": ["linear", "sigmoid"]}),          # SVC defaults to 'rbf'
    ("dt", {"max_depth": [2, 3]}),                       # default None
    ("lr", {"C": [0.25, 0.5]}),                          # default 1.0
    ("nb", {"var_smoothing": [1e-8, 1e-7]}),             # default 1e-9
    ("rf", {"n_estimators": [5, 7]}),                    # default 100
    ("xgb", {"max_depth": [2, 3]}),                      # default 6
    ("mlp", {"alpha": [0.01, 0.02], "max_iter": [50]}),  # default alpha 1e-4
    ("catboost", {"depth": [2, 3], "iterations": [10]}),  # default depth 6
    # TabPFN carries no skip guard on purpose. QBioCode pins model version v2, whose
    # weights are ungated -- no API token and no license acceptance -- so a tuned
    # TabPFN either dispatches or the [tabpfn] extra is missing, and the latter is a
    # broken dev install rather than a condition to tolerate quietly.
    ("tabpfn", {"n_estimators": [1, 2]}),
]

CLASSICAL_IDS = [name for name, _ in CLASSICAL_BLOCKS]


@pytest.fixture(scope="module")
def tuned_classical():
    """``model_run`` with ``grid_search: True``, once per model, cached.

    Returns ``(frame, calls)``: what ``model_run`` returned, and the keyword arguments
    the dispatcher handed ``compute_<model>_opt`` on the way there. The second half is
    what makes the label in the first half mean anything -- see ``watched_opt_twin``.
    Recording it in the fixture rather than in a test keeps it to one run per model.

    Module-scoped because two tests ask different questions of the same run and the
    fits are the expensive part; the runs are deterministic (``seed`` is fixed and
    reaches both the sampler and every estimator), so sharing one cannot make either
    test depend on the other.
    """
    cache = {}
    dataset = _classical_dataset()

    def run(model, block):
        if model not in cache:
            X_train, X_test, y_train, y_test = dataset
            args = {
                **CLASSICAL_ARGS,
                "model": [model],
                f"gridsearch_{model}_args": dict(block),
            }
            with watched_opt_twin(model) as calls:
                raw = model_run(X_train, X_test, y_train, y_test, "opt-twins", args)
            cache[model] = (raw, calls)
        return cache[model]

    return run


class TestEveryClassicalTwinIsReachedThroughTheDispatcher:
    """``grid_search: True`` must dispatch the ``_opt`` twin, for all nine of them."""

    @pytest.mark.parametrize("model,block", CLASSICAL_BLOCKS, ids=CLASSICAL_IDS)
    def test_the_tuned_run_is_reported_under_the_opt_label(
        self, model, block, tuned_classical
    ):
        """The ``_opt`` label has to name the function that actually ran.

        Checking the columns for the suffix cannot establish that, and it is worth being
        precise about why, because it is the mistake this test used to make: the
        dispatcher computes the label as ``model=method + "_opt"`` and resolves the
        function as ``compute_ml_dict[method + "_opt"]`` in two independent expressions,
        and ``modeleval`` then names ``results_``, ``y_test_``, ``y_predicted_`` and the
        row's own ``model`` field from that one keyword. So all four agree about the
        suffix no matter which function produced the numbers -- a branch that fell
        through to the untuned learner reports a complete, plausible ``<model>_opt`` row.

        The delegate closes that gap. The label is asserted against the ``model=`` of a
        call that reached ``compute_<model>_opt`` itself, and that call is required to
        carry the three keywords which make it a tuned call rather than a plain one:
        ``cv``, ``n_trials`` and ``tuner`` are added by the tuned branch alone, and the
        untuned branch passes ``data_key`` in their place.
        """
        raw, dispatched = tuned_classical(model, block)

        assert len(dispatched) == 1, (
            f"grid_search: True on {model!r} called compute_{model}_opt "
            f"{len(dispatched)} times, expected exactly once -- zero means the "
            f"dispatcher ran a different function and labelled the row {model}_opt "
            f"regardless, which no column in the frame can reveal"
        )
        call = dispatched[0]
        assert call.get("cv") == CLASSICAL_ARGS["cross_validation"], (
            f"compute_{model}_opt was called with cv={call.get('cv')!r} rather than "
            f"args['cross_validation']={CLASSICAL_ARGS['cross_validation']!r}"
        )
        assert call.get("n_trials") == CLASSICAL_ARGS["n_trials"], (
            f"compute_{model}_opt was called with n_trials={call.get('n_trials')!r} "
            f"rather than args['n_trials']={CLASSICAL_ARGS['n_trials']!r}"
        )
        assert call.get("tuner") == "optuna", (
            f"compute_{model}_opt was called with tuner={call.get('tuner')!r}; the "
            f"config named no engine, so the dispatcher's default must arrive as the "
            f"documented 'optuna' -- a missing keyword leaves the learner to default it "
            f"silently, and no results row records which engine ran"
        )
        label = call.get("model")
        assert label == f"{model}_opt", (
            f"the tuned twin ran but was told to report itself as {label!r}; every "
            f"column of the frame and the row's own 'model' field are named from this "
            f"one keyword, so the whole run would be filed under the untuned key"
        )

        rows = results_rows(raw)
        assert set(rows) == {f"results_{label}"}, (
            f"the run labelled {label!r} is reported as {sorted(rows)}"
        )
        for prefix in ("y_test_", "y_predicted_"):
            assert f"{prefix}{label}" in raw, (
                f"{prefix!r} column does not carry the label {label!r} the tuned twin "
                f"was given, so the three columns of one run disagree: {sorted(raw)}"
            )
        row = rows[f"results_{label}"]
        assert row["model"] == label, (
            f"the row's own 'model' field says {row['model']!r} rather than {label!r}, "
            f"and that field is what lands in ModelResults.csv and in QuantumSage's "
            f"training data"
        )
        # A real ranking ROC AUC since the auc fix, not balanced accuracy -- and the
        # only thing here that would catch an _opt call site that forgot to pass
        # `y_score=` to modeleval, which records NaN when no score arrives. `accuracy`
        # and `f1_score` are not checked: sklearn returns a float in [0, 1] for any
        # binary y_true/y_pred pair, so `isfinite` on either cannot fail.
        assert math.isfinite(row["auc"]) and 0.0 <= row["auc"] <= 1.0, (
            f"{model}_opt reported auc={row['auc']!r}; every learner shipped here "
            f"produces a real score on a binary target, so NaN means the tuned call "
            f"site dropped the y_score keyword"
        )

    @pytest.mark.parametrize("model,block", CLASSICAL_BLOCKS, ids=CLASSICAL_IDS)
    def test_the_reported_choice_is_one_the_config_block_offered(
        self, model, block, tuned_classical
    ):
        """``args['gridsearch_<model>_args']`` has to reach the search, not just the call.

        The dispatcher splats that block through ``_seeded_kwargs`` into
        ``compute_<model>_opt``. Every value offered above is one the estimator would
        not have chosen for itself, so a reported value from the block is proof the
        search saw it -- and a mis-spelt prefix would instead reach
        ``build_search_space`` with nothing and raise.
        """
        raw, _ = tuned_classical(model, block)
        row = results_rows(raw)[f"results_{model}_opt"]
        tuned = row["BestParams_Tuned"]
        assert isinstance(tuned, dict), f"BestParams_Tuned is {type(tuned).__name__}"
        for name, offered in block.items():
            assert name in tuned, (
                f"{name!r} was offered for tuning but is absent from the reported "
                f"parameters {sorted(tuned)}, so nothing says it was searched"
            )
            assert tuned[name] in offered, (
                f"{model}_opt reports {name}={tuned[name]!r}, which is not one of the "
                f"offered values {offered} -- the config block did not reach the search"
            )


#: ``(model, estimator class, hyperparameter, the single value to pin)``. The class is
#: instantiated in the test to show the pinned value differs from the estimator's own
#: default, so the assertion cannot pass by the search being skipped. CatBoost, XGBoost
#: and TabPFN are absent only because their defaults are not sklearn attributes to read.
PINNED = [
    ("svc", SVC, "kernel", "sigmoid"),
    ("dt", DecisionTreeClassifier, "max_depth", 3),
    ("lr", LogisticRegression, "C", 0.25),
    ("nb", GaussianNB, "var_smoothing", 1e-7),
    ("rf", RandomForestClassifier, "n_estimators", 7),
    ("mlp", MLPClassifier, "alpha", 0.01),
]


class TestTheConfigBlockReachesTheSearch:
    """A one-value block is a complete config, and it must pin the result exactly."""

    @pytest.mark.parametrize(
        "model,estimator_cls,name,value", PINNED, ids=[m for m, *_ in PINNED]
    )
    def test_offering_a_single_value_pins_the_reported_parameter(
        self, model, estimator_cls, name, value, classical_data
    ):
        assert getattr(estimator_cls(), name) != value, (
            f"{estimator_cls.__name__} already defaults {name} to {value!r}, so this "
            f"test would pass without the search running at all -- pick another value"
        )
        X_train, X_test, y_train, y_test = classical_data
        args = {
            **CLASSICAL_ARGS,
            "model": [model],
            f"gridsearch_{model}_args": {name: [value]},
        }
        raw = model_run(X_train, X_test, y_train, y_test, "pinned", args)
        tuned = results_rows(raw)[f"results_{model}_opt"]["BestParams_Tuned"]
        assert tuned[name] == value, (
            f"{model}_opt was offered exactly one value for {name} and reported "
            f"{tuned[name]!r} instead of {value!r}"
        )

    def test_the_runs_seed_reaches_the_tuner_and_the_tuned_estimator(self, classical_data):
        """``_seeded_kwargs`` fills ``random_state`` in on the way to the ``_opt`` twin.

        The seed has two jobs inside a tuned run -- it seeds Optuna's sampler and it is
        handed to every candidate estimator as ``fixed={'random_state': ...}`` -- and
        neither is visible in the results row, because ``BestParams_Tuned`` reports only
        the *searched* hyperparameters. So the tuner entry point is wrapped in a
        recording delegate that still runs the real search on real data; nothing about
        the model is faked.
        """
        module = learner_module("dt")
        real_run_study = module.run_study
        seen = []

        def recording_run_study(*args, **kwargs):
            seen.append(kwargs)
            return real_run_study(*args, **kwargs)

        X_train, X_test, y_train, y_test = classical_data
        args = {
            **CLASSICAL_ARGS,
            "model": ["dt"],
            "gridsearch_dt_args": {"max_depth": [2, 3]},
        }
        monkeypatched = pytest.MonkeyPatch()
        try:
            monkeypatched.setattr(module, "run_study", recording_run_study)
            model_run(X_train, X_test, y_train, y_test, "seeded", args)
        finally:
            monkeypatched.undo()

        assert len(seen) == 1, f"the tuner ran {len(seen)} times, expected once"
        assert seen[0]["seed"] == CLASSICAL_ARGS["seed"], (
            f"Optuna's sampler was seeded with {seen[0]['seed']!r} rather than "
            f"args['seed']={CLASSICAL_ARGS['seed']!r}, so two runs at one seed may "
            f"search differently"
        )
        assert seen[0]["fixed"] == {"random_state": CLASSICAL_ARGS["seed"]}, (
            f"the candidate estimators were built with fixed={seen[0]['fixed']!r}; "
            f"_seeded_kwargs must reach compute_dt_opt's random_state, or a tie "
            f"between two equally-good splits breaks at random"
        )


class TestTheFoldCountAndTrialBudgetReachTheTuner:
    """``cross_validation``, ``n_trials`` and ``tuner`` are forwarded by the dispatcher.

    All three are set on the classical branch of ``model_run`` and nowhere else -- the
    ``_opt`` functions default them to 5, 50 and ``'optuna'`` -- so a direct call to
    ``compute_<model>_opt`` cannot tell whether the config value ever arrived.

    ``tuner`` is the awkward one, and the reason these tests watch the engines rather
    than read messages: the two search paths are interchangeable from the outside.
    ``tuner`` is asserted here by which entry point the learner reached, because a
    ``tuner=`` keyword dropped from the dispatcher's call would substitute Optuna for a
    configured grid without changing anything a caller can see -- the error text for an
    impossible fold count is sklearn's either way, and a successful grid run and a
    successful Optuna run over the same list are indistinguishable in the frame.
    """

    @pytest.mark.parametrize("tuner", ["optuna", "grid"])
    def test_an_impossible_fold_count_is_refused_by_the_search_that_received_it(
        self, tuner, classical_data
    ):
        """100 folds over 42 training rows, refused by the engine the config named.

        The number in the message came from the config and from nothing else, so seeing
        it proves ``cross_validation`` was forwarded. It does not prove *where* it went:
        ``cross_val_score`` inside the study and ``GridSearchCV.fit`` raise the same
        sentence, so the message is the same under either engine and under a dropped
        ``tuner=``. The engine is therefore identified by which entry point was reached
        and with what ``cv`` -- and the other one is required not to have run at all,
        which is what a silent substitution would look like.
        """
        other = "grid" if tuner == "optuna" else "optuna"
        X_train, X_test, y_train, y_test = classical_data
        args = {
            **CLASSICAL_ARGS,
            "model": ["dt"],
            "tuner": tuner,
            "cross_validation": 100,
            "gridsearch_dt_args": {"max_depth": [2]},
        }
        with watched_engines("dt") as seen:
            with pytest.raises(ValueError) as excinfo:
                model_run(X_train, X_test, y_train, y_test, "folds", args)
        message = str(excinfo.value)
        assert "100" in message and "splits" in message, (
            f"the failure does not mention the 100 folds the config asked for, so "
            f"args['cross_validation'] may never have reached the search: {message}"
        )
        assert len(seen[tuner]) == 1 and not seen[other], (
            f"args['tuner'] = {tuner!r} but the run entered the {tuner} engine "
            f"{len(seen[tuner])} times and the {other} engine {len(seen[other])} "
            f"times; the fold count above was refused by whichever one ran, so the "
            f"message says nothing about this"
        )
        assert seen[tuner][0]["cv"] == 100, (
            f"the {tuner} engine was entered with cv={seen[tuner][0].get('cv')!r} "
            f"rather than the configured 100"
        )

    def test_the_grid_engine_reaches_the_same_labelled_row_as_the_optuna_engine(
        self, classical_data
    ):
        """The happy path of ``tuner: grid`` through the dispatcher, which had none.

        Everything an exhaustive sweep is asked to produce is asserted here in one
        place, because ``tuner: grid`` is the branch a user reaches for to reproduce a
        published number and it was only ever run to a failure above: the fold count
        arrives at ``GridSearchCV`` itself, the Optuna entry point stays untouched, and
        the row comes back under the tuned label carrying a value out of the block. The
        engine watch is what distinguishes this from an Optuna run -- the row alone
        would look identical if ``tuner=`` never left ``model_run``.
        """
        X_train, X_test, y_train, y_test = classical_data
        args = {
            **CLASSICAL_ARGS,
            "model": ["dt"],
            "tuner": "grid",
            "cross_validation": 3,
            "gridsearch_dt_args": {"max_depth": [2, 3]},
        }
        with watched_engines("dt") as seen:
            raw = model_run(X_train, X_test, y_train, y_test, "grid-engine", args)

        assert len(seen["grid"]) == 1 and not seen["optuna"], (
            f"tuner: grid entered GridSearchCV {len(seen['grid'])} times and run_study "
            f"{len(seen['optuna'])} times; Optuna answering a request for the grid is "
            f"the substitution nothing else in a results frame can show"
        )
        assert seen["grid"][0]["cv"] == 3, (
            f"GridSearchCV was built with cv={seen['grid'][0].get('cv')!r} rather than "
            f"the configured cross_validation: 3"
        )
        row = results_rows(raw)["results_dt_opt"]
        assert row["model"] == "dt_opt", (
            f"a grid-tuned run must be labelled like an Optuna-tuned one; got "
            f"{row['model']!r}"
        )
        assert row["BestParams_Tuned"]["max_depth"] in (2, 3), (
            f"the grid reported max_depth={row['BestParams_Tuned']['max_depth']!r}, "
            f"which is not one of the values gridsearch_dt_args offered"
        )

    def test_a_zero_trial_budget_is_refused_in_the_config_s_own_terms(self, classical_data):
        """``n_trials: 0`` used to reach ``study.best_params`` with nothing completed."""
        X_train, X_test, y_train, y_test = classical_data
        args = {
            **CLASSICAL_ARGS,
            "model": ["dt"],
            "n_trials": 0,
            "gridsearch_dt_args": {"max_depth": [2, 3]},
        }
        with pytest.raises(ValueError) as excinfo:
            model_run(X_train, X_test, y_train, y_test, "budget", args)
        message = str(excinfo.value)
        assert "n_trials is 0" in message, (
            f"args['n_trials'] did not reach the budget check: {message}"
        )
        assert "grid_search: False" in message, "must say how to opt out of tuning"

    def test_both_values_arrive_at_the_tuner_unaltered(self, classical_data):
        """The positive half of the two tests above: the legal values arrive as given.

        Five candidate depths against a budget of three keeps the two numbers
        distinguishable -- a budget silently replaced by the space size would show up
        as 5, and one left at the ``_opt`` default as 50. ``n_trials`` is meaningful to
        one engine only, which is why the study is also required to be the engine that
        received it: an exhaustive grid fits all five depths whatever budget it is given.
        """
        X_train, X_test, y_train, y_test = classical_data
        args = {
            **CLASSICAL_ARGS,
            "model": ["dt"],
            "cross_validation": 3,
            "n_trials": 3,
            "gridsearch_dt_args": {"max_depth": [2, 3, 4, 5, 6]},
        }
        with watched_engines("dt") as seen:
            model_run(X_train, X_test, y_train, y_test, "forwarded", args)

        study = seen["optuna"]
        assert len(study) == 1 and not seen["grid"], (
            f"the config named no engine, so the study is the one that must run: "
            f"run_study {len(study)} times, GridSearchCV {len(seen['grid'])} times"
        )
        assert study[0]["cv"] == 3, (
            f"the search cross-validated with cv={study[0]['cv']!r} rather than the "
            f"configured cross_validation: 3"
        )
        assert study[0]["n_trials"] == 3, (
            f"the search was given n_trials={study[0]['n_trials']!r} rather than the "
            f"configured 3"
        )


# --------------------------------------------------------------------------------------
# The five quantum twins
# --------------------------------------------------------------------------------------
#
# Deliberately NOT marked requires_quantum: that marker is for a real device. These run
# on the local simulator, so they belong in the default tier where a broken dispatch
# branch is actually noticed.
#
# `reps: [1]` everywhere keeps every circuit shallow. Each space is finite and tiny, so
# the budget cap makes the studies exhaustive and quick.
QUANTUM_BLOCKS = [
    ("qsvc", {"encoding": ["Z", "ZZ"], "reps": [1]}),
    ("vqc", {"encoding": ["Z"], "reps": [1], "maxiter": [5, 8]}),
    ("qnn", {"encoding": ["Z"], "reps": [1], "maxiter": [5, 8]}),
]


class TestEveryQuantumTwinIsReachedThroughTheDispatcher:
    """``grid_search`` *and* ``tune_quantum`` together select the quantum ``_opt`` twin.

    The quantum branch of ``model_run`` is a third code path, not a variation of the
    classical one: it passes ``n_trials=args['n_trials_quantum']`` and
    ``validation_split=``, and passes no ``cv`` or ``tuner`` at all, because a quantum
    candidate is scored on one stratified holdout rather than k folds and there is no
    exhaustive-grid engine for it.
    """

    @pytest.mark.parametrize(
        "model,block", QUANTUM_BLOCKS, ids=[m for m, _ in QUANTUM_BLOCKS]
    )
    def test_the_tuned_run_is_reported_under_the_opt_label(
        self, model, block, quantum_data, sim_args
    ):
        X_train, X_test, y_train, y_test = quantum_data
        args = {**sim_args, "model": [model], f"gridsearch_{model}_args": dict(block)}
        raw = model_run(X_train, X_test, y_train, y_test, "opt-twins-q", args)
        rows = results_rows(raw)

        assert set(rows) == {f"results_{model}_opt"}, (
            f"tune_quantum on {model!r} produced {sorted(rows)}; the dispatcher must "
            f"run compute_{model}_opt and label it {model}_opt"
        )
        row = rows[f"results_{model}_opt"]
        assert row["model"] == f"{model}_opt"
        assert "BestParams_Tuned" in row and "Model_Parameters" not in row, sorted(row)
        assert math.isfinite(row["accuracy"]), f"accuracy={row['accuracy']!r}"

    @pytest.mark.parametrize(
        "model,block", QUANTUM_BLOCKS, ids=[m for m, _ in QUANTUM_BLOCKS]
    )
    def test_the_reported_choice_is_one_the_config_block_offered(
        self, model, block, quantum_data, sim_args
    ):
        """The block is splatted into ``compute_<model>_opt`` on the quantum branch too.

        ``record_tuned_params`` lays the tuned values over the base function's own
        parameter dict, so a reported ``encoding``/``reps``/``maxiter`` is one the search
        chose; the feature-map class name sitting beside it came from the base call.
        """
        X_train, X_test, y_train, y_test = quantum_data
        args = {**sim_args, "model": [model], f"gridsearch_{model}_args": dict(block)}
        raw = model_run(X_train, X_test, y_train, y_test, "opt-twins-q", args)
        tuned = results_rows(raw)[f"results_{model}_opt"]["BestParams_Tuned"]
        for name, offered in block.items():
            assert name in tuned, (
                f"{name!r} was offered for tuning but is absent from {sorted(tuned)}"
            )
            assert tuned[name] in offered, (
                f"{model}_opt reports {name}={tuned[name]!r}, not one of {offered}"
            )

    def test_the_quantum_budget_comes_from_its_own_config_key(self, quantum_data, sim_args):
        """``n_trials_quantum``, not ``n_trials``.

        The two keys are separate on purpose -- a quantum trial is a whole quantum fit,
        so the classical default of 50 would be ruinous -- and a branch that read the
        classical key would be invisible in any results row. Setting ``n_trials`` to a
        value the budget check rejects while ``n_trials_quantum`` stays legal makes the
        confusion fatal if it exists, and a no-op if it does not.
        """
        X_train, X_test, y_train, y_test = quantum_data
        args = {
            **sim_args,
            "model": ["qsvc"],
            "n_trials": 0,
            "n_trials_quantum": 1,
            "gridsearch_qsvc_args": {"reps": [1]},
        }
        raw = model_run(X_train, X_test, y_train, y_test, "q-budget", args)
        assert "results_qsvc_opt" in raw, sorted(raw)

    def test_a_zero_quantum_budget_is_refused_in_the_config_s_own_terms(
        self, quantum_data, sim_args
    ):
        X_train, X_test, y_train, y_test = quantum_data
        args = {
            **sim_args,
            "model": ["qsvc"],
            "n_trials_quantum": 0,
            "gridsearch_qsvc_args": {"reps": [1]},
        }
        with pytest.raises(ValueError) as excinfo:
            model_run(X_train, X_test, y_train, y_test, "q-budget", args)
        assert "n_trials is 0" in str(excinfo.value), str(excinfo.value)

    def test_the_validation_split_reaches_the_quantum_tuner(self, quantum_data, sim_args):
        """``validation_split`` is forwarded only by the quantum branch of ``model_run``.

        It is refused in the config's own vocabulary rather than by ``train_test_split``,
        which would report it against ``test_size`` -- a name that appears in no config.
        """
        X_train, X_test, y_train, y_test = quantum_data
        args = {
            **sim_args,
            "model": ["qsvc"],
            "validation_split": 1.5,
            "gridsearch_qsvc_args": {"reps": [1]},
        }
        with pytest.raises(ValueError) as excinfo:
            model_run(X_train, X_test, y_train, y_test, "q-split", args)
        message = str(excinfo.value)
        assert "validation_split" in message and "1.5" in message, message
        assert "test_size" not in message, "must not blame a parameter the user never set"


# --------------------------------------------------------------------------------------
# PQK and QPL: tuned, and now reported under a name of their own
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tuned_pqk(tmp_path_factory):
    """One tuned PQK dispatch, shared by the two tests that read it.

    Module-scoped because the run costs a few seconds of circuit simulation and both
    tests below ask about the same frame; ``tmp_path_factory`` keeps the projection
    cache out of the repository root.
    """
    scratch = tmp_path_factory.mktemp("pqk_dispatch")
    X_train, X_test, y_train, y_test = _quantum_dataset()
    args = {
        "backend": "simulator",
        "shots": 64,
        "seed": 7,
        "q_seed": 7,
        "n_jobs": 1,
        "grid_search": True,
        "tune_quantum": True,
        "n_trials_quantum": 2,
        "model": ["pqk"],
        "pqk_projection_dir": str(scratch / "pqk_projections"),
        "gridsearch_pqk_args": {"encoding": ["Z"], "reps": [1]},
    }
    return model_run(X_train, X_test, y_train, y_test, "pqk-dispatch", args)


#: The heads ``compute_qpl`` runs when ``classical_models`` is left at its default --
#: which is every head a tuned run has, because ``compute_qpl_opt`` takes no
#: ``classical_models`` argument at all.
#:
#: Read off the source rather than guessed: ``compute_qpl`` sets
#: ``classical_models = ["rf", "mlp", "svc", "lr", "xgb", "catboost"]``. Note what is
#: NOT there -- ``tabpfn``. An earlier version of this constant listed it as a seventh
#: head, and the error was invisible because the set was only ever *intersected* with the
#: labels that came back: a name that never appears simply drops out, so the premise could
#: be wrong without any assertion noticing.
#: ``test_the_default_head_list_is_what_this_file_assumes`` below pins it against the
#: source so it cannot drift again, and the two QPL tests now compare the emitted head set
#: against it by *equality*, which is what makes a missing head fail as well as a surplus
#: one. Equality is honest here: ``compute_qpl`` does drop ``xgb`` or ``catboost`` when
#: their import fails, but both are declared in requirements-base.txt, so neither is
#: absent in any supported install -- the same reasoning CLASSICAL_BLOCKS above uses to
#: carry no skip guard.
DEFAULT_QPL_HEADS = ("rf", "mlp", "svc", "lr", "xgb", "catboost")

#: The labels an *untuned* QPL run writes. ``compute_qpl`` names each head's columns
#: ``f"{model}_{method}"``, so an untuned run (``model="qpl"``) writes ``qpl_<head>`` and
#: a tuned one (``model="qpl_opt"``) writes ``qpl_opt_<head>``. A tuned run must reuse
#: none of these, which is the discharged pin below.
UNTUNED_QPL_LABELS = frozenset(f"qpl_{head}" for head in DEFAULT_QPL_HEADS)


def split_qpl_label(label):
    """Split a QPL model label into its ``model=`` prefix and its classical head.

    ``compute_qpl`` names each head's columns ``f"{model}_{method}"``, so the head is the
    final ``_``-separated token and everything before it is whatever the dispatcher passed
    as ``model=``: ``('qpl', 'rf')`` for an untuned run, ``('qpl_opt', 'rf')`` for a tuned
    one. Partitioning from the right rather than stripping a known prefix is what lets one
    helper read both spellings -- and no head name contains an underscore, which is what
    makes the split unambiguous.
    """
    prefix, _, head = label.rpartition("_")
    return prefix, head


def test_the_default_head_list_is_what_this_file_assumes():
    """``DEFAULT_QPL_HEADS`` is a premise, and an unpinned premise silently rots.

    The constant used to be intersected with the labels a run produced, so a head listed
    here that ``compute_qpl`` does not actually run contributed nothing and was never
    noticed. That is exactly what happened: the list carried a seventh head, ``tabpfn``,
    that the default has never included, and every assertion built on it still passed.

    Reading the default out of the source keeps the premise honest, and the QPL tests
    below now assert set equality against it so that a head which stops running fails too.
    Both halves are needed and neither substitutes for the other: equality catches a head
    vanishing from a run, and this test catches the constant drifting away from the
    source. If someone adds a head to ``compute_qpl``, this one fails and points at the
    constant above rather than letting six label assertions quietly cover one fewer model
    than they claim.
    """
    import inspect
    import re

    # `qbiocode.learning.compute_qpl` is the re-exported FUNCTION, not the module of the
    # same name, so inspect the function directly rather than reaching through a module.
    from qbiocode.learning import compute_qpl

    source = inspect.getsource(compute_qpl)
    match = re.search(r"classical_models\s*=\s*\[([^\]]*)\]", source)
    assert match, "compute_qpl no longer assigns a default classical_models list"
    declared = tuple(name.strip().strip("\"'") for name in match.group(1).split(",") if name.strip())

    assert declared == DEFAULT_QPL_HEADS, (
        f"compute_qpl's default heads are {declared}, but this file assumes "
        f"{DEFAULT_QPL_HEADS}. Update DEFAULT_QPL_HEADS -- the QPL tests below compare "
        "the emitted head set against it for equality, and UNTUNED_QPL_LABELS is derived "
        "from it, so a stale entry breaks both in the same stroke."
    )


class TestTunedPqkAndQplAreLabelledApartFromAnUntunedRun:
    """The dispatch works, and the label it reports now distinguishes it from an untuned run.

    It did not always, and both halves of this class were strict xfails until it did. Both
    ``_opt`` wrappers hand the base function ``model="<name>_opt"``, and both base
    functions used to throw it away: each rebound that very parameter to its fitted
    estimator (``model = create_svc_model(...)``) and then rebuilt the label from a
    constant -- ``method_pqk = "pqk"`` in ``compute_pqk``, ``method_qpl = "qpl_" + head``
    in ``compute_qpl``. Everything downstream -- the ``results_`` column, the row's
    ``model`` field, the ``y_test_``/``y_predicted_`` columns -- is named from that value,
    so a tuned run and an untuned one came out byte-identical in shape. ``qsvc``, ``vqc``
    and ``qnn`` always got this right, which is why it read as an oversight rather than a
    convention.

    The fitted objects are named ``estimator`` now, ``method_pqk = model`` and
    ``method_qpl = f"{model}_{method}"``, and the two tests that were pins assert the
    labels those produce.
    """

    def test_the_dispatcher_does_reach_the_tuned_pqk_wrapper(self, tuned_pqk):
        """Located before the naming: the search really ran.

        ``encoding`` and ``reps`` appear in the parameter dict only because
        ``record_tuned_params`` laid the study's result over what ``compute_pqk``
        recorded, and only ``compute_pqk_opt`` calls it. The column is still found by
        prefix rather than named, so that this test keeps answering "did the search run?"
        on its own: the name it carries is the subject of the test below, which was a
        strict xfail, and naming it here would make one regression fail both.
        """
        rows = results_rows(tuned_pqk)
        assert len(rows) == 1, f"expected one metrics row, got {sorted(rows)}"
        (row,) = rows.values()
        tuned = row["BestParams_Tuned"]
        assert tuned.get("encoding") == "Z" and tuned.get("reps") == 1, (
            f"the tuned hyperparameters are absent from {sorted(tuned)}, so "
            f"compute_pqk_opt was never reached -- the dispatcher ran the untuned "
            f"function instead"
        )
        assert math.isfinite(row["accuracy"]), f"accuracy={row['accuracy']!r}"

    def test_a_tuned_pqk_is_labelled_pqk_opt(self, tuned_pqk):
        """A tuned PQK must not be filed under the untuned key.

        This was a strict xfail. ``compute_pqk`` rebound its own ``model`` parameter to
        the fitted SVC and then set ``method_pqk = "pqk"``, so the ``model="pqk_opt"``
        that ``compute_pqk_opt`` hands it was discarded before it could be used: a tuned
        run produced ``results_pqk`` with ``model='pqk'``, indistinguishable in
        ModelResults.csv from a run with no search at all. The estimator is called
        ``estimator`` now and ``method_pqk = model``, so the label survives.

        What this guards is that the untuned spelling appears nowhere in the frame -- the
        shape the bug produced, and the one statement that covers all three column
        families at once -- and then that all four names built from that one keyword agree
        and carry the label: the ``results_`` column, the row's own ``model`` field, and
        the ``y_test_``/``y_predicted_`` columns. That order is deliberate. The absence
        check is what a reversion of the shadowing breaks, and asserted after the presence
        checks it could never be the assertion that fails, since no single dispatch can
        produce ``results_pqk`` and ``results_pqk_opt`` at once. Each check below it stays
        reachable on its own: a label misspelt some third way (fix (B)'s upper-case
        ``'PQK'``, say) satisfies the absence check and fails the presence ones.

        PQK has a single head and no head name to place, so ``pqk_opt`` is the only
        spelling a fix could produce; its QPL sibling below needed more care about that.
        """
        untuned = sorted(name for name in tuned_pqk if name.endswith("_pqk"))
        assert not untuned, (
            f"{untuned} are the exact keys an untuned PQK run writes, so a whole "
            f"hyperparameter search leaves no trace in ModelResults.csv"
        )
        assert "results_pqk_opt" in tuned_pqk, sorted(tuned_pqk)
        assert tuned_pqk["results_pqk_opt"][0]["model"] == "pqk_opt", (
            f"the results column carries the tuned label but the row's own 'model' field "
            f"says {tuned_pqk['results_pqk_opt'][0]['model']!r}, and that field is what "
            f"lands in ModelResults.csv and in QuantumSage's training data"
        )
        for prefix in ("y_test_", "y_predicted_"):
            assert f"{prefix}pqk_opt" in tuned_pqk, (
                f"{prefix}pqk_opt is missing, so the three columns of one run disagree "
                f"about its name: {sorted(tuned_pqk)}"
            )

    def test_the_dispatcher_does_reach_the_tuned_qpl_wrapper_for_every_head(self, tuned_qpl):
        """QPL fits one classical head per entry in ``classical_models`` on the same
        projection, so the frame carries one ``results_`` column per head. Every one of
        them must show the tuned values: ``record_tuned_params`` corrects each row, and
        a helper that assumed a single column would leave the rest uncorrected.

        The head set is asserted by equality, which is a widening. This test used to
        accept any ``len(rows) > 1``, so nothing in this file said *which* heads a tuned
        QPL run emits -- which is why it had gone unnoticed that a tuned frame carries six
        of them and no ``qpl_tabpfn``. Under the old assertion a head dropped from the run
        left the study one model short with no test the wiser, because the label
        assertions below only ever intersected against the constant and a name that never
        appears intersects away silently.

        The heads are read with ``split_qpl_label``, so this holds for either spelling: it
        asks which heads ran, and the test after it asks what they were called.
        """
        rows = results_rows(tuned_qpl)
        heads = {split_qpl_label(name[len("results_"):])[1] for name in rows}
        assert heads == set(DEFAULT_QPL_HEADS), (
            f"a tuned QPL run reported heads {sorted(heads)} out of columns "
            f"{sorted(rows)}, not {sorted(DEFAULT_QPL_HEADS)}. compute_qpl_opt takes no "
            f"classical_models argument, so the default list is what every trial and the "
            f"final fit search -- a head missing here is a model silently absent from the "
            f"run, and a surplus one means this file's premise has rotted"
        )
        for name, row in rows.items():
            tuned = row["BestParams_Tuned"]
            assert tuned.get("encoding") == "Z" and tuned.get("reps") == 1, (
                f"{name} does not carry the tuned hyperparameters ({sorted(tuned)}), "
                f"so compute_qpl_opt was not reached for that head"
            )
            assert math.isfinite(row["accuracy"]), f"{name}: {row['accuracy']!r}"

    def test_a_tuned_qpl_is_labelled_apart_from_an_untuned_one(self, tuned_qpl):
        """A tuned QPL head must not be filed under an untuned head's name.

        This was a strict xfail. ``compute_qpl`` built ``method_qpl = "qpl_" + head`` and
        ignored the ``model="qpl_opt"`` that ``compute_qpl_opt`` hands it -- the same
        shadowing as ``compute_pqk``, since the parameter had already been rebound to the
        fitted estimator -- so a tuned run wrote ``results_qpl_rf``, ``results_qpl_svc``
        and four more: the exact keys an untuned run writes. It is
        ``method_qpl = f"{model}_{method}"`` now, so an untuned run writes ``qpl_<head>``
        and a tuned one ``qpl_opt_<head>``.

        While this was a pin the predicate had to be the *property* rather than the
        spelling, because the natural fix yields ``qpl_opt_rf``, which does not end in
        ``_opt``: a pin spelt ``endswith("_opt")`` would have gone on failing after the
        bug was fixed and could never have turned XPASS to force its own removal. That
        caution is spent -- the fix has landed and the spelling is measured -- so the
        collision-freedom the reason string named is asserted first, for the pointed
        failure message it gives if the shadowing ever returns, and then the emitted label
        set exactly. Both are reachable in that order and neither is spare: a reversion
        collides with all six untuned keys, and a head dropped from the run collides with
        none while failing the set. The exactness of the set earns its keep downstream --
        ``qc_winner_finder`` recognises a quantum row by the lower-cased first
        ``_``-token, so ``qpl_opt_rf`` still lands in qml_winners.csv while a prefix
        reordered to ``opt_qpl_rf`` would silently stop -- so the prefix needs no separate
        assertion of its own, which would only restate the set.

        The ``pqk`` sibling above needed none of this care: one head, no head name to
        place, so ``results_pqk_opt`` was the only spelling a fix could produce.
        """
        rows = results_rows(tuned_qpl)
        labels = {name[len("results_"):] for name in rows}
        assert labels, "no results_ columns at all, so there is nothing to label"

        indistinguishable = sorted(labels & UNTUNED_QPL_LABELS)
        assert not indistinguishable, (
            f"{indistinguishable} are the exact keys an untuned QPL run writes, so a "
            f"whole hyperparameter search leaves no trace in ModelResults.csv"
        )
        assert labels == {f"qpl_opt_{head}" for head in DEFAULT_QPL_HEADS}, (
            f"a tuned QPL run reported {sorted(labels)}. compute_qpl_opt passes "
            f"model='qpl_opt' and compute_qpl appends the head to it, so each of the six "
            f"default heads must arrive as 'qpl_opt_<head>'"
        )
        for label in sorted(labels):
            row = rows[f"results_{label}"]
            assert row["model"] == label, (
                f"results_{label} holds a row whose own 'model' field says "
                f"{row['model']!r}; that field is what lands in ModelResults.csv and in "
                f"QuantumSage's training data, and qc_winner_finder reads it too"
            )
            for column in ("y_test_", "y_predicted_"):
                assert f"{column}{label}" in tuned_qpl, (
                    f"{column}{label} is missing, so the three columns of one head "
                    f"disagree about its name: {sorted(tuned_qpl)}"
                )


@pytest.fixture(scope="module")
def tuned_qpl(tmp_path_factory):
    """One tuned QPL dispatch, shared by the two tests that read it.

    It costs about twenty seconds even at this size, because ``compute_qpl_opt`` takes
    no ``classical_models`` argument, so every trial and the final fit search all six
    heads (each a 40-candidate RandomizedSearchCV). Both consumers used to be marked
    ``slow`` for that reason, and that is deliberately no longer so:
    ``-m 'not slow and not requires_quantum'`` is the addopts default and what CI runs, so
    a ``slow`` mark would have kept these two out of every default run. That mattered
    while one of them was a strict xfail -- a pin never selected can neither report its bug
    nor turn XPASS when the bug is fixed -- and it matters just as much now that the pin is
    a positive assertion, because a label regression nothing selects is a label regression
    nothing catches. Twenty seconds is what it costs to be in the tier that notices.
    """
    scratch = tmp_path_factory.mktemp("qpl_dispatch")
    X_train, X_test, y_train, y_test = _quantum_dataset()
    args = {
        "backend": "simulator",
        "shots": 64,
        "seed": 7,
        "q_seed": 7,
        "n_jobs": 1,
        "grid_search": True,
        "tune_quantum": True,
        "n_trials_quantum": 2,
        "model": ["qpl"],
        "qpl_projection_dir": str(scratch / "qpl_projections"),
        "gridsearch_qpl_args": {"encoding": ["Z"], "reps": [1]},
    }
    return model_run(X_train, X_test, y_train, y_test, "qpl-dispatch", args)


# --------------------------------------------------------------------------------------
# 'BestParams_Tuned' is written per row, by what produced that row's parameters
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mixed_run(tmp_path_factory):
    """``grid_search: True`` with ``tune_quantum`` left at its default of off.

    The documented configuration for anyone who wants tuned classical baselines beside
    quantum models at their configured hyperparameters -- and therefore the one run whose
    frame has to carry both parameter columns at once, one model having been searched and
    the other not.
    """
    scratch = tmp_path_factory.mktemp("mixed_dispatch")
    X_train, X_test, y_train, y_test = _quantum_dataset()
    args = {
        "backend": "simulator",
        "shots": 64,
        "seed": 7,
        "q_seed": 7,
        "n_jobs": 1,
        "grid_search": True,
        "model": ["dt", "qsvc"],
        "pqk_projection_dir": str(scratch / "pqk_projections"),
        "gridsearch_dt_args": {"max_depth": [2, 3]},
        "qsvc_args": {"reps": 1},
    }
    return model_run(X_train, X_test, y_train, y_test, "mixed", args)


class TestTheParameterColumnIsNamedPerRow:
    """A ``grid_search: True`` run tunes the classical models and not the quantum ones.

    That part was always deliberate and is asserted first. What used to follow from it was
    not: the column name came from ``args['grid_search']`` in
    ``qbiocode/evaluation/model_evaluation.py``, a run-wide flag, so the untuned quantum
    row filed its feature-map and kernel *defaults* under ``BestParams_Tuned`` -- a column
    whose name asserts a search that never ran. Both readers of these frames,
    ``qbiocode/apps/sage/sage.py`` and ``qbiocode/utils/qc_winner_finder.py``, look for
    that name before ``Model_Parameters``, so whatever it held was what a downstream reader
    believed had been searched.

    ``modeleval`` decides it per row now, from an explicit ``tuned=`` keyword where the
    caller knows (the nine classical ``compute_<m>_opt`` functions, whose ``model``
    argument is left at a display name like ``'Decision Tree'`` on a direct call and so
    carries no marker to infer from) and otherwise from ``_was_tuned``'s fallback on the
    ``_opt`` suffix. The consequence is what the last test below asserts and what the
    strict xfail it replaces could only describe: one frame carrying both columns.
    """

    def test_only_the_classical_model_is_tuned(self, mixed_run):
        rows = results_rows(mixed_run)
        assert set(rows) == {"results_dt_opt", "results_qsvc"}, (
            f"grid_search without tune_quantum must tune dt and leave qsvc alone; "
            f"got {sorted(rows)}"
        )
        assert rows["results_dt_opt"]["model"] == "dt_opt"
        assert rows["results_qsvc"]["model"] == "qsvc"

    def test_the_tuned_classical_row_reports_what_was_actually_searched(self, mixed_run):
        """The control for the test below: on the row that *was* tuned, the column holds
        the search result and nothing else.

        Worth keeping separate from the column-naming test, because it is the half that
        would still hold if ``_was_tuned`` answered ``True`` for everything -- so a
        failure here and a failure there mean different things.
        """
        tuned = results_rows(mixed_run)["results_dt_opt"]["BestParams_Tuned"]
        assert set(tuned) == {"max_depth"}, (
            f"dt_opt searched only max_depth but reports {sorted(tuned)}"
        )
        assert tuned["max_depth"] in (2, 3), tuned["max_depth"]

    def test_one_run_carries_both_parameter_columns_one_per_row(self, mixed_run):
        """Two rows of one frame, two different parameter columns, neither claiming the
        other's.

        This was a strict xfail asserting only half of it -- that the untuned ``qsvc`` row
        does not report under ``BestParams_Tuned`` -- because that was all the old
        behaviour let anyone describe. ``modeleval`` named the column from
        ``args['grid_search']``, so in this run, the documented one for tuned classical
        baselines beside untuned quantum models, ``qsvc`` filed its feature-map and kernel
        defaults under a column whose name claims a search; and both readers of these
        frames look for that name before ``Model_Parameters``.

        The mixed run is the valuable case and coexistence is the point: no answer
        computed once per run can put both columns in one frame, *whichever way* it is
        read. So that is asserted first and the two rows are then taken one at a time --
        an order chosen so that each of the three can be the one that fails. Coexistence
        alone is what a reversion to a run-wide flag breaks, and it is all such a
        reversion breaks in the direction that says "tuned" for everything; the per-row
        assertions are what catch the columns being present but swapped, which satisfies
        coexistence exactly. Asserted the other way round, the summary could never fail
        first and so could never fail at all.

        Which two rows exist is not this test's subject: it asks for them by the names
        ``test_only_the_classical_model_is_tuned`` establishes, so a regression in the
        dispatch itself is reported there instead of here.
        """
        rows = results_rows(mixed_run)

        present = {
            column
            for row in rows.values()
            for column in ("BestParams_Tuned", "Model_Parameters")
            if column in row
        }
        assert present == {"BestParams_Tuned", "Model_Parameters"}, (
            f"the rows of this one run carry only {sorted(present)}. Both names must "
            f"appear: tuning is not run-wide -- quantum models stay untuned unless "
            f"tune_quantum is set too -- so the column is a per-row fact, and any answer "
            f"computed once per run gives every row in the frame the same column"
        )

        tuned_row = rows["results_dt_opt"]
        untuned_row = rows["results_qsvc"]
        assert "BestParams_Tuned" in tuned_row and "Model_Parameters" not in tuned_row, (
            f"dt_opt is the product of a real Optuna search, so its parameters belong "
            f"under BestParams_Tuned and that column alone; it reports {sorted(tuned_row)}"
        )
        assert "Model_Parameters" in untuned_row and "BestParams_Tuned" not in untuned_row, (
            f"qsvc ran at its configured hyperparameters -- no search happened -- so its "
            f"parameters belong under Model_Parameters and that column alone; it reports "
            f"{sorted(untuned_row)}"
        )

        parameters = untuned_row["Model_Parameters"]
        assert {"feature_map", "quantum_kernel"} <= set(parameters), (
            f"qsvc's parameters are {parameters}, which is not the record compute_qsvc "
            f"writes for itself; the column split is only meaningful because what lands "
            f"there is the untuned run's own feature-map and kernel description rather "
            f"than any search result"
        )
