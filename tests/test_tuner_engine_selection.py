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

"""``tuner: grid`` has to really reach ``GridSearchCV``, and nothing says when it does not.

Every classical ``compute_*_opt`` learner picks its search engine with the same block::

    if tuner == "grid":
        search = GridSearchCV(Estimator(...), param_grid=build_param_grid(...), cv=cv)
        search.fit(X_train, y_train)
        best_params = search.best_params_
    else:
        best_params = run_study(Estimator, build_search_space(...), ..., n_trials=n_trials)

That block is written out nine times, once per learner, and the ``else`` swallows
everything the ``if`` fails to match. Any of the ordinary ways one copy goes wrong --
``tuner`` left out of the signature so the parameter arrives at its ``"optuna"``
default, the comparison made against the wrong string, the keyword dropped from the
dispatcher's call -- puts the run on the Optuna path while the config says ``grid``.
The result is not an error and does not look like one: the frame carries a finite
accuracy, a plausible time, and a ``BestParams_Tuned`` dict of values drawn from the
very lists the grid would have enumerated. There is nothing in a results frame that
identifies which engine produced it.

``tests/test_grid_search_partial.py`` does run all nine learners under both engines,
which is why the substitution is easy to believe is covered. It is not: that file
asserts the accuracy is finite and that the winning value is one of the values that
were searched, and both hold exactly as well when Optuna answered a request for the
grid. Only ``svc`` (at the ``build_param_grid`` unit level) and ``catboost`` (through
the real function) had an assertion that could tell the engines apart.

The property that can is what the two engines are able to *express*. A grid can only
enumerate, so a hyperparameter written as a ``{low, high}`` range has to be REFUSED
under ``tuner: grid`` and SAMPLED under ``tuner: optuna``. Neither half of that
asymmetry can be produced by the other engine, and it is established here for the
seven learners that lacked it: dt, lr, nb, rf, xgb, mlp, tabpfn -- and for ``svc``,
whose existing check is against the ``build_param_grid`` helper rather than against
``compute_svc_opt``, so it says nothing about that learner's own wiring.

That asymmetry is then applied twice over, because it answers two separate questions.
Handed ``tuner="grid"`` directly, it says the learner's own ``if`` works. Driven from
``args['tuner']`` through ``model_run``, it says the config value survives the one hop in
between -- ``tuner=args.get("tuner", "optuna")`` on the classical branch of the
dispatcher -- which is the third failure mode listed above and the only one a direct call
cannot see: a dispatcher that forwards nothing leaves every learner on its own
``"optuna"`` default while the config still reads ``grid``, and every by-hand ``tuner=``
keeps passing. The default is pinned from the same side, so the hop cannot be wired shut
either: with no ``tuner`` key the range must still be sampled. A zero trial budget --
meaningless to an exhaustive sweep, fatal to a sampled one -- says the same thing a
second way, through a mechanism that does not depend on the grid refusing a range at all.

The rest of the file follows the numbers the dispatcher is supposed to hand to the
search, none of which were checked at the ``model_run`` boundary: ``args['n_trials']``
for the classical learners, the deliberately separate ``args['n_trials_quantum']`` and
``args['validation_split']`` for the quantum ones, and ``args['seed']`` for the TPE
sampler. ``test_optuna_tuning.py`` shows ``run_study`` honours a seed handed to it by
hand; that ``model_run`` supplies one at all -- through ``_seeded_kwargs`` for the
eight learners that take a ``random_state``, and by a second route for naive Bayes,
which takes none -- is what decides whether a tuned run repeats.
"""

import importlib
import sys
import warnings

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Importing anything from the package runs `qbiocode/__init__.py`, which orders the
# three vendored OpenMP runtimes before xgboost or torch can map one of their own.
from qbiocode.evaluation.model_run import model_run
from qbiocode.learning._tuning import build_search_space, run_function_study, run_study

# --------------------------------------------------------------------------------------
# One range per learner, chosen so the engines cannot agree about it
# --------------------------------------------------------------------------------------
#
# Each is a *float* range even where the hyperparameter also accepts an int, so the
# value Optuna returns can be required to fall strictly inside the open interval: an
# enumerating engine can only ever return a value that was written down somewhere, and
# nothing here writes down 0.0650076939. `nb` is the one learner whose signature
# defaults its hyperparameter to a populated list rather than None, so the range has to
# displace that default -- which is itself worth exercising.
#
# `svc` is included even though `test_optuna_tuning.test_a_range_is_rejected_by_the_grid_tuner`
# already refuses a range for it: that test calls `build_param_grid("svc", ...)` directly,
# which says what the helper does and nothing about whether `compute_svc_opt` reaches it.
# `catboost` is the one learner deliberately left out: in test_catboost_tabpfn.py,
# TestCatBoostTuning.test_a_range_is_accepted_by_optuna_and_refused_by_the_grid already
# makes exactly this pair of calls through the real function.
RANGE_BLOCKS = [
    ("dt", "min_samples_leaf", {"low": 0.05, "high": 0.3}),
    ("lr", "C", {"low": 0.01, "high": 10.0, "log": True}),
    ("nb", "var_smoothing", {"low": 1e-10, "high": 1e-2, "log": True}),
    ("rf", "max_features", {"low": 0.2, "high": 0.9}),
    ("svc", "C", {"low": 0.01, "high": 10.0, "log": True}),
    ("xgb", "learning_rate", {"low": 0.01, "high": 0.5, "log": True}),
    ("mlp", "alpha", {"low": 1e-5, "high": 1e-1, "log": True}),
    ("tabpfn", "softmax_temperature", {"low": 0.5, "high": 1.5}),
]

RANGE_IDS = [model for model, _, _ in RANGE_BLOCKS]


def learner_module(model):
    """The ``compute_<model>`` *module*.

    ``qbiocode.learning.__init__`` re-exports each ``compute_<model>`` function under
    the module's own name, so ``qbiocode.learning.compute_dt`` is the function and the
    module has to be taken out of ``sys.modules``. The tests below patch names *inside*
    these modules, which is only possible with the module object.
    """
    importlib.import_module(f"qbiocode.learning.compute_{model}")
    return sys.modules[f"qbiocode.learning.compute_{model}"]


def opt_function(model):
    """The ``compute_<model>_opt`` callable, reached through its own module."""
    return getattr(learner_module(model), f"compute_{model}_opt")


def metrics_of(frame):
    """The single metrics dict out of a ``modeleval`` frame.

    The column is ``results_<display name>`` -- ``'Random Forest'`` rather than the
    ``rf`` dispatch key -- so it is found rather than assumed.
    """
    columns = [c for c in frame.columns if c.startswith("results_")]
    assert len(columns) == 1, f"expected one results_ column, got {columns}"
    return frame[columns[0]].iloc[0]


def recording(original, log):
    """A pass-through spy: record the keyword arguments, then do the real work.

    Nothing is faked. The search still runs, the estimators are still fitted on real
    data, and the caller still gets the real answer -- the wrapper only makes visible
    which budget and which seed the dispatcher chose, neither of which survives into
    the results frame.
    """

    def recorder(*args, **kwargs):
        log.append(kwargs)
        return original(*args, **kwargs)

    return recorder


@pytest.fixture
def data():
    """A small separable binary problem; the same shape the other tuning tests use."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    return X[:40], X[40:], y[:40], y[40:]


@pytest.fixture
def quantum_data():
    """Two features and twenty-four rows: a two-qubit circuit is cheap to simulate."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(32, 2))
    y = (X[:, 0] > 0).astype(int)
    return X[:24], X[24:], y[:24], y[24:]


@pytest.fixture
def sim_args():
    """Config for a tuned quantum run on the simulator, where a trial costs only CPU."""
    return {
        "backend": "simulator",
        "shots": 64,
        "seed": 5,
        "n_jobs": 1,
        "grid_search": True,
        "tune_quantum": True,
    }


TUNED = {"seed": 7, "grid_search": True}

#: Everything ``model_run`` needs for a tuned classical run except the model, its
#: ``gridsearch_<model>_args`` block, and ``tuner`` -- the variable under test, which each
#: dispatcher-mediated test below sets or deliberately leaves out. ``n_jobs: 1`` keeps
#: joblib in this process, so an engine's own refusal arrives as itself rather than as a
#: worker's wrapper naming nothing. ``cross_validation`` -- not ``cv`` -- is the config
#: key the dispatcher reads.
DISPATCHED = {"seed": 7, "n_jobs": 1, "grid_search": True, "cross_validation": 3}


# ======================================================================================
# The asymmetry that identifies the engine
# ======================================================================================


class TestEachEngineIsReallyReached:
    """A range is expressible to one engine and not the other, per learner.

    Used twice on purpose: with ``tuner=`` handed to the learner, which pins the
    learner's own branch, and with ``args['tuner']`` through ``model_run``, which pins
    the single line that carries a user's choice to that branch.
    """

    @pytest.mark.parametrize("model,name,spec", RANGE_BLOCKS, ids=RANGE_IDS)
    def test_the_grid_engine_refuses_a_range_it_could_never_enumerate(
        self, model, name, spec, data
    ):
        """Reaching this error is the proof that ``tuner='grid'`` took the grid branch.

        ``build_param_grid`` is called from inside the ``if tuner == "grid"`` block and
        from nowhere else, so its refusal cannot be produced by the Optuna path -- which
        accepts the identical block, as the next test shows. A learner whose ``tuner``
        keyword never arrives would sample the range and return a happy frame here.
        """
        X_train, X_test, y_train, y_test = data
        with pytest.raises(ValueError) as excinfo:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                opt_function(model)(
                    X_train, X_test, y_train, y_test, TUNED,
                    cv=3, tuner="grid", n_trials=3, **{name: spec},
                )
        message = str(excinfo.value)
        assert f"{model!r}" in message, f"the message must name the model: {message}"
        assert f"{name!r}" in message, f"the message must name the hyperparameter: {message}"
        assert "tuner: optuna" in message, (
            f"the message has to name the engine that would accept a range, so the "
            f"user can act on it: {message}"
        )

    @pytest.mark.parametrize("model,name,spec", RANGE_BLOCKS, ids=RANGE_IDS)
    def test_the_optuna_engine_samples_the_very_range_the_grid_refused(
        self, model, name, spec, data
    ):
        """The other half: the default engine sends the same block to ``run_study``.

        The winning value is required to lie *strictly* inside the open interval. An
        enumerating engine can only return a value that appears in some list, and no
        list here contains an arbitrary interior float, so a value like 0.0650076939 is
        evidence that a continuous distribution was sampled rather than a grid walked.
        """
        X_train, X_test, y_train, y_test = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = opt_function(model)(
                X_train, X_test, y_train, y_test, TUNED,
                cv=3, n_trials=4, **{name: spec},
            )
        best = metrics_of(frame)["BestParams_Tuned"]
        assert set(best) == {name}, (
            f"{name!r} was the only hyperparameter supplied, so it must be the only one "
            f"searched; got {sorted(best)}"
        )
        value = best[name]
        assert isinstance(value, float), (
            f"a float range must yield a float, not {value!r} of type {type(value).__name__}"
        )
        assert spec["low"] < value < spec["high"], (
            f"{name}={value!r} is not strictly inside the configured range "
            f"({spec['low']}, {spec['high']}), so it did not come from a continuous draw"
        )

    def test_a_zero_trial_budget_is_an_optuna_idea_the_grid_has_no_use_for(self, data):
        """A second, independent way to tell the engines apart on one learner.

        ``n_trials`` bounds a sampled search and means nothing to an exhaustive one, so
        the grid must fit every point of a zero-budget request while Optuna refuses it.
        A learner running Optuna behind ``tuner: grid`` would raise on the first call
        here instead of returning a result.
        """
        X_train, X_test, y_train, y_test = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = opt_function("dt")(
                X_train, X_test, y_train, y_test, TUNED,
                cv=3, tuner="grid", n_trials=0, max_depth=[2, 3],
            )
        assert metrics_of(frame)["BestParams_Tuned"]["max_depth"] in (2, 3)

        with pytest.raises(ValueError, match="at least one trial"):
            opt_function("dt")(
                X_train, X_test, y_train, y_test, TUNED,
                cv=3, tuner="optuna", n_trials=0, max_depth=[2, 3],
            )

    def test_an_unrecognised_tuner_falls_through_to_optuna_inside_the_learner(self, data):
        """Why the dispatcher's own guard is load-bearing and must not be deleted.

        The ``else`` branch is reached by anything that is not the string ``"grid"``, so
        a typo does not raise here -- it silently runs Optuna. That is deliberate: the
        only validation of ``args['tuner']`` lives in ``model_run``
        (``test_optuna_tuning.test_model_run_rejects_an_unknown_tuner``). This test pins
        the fall-through so that guard can never be removed as redundant, and it uses
        the range asymmetry to say which engine actually ran rather than inferring it.
        """
        X_train, X_test, y_train, y_test = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = opt_function("dt")(
                X_train, X_test, y_train, y_test, TUNED,
                cv=3, tuner="gird", n_trials=3,
                min_samples_leaf={"low": 0.05, "high": 0.3},
            )
        best = metrics_of(frame)["BestParams_Tuned"]
        assert 0.05 < best["min_samples_leaf"] < 0.3, (
            "a misspelt tuner sampled the range, i.e. it ran Optuna; if this ever "
            "raises instead, the learners have grown their own validation and the "
            "docstring above is out of date"
        )

    # -- the same asymmetry, driven from `args` instead of by hand -----------------------

    @pytest.mark.parametrize("model,name,spec", RANGE_BLOCKS, ids=RANGE_IDS)
    def test_a_config_that_names_the_grid_gets_the_grid_and_not_the_sampler(
        self, model, name, spec, data
    ):
        """The hop the three tests above skip: ``args['tuner']`` -> the learner's keyword.

        Those tests hand ``tuner="grid"`` to ``compute_<model>_opt`` themselves, so they
        say the learner's own ``if`` works and nothing at all about whether a user's
        ``tuner: grid`` ever reaches it. Between a config and that ``if`` lies exactly
        one line -- ``model_run``'s ``tuner=args.get("tuner", "optuna")`` -- and it is the
        third failure mode this module's docstring names. Delete it and every classical
        learner falls back to its own ``"optuna"`` default while the config still reads
        ``grid``; the range is sampled, the frame carries a finite accuracy, and
        ``BestParams_Tuned`` is indistinguishable from a grid's answer. Every by-hand
        ``tuner=`` in this file keeps passing, which is precisely why the substitution
        needs asserting from this side.

        The refusal is raised while ``GridSearchCV`` is being constructed, before a
        single fit, so the sweep is free per learner -- and each case pins one more
        ``gridsearch_<model>_args`` block onto its own learner's grid branch, which is
        the rest of the journey the direct calls also miss.
        """
        X_train, X_test, y_train, y_test = data
        args = {
            **DISPATCHED,
            "model": [model],
            "tuner": "grid",
            f"gridsearch_{model}_args": {name: spec},
        }
        with pytest.raises(ValueError) as excinfo:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_run(X_train, X_test, y_train, y_test, "key", args)
        message = str(excinfo.value)
        assert f"{model!r}" in message, f"the message must name the model: {message}"
        assert f"{name!r}" in message, f"the message must name the hyperparameter: {message}"
        assert "tuner: optuna" in message, (
            f"a range did reach the grid, which is what this test wants, but the message "
            f"does not name the engine that would have accepted it: {message}"
        )

    @pytest.mark.parametrize("tuner", [None, "optuna"], ids=["absent", "explicit"])
    def test_a_config_that_does_not_name_the_grid_gets_the_sampler(self, tuner, data):
        """The other direction across the same hop, which is what keeps it a choice.

        A dispatcher that forwarded ``"grid"`` unconditionally, or whose default flipped
        to it, is the mirror image of the bug above and just as quiet in a results frame.
        The range is the witness again: it has to be *sampled*, so an absent ``tuner``
        key must mean the documented ``optuna`` and an explicit ``optuna`` must be
        honoured rather than overridden. Either failure surfaces as the grid's refusal
        escaping this test instead of a result; the interior-float check below is for the
        subtler case of a grid that answered from a list.
        """
        X_train, X_test, y_train, y_test = data
        args = {
            **DISPATCHED,
            "model": ["dt"],
            "n_trials": 4,
            "gridsearch_dt_args": {"min_samples_leaf": {"low": 0.05, "high": 0.3}},
        }
        if tuner is not None:
            args["tuner"] = tuner
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = model_run(X_train, X_test, y_train, y_test, "key", args)
        best = out["results_dt_opt"][0]["BestParams_Tuned"]
        assert 0.05 < best["min_samples_leaf"] < 0.3, (
            f"args['tuner']={tuner!r} had to be sampled, but min_samples_leaf came back "
            f"as {best['min_samples_leaf']!r}, which is not an interior draw"
        )

    def test_a_zero_budget_divides_the_engines_the_config_chose_between(self, data):
        """A second witness for the same hop, independent of ``build_param_grid``.

        The two tests above both turn on the grid's refusal of a range, so a day when
        that refusal is softened -- a grid that quietly discretises a range, say -- is a
        day the dispatcher's hop silently loses its cover again. ``n_trials`` is the
        other thing one engine cannot make sense of: an exhaustive sweep has no budget to
        spend, so the zero that stops an Optuna run has to be beneath a grid one's
        notice. The two configs here differ in ``args['tuner']`` and in nothing else,
        and the grid leg is the one place in this file where a config's ``grid`` request
        produces a tuned result rather than an error -- so a dispatcher that stopped
        forwarding the keyword fails it by raising, not by returning something wrong.

        The Optuna leg's refusal is the one
        ``TestTheDispatcherSuppliesTheBudgetAndTheSeed`` checks in its own terms; what is
        new is that it is now the *contrast*, and that the same zero is required not to
        raise on the other engine.
        """
        X_train, X_test, y_train, y_test = data
        args = {
            **DISPATCHED,
            "model": ["dt"],
            "n_trials": 0,
            "gridsearch_dt_args": {"max_depth": [2, 3]},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = model_run(
                X_train, X_test, y_train, y_test, "key", {**args, "tuner": "grid"}
            )
        assert out["results_dt_opt"][0]["BestParams_Tuned"]["max_depth"] in (2, 3), (
            "a grid fits every point of its sweep whatever n_trials says; a run that "
            "took notice of the budget here was not the grid"
        )

        with pytest.raises(ValueError, match="at least one trial"):
            model_run(X_train, X_test, y_train, y_test, "key", {**args, "tuner": "optuna"})


# ======================================================================================
# The budget: exactly the trials that were asked for
# ======================================================================================


class _FitCountingTree(DecisionTreeClassifier):
    """A real decision tree that keeps a tally of how many times it has been fitted.

    Not a mock: every ``fit`` is the genuine sklearn implementation on genuine data, and
    the tuned result is the one the real estimator earns. The subclass exists only
    because the number of trials a study spent is not observable from its return value,
    and ``cross_val_score`` clones the class it was handed, so a class-level counter
    sees exactly ``n_trials * cv`` fits and nothing else.
    """

    fits = 0

    def fit(self, X, y, *args, **kwargs):
        _FitCountingTree.fits += 1
        return super().fit(X, y, *args, **kwargs)


class TestTheTrialBudget:
    """``n_trials`` is spent, in full, and not exceeded."""

    @pytest.mark.parametrize("n_trials,cv", [(7, 3), (2, 4), (5, 2)])
    def test_a_study_over_a_range_runs_exactly_the_trials_it_was_given(self, n_trials, cv):
        """The budget is the whole stopping rule when the space has no finite size.

        A range is infinite, so nothing caps the search the way ``_finite_size`` caps a
        categorical one (which ``test_optuna_tuning`` already covers). Counting fits is
        the exact measure -- one trial is one ``cross_val_score``, so ``n_trials * cv``
        fits -- and it distinguishes a budget that was honoured from one that was
        ignored, doubled, or read off Optuna's own default.
        """
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 5))
        y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
        space = build_search_space("dt", {"min_samples_leaf": {"low": 0.05, "high": 0.3}})

        _FitCountingTree.fits = 0
        best = run_study(_FitCountingTree, space, X, y, cv=cv, n_trials=n_trials, seed=0)

        assert _FitCountingTree.fits == n_trials * cv, (
            f"a budget of {n_trials} trials at cv={cv} is {n_trials * cv} fits, but the "
            f"estimator was fitted {_FitCountingTree.fits} times"
        )
        assert 0.05 < best["min_samples_leaf"] < 0.3

    def test_the_two_paths_spell_the_model_differently_in_the_same_budget_error(self):
        """One helper, two naming conventions -- pinned as it stands, not xfailed.

        ``_validate_budget`` is shared, but ``run_study`` is handed
        ``estimator_cls.__name__`` (``_tuning.py:291``) while ``run_function_study`` is
        handed the dispatch key (``_tuning.py:449``). So a classical zero budget reports
        ``Tuning 'RandomForestClassifier'`` and a quantum one ``Tuning 'qsvc'``, for the
        same mistake in the same config.

        Pinned rather than ``xfail(strict=True)`` because neither spelling misleads: both
        identify the model a user can find in ``args['model']``, and both carry the part
        that matters -- the offending value and the two ways out. Unifying them would be
        an improvement, not a bug fix, and this test is what would notice it happening
        and make the choice explicit.
        """
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 4))
        y = (X[:, 0] > 0).astype(int)
        with pytest.raises(ValueError) as classical:
            run_study(
                RandomForestClassifier,
                build_search_space("rf", {"max_depth": [2, 3]}),
                X, y, cv=3, n_trials=0, seed=0,
            )
        with pytest.raises(ValueError) as quantum:
            run_function_study(
                learner_module("qsvc").compute_qsvc,
                build_search_space("qsvc", {"reps": [1]}),
                X,
                y,
                {"backend": "simulator", "shots": 64},
                model="qsvc",
                n_trials=0,
                seed=0,
            )

        classical_message, quantum_message = str(classical.value), str(quantum.value)
        # The inconsistency itself.
        assert "Tuning 'RandomForestClassifier'" in classical_message, classical_message
        assert "Tuning 'qsvc'" in quantum_message, quantum_message
        # The part a user acts on, which both must carry.
        for message in (classical_message, quantum_message):
            assert "n_trials is 0" in message, f"the offending value is missing: {message}"
            assert "positive integer" in message, f"no remedy offered: {message}"
            assert "grid_search: False" in message, f"no way out offered: {message}"


# ======================================================================================
# What the dispatcher hands the search
# ======================================================================================


class TestTheDispatcherSuppliesTheBudgetAndTheSeed:
    """``model_run`` is the only place these numbers are chosen, and it was untested."""

    def test_the_classical_budget_comes_from_n_trials_and_defaults_to_fifty(
        self, data, monkeypatch
    ):
        """Documented as "Trial budget for the Optuna tuner, default 50"; now checked.

        The forwarded value is read rather than the number of trials that ran, because a
        finite space is capped to its own size on the way in -- ``max_depth: [2, 3]``
        would run two trials whatever the budget said, which is exactly how a dropped
        keyword could hide.
        """
        X_train, X_test, y_train, y_test = data
        log = []
        module = learner_module("dt")
        monkeypatch.setattr(module, "run_study", recording(module.run_study, log))

        args = {
            "model": ["dt"],
            "seed": 11,
            "n_jobs": 1,
            "grid_search": True,
            "cross_validation": 3,
            "n_trials": 4,
            "gridsearch_dt_args": {"max_depth": [2, 3]},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_run(X_train, X_test, y_train, y_test, "key", args)
        assert [entry["n_trials"] for entry in log] == [4]
        assert [entry["cv"] for entry in log] == [3], "cross_validation must reach the study"

        log.clear()
        del args["n_trials"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_run(X_train, X_test, y_train, y_test, "key", args)
        assert [entry["n_trials"] for entry in log] == [50], (
            "an absent args['n_trials'] must become the documented default of 50, not "
            "Optuna's own default or None"
        )

    def test_the_quantum_budget_is_n_trials_quantum_and_never_n_trials(
        self, quantum_data, sim_args, monkeypatch
    ):
        """The two budgets are separate on purpose and must not leak into each other.

        A quantum trial simulates an n-by-n fidelity kernel, so the classical default of
        50 would be tens of ordinary runs of the model. The quantum default is 10, and a
        config that raises ``n_trials`` for the classical learners must leave the quantum
        budget alone -- here ``n_trials: 99`` is set and the quantum study must still
        receive 10.
        """
        X_train, X_test, y_train, y_test = quantum_data
        log = []
        module = learner_module("qsvc")
        monkeypatch.setattr(
            module, "run_function_study", recording(module.run_function_study, log)
        )

        args = {
            **sim_args,
            "model": ["qsvc"],
            "n_trials": 99,
            "gridsearch_qsvc_args": {"encoding": ["Z"], "reps": [1]},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = model_run(X_train, X_test, y_train, y_test, "key", args)
        assert "results_qsvc_opt" in out
        assert [entry["n_trials"] for entry in log] == [10], (
            "args['n_trials'] is the classical budget; the quantum study must take "
            "args['n_trials_quantum'], whose default is 10"
        )

    def test_the_quantum_budget_and_validation_split_are_forwarded_when_set(
        self, quantum_data, sim_args, monkeypatch
    ):
        """``validation_split`` exists only on the quantum path and only in ``args``.

        A quantum candidate is scored on one stratified holdout rather than k folds, so
        the fraction held back is a config knob of its own. It has never been read
        through ``model_run`` in a test, and a dropped keyword would silently restore
        0.25 -- a change of score with no change of config.
        """
        X_train, X_test, y_train, y_test = quantum_data
        log = []
        module = learner_module("qsvc")
        monkeypatch.setattr(
            module, "run_function_study", recording(module.run_function_study, log)
        )

        args = {
            **sim_args,
            "model": ["qsvc"],
            "n_trials_quantum": 3,
            "validation_split": 0.4,
            "gridsearch_qsvc_args": {"encoding": ["Z"], "reps": [1]},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_run(X_train, X_test, y_train, y_test, "key", args)
        assert [entry["n_trials"] for entry in log] == [3]
        assert [entry["validation_split"] for entry in log] == [0.4]
        assert [entry["seed"] for entry in log] == [5], "args['seed'] seeds the inner split too"

    def test_a_zero_budget_stops_the_run_at_whichever_budget_was_zeroed(
        self, data, quantum_data, sim_args
    ):
        """The user-facing half of the budget check: through ``model_run``, in ``args``.

        The two budgets are validated by the same helper but reached by different keys,
        so a zero has to be refused twice over -- ``n_trials`` for the classical
        learners, ``n_trials_quantum`` for the quantum ones -- and the error has to
        survive being raised inside a joblib worker rather than arriving as a
        joblib wrapper naming nothing.
        """
        X_train, X_test, y_train, y_test = data
        with pytest.raises(ValueError) as classical:
            model_run(
                X_train, X_test, y_train, y_test, "key",
                {
                    "model": ["rf"], "seed": 3, "n_jobs": 1, "grid_search": True,
                    "cross_validation": 3, "n_trials": 0,
                    "gridsearch_rf_args": {"n_estimators": [5, 10]},
                },
            )
        assert "at least one trial" in str(classical.value)
        assert "n_trials is 0" in str(classical.value)

        qX_train, qX_test, qy_train, qy_test = quantum_data
        with pytest.raises(ValueError) as quantum:
            model_run(
                qX_train, qX_test, qy_train, qy_test, "key",
                {
                    **sim_args, "model": ["qsvc"], "n_trials_quantum": 0,
                    "gridsearch_qsvc_args": {"encoding": ["Z"], "reps": [1]},
                },
            )
        assert "at least one trial" in str(quantum.value)
        assert "n_trials is 0" in str(quantum.value), (
            "a zeroed args['n_trials_quantum'] must be reported as the trial budget it "
            "is, not left to surface as 'every tuning trial failed'"
        )

    def test_the_seed_reaches_the_sampler_from_args(self, data, monkeypatch):
        """``_seeded_kwargs`` fills ``random_state``, which becomes the sampler seed.

        The journey is ``args['seed']`` -> ``random_state=`` on the ``_opt`` call ->
        ``seed=`` on ``run_study`` -> ``TPESampler(seed=...)``. Only the last hop was
        tested before; if any earlier one broke, the sampler would draw from OS entropy
        and two runs of one config would disagree with nothing to say why.
        """
        X_train, X_test, y_train, y_test = data
        log = []
        module = learner_module("rf")
        monkeypatch.setattr(module, "run_study", recording(module.run_study, log))

        args = {
            "model": ["rf"],
            "seed": 13,
            "n_jobs": 1,
            "grid_search": True,
            "cross_validation": 3,
            "n_trials": 2,
            "gridsearch_rf_args": {"n_estimators": [5, 10]},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_run(X_train, X_test, y_train, y_test, "key", args)
        (entry,) = log
        assert entry["seed"] == 13, f"the sampler seed must be args['seed']: {entry}"
        assert entry["fixed"] == {"random_state": 13}, (
            f"the same seed must also reach the estimator, or the fits themselves stay "
            f"unreproducible: {entry}"
        )

    def test_naive_bayes_gets_its_sampler_seed_from_args_directly(self, data, monkeypatch):
        """The one learner reached by a second route, and the one most likely to rot.

        ``GaussianNB`` has no ``random_state``, so ``_seeded_kwargs`` deliberately gives
        ``compute_nb_opt`` nothing -- it reads ``args['seed']`` itself instead. A range
        over ``var_smoothing`` is unreproducible the moment that read is dropped, and
        every other learner would still pass.
        """
        X_train, X_test, y_train, y_test = data
        log = []
        module = learner_module("nb")
        monkeypatch.setattr(module, "run_study", recording(module.run_study, log))

        args = {
            "model": ["nb"],
            "seed": 23,
            "n_jobs": 1,
            "grid_search": True,
            "cross_validation": 3,
            "n_trials": 3,
            "gridsearch_nb_args": {"var_smoothing": {"low": 1e-10, "high": 1e-2, "log": True}},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_run(X_train, X_test, y_train, y_test, "key", args)
        (entry,) = log
        assert entry["seed"] == 23, (
            f"naive Bayes takes no random_state, so args['seed'] has to be read from the "
            f"config by the learner itself: {entry}"
        )
        assert entry["fixed"] == {}, "GaussianNB has nothing to fix; see compute_nb_opt"

    @pytest.mark.parametrize(
        "model,block",
        [
            ("dt", {"min_samples_leaf": {"low": 0.05, "high": 0.3}}),
            ("nb", {"var_smoothing": {"low": 1e-10, "high": 1e-2, "log": True}}),
            ("lr", {"C": {"low": 0.01, "high": 10.0, "log": True}}),
        ],
    )
    def test_two_tuned_runs_at_one_seed_choose_the_same_hyperparameters(
        self, model, block, data
    ):
        """The behaviour the seed plumbing exists for, checked end to end.

        A continuous range makes this a real test: over a categorical block of two
        values, two runs would often agree by luck. Over an interval the sampler's draws
        are the answer, so equality means one seed produced one sequence of draws. The
        two seeding routes are both represented -- ``nb`` reads ``args['seed']``, the
        other two receive it as ``random_state``.
        """
        X_train, X_test, y_train, y_test = data

        def tuned_params(seed):
            args = {
                "model": [model],
                "seed": seed,
                "n_jobs": 1,
                "grid_search": True,
                "cross_validation": 3,
                "n_trials": 6,
                f"gridsearch_{model}_args": block,
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = model_run(X_train, X_test, y_train, y_test, "key", args)
            return out[f"results_{model}_opt"][0]["BestParams_Tuned"]

        first, second = tuned_params(11), tuned_params(11)
        assert first == second, (
            f"two runs of {model!r} at seed 11 chose {first} and {second}; a tuned run "
            f"is not reproducible if the sampler is not seeded from args['seed']"
        )
        # Not a correctness requirement, only evidence the seed is doing the choosing:
        # a different seed draws a different sequence from the same interval.
        assert tuned_params(12) != first, (
            f"seeds 11 and 12 both chose {first} out of a continuous range, which "
            f"suggests the seed is not reaching the sampler at all"
        )
