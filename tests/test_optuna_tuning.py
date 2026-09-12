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

"""Optuna tuning: the search space the config describes, and the budget it spends.

``GridSearchCV`` fitted every point of the cross product -- 576 combinations for
``gridsearch_rf_args`` as shipped, 2880 fits at ``cross_validation: 5``. Optuna
spends a fixed ``n_trials`` instead and steers them, which only pays off if a
hyperparameter can be given as a *range* rather than a list of decades. So the
config now accepts both shapes, and these tests pin which shape means what.

The old exhaustive path stays reachable through ``tuner: grid``; that it still
searches the same hyperparameters is checked in ``test_grid_search_partial.py``,
which runs every ``_opt`` learner under both engines.
"""

import warnings

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from qbiocode.learning._grid import build_param_grid
from qbiocode.learning._tuning import _Categorical, _Range, build_search_space, run_study


@pytest.fixture
def data():
    """The same small separable problem the grid-search tests use."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


class TestBuildSearchSpace:
    """What each config shape is taken to mean."""

    def test_a_list_is_a_categorical_choice(self):
        """Every config in the tree writes lists, and they must keep their meaning."""
        space = build_search_space("svc", {"kernel": ["linear", "rbf"]})
        assert isinstance(space["kernel"], _Categorical)
        assert space["kernel"].values == ["linear", "rbf"]

    def test_integer_bounds_give_an_integer_range(self):
        """``n_estimators: {low: 10, high: 500}`` must not propose 214.7 trees."""
        space = build_search_space("rf", {"n_estimators": {"low": 10, "high": 500}})
        assert isinstance(space["n_estimators"], _Range)
        assert space["n_estimators"].is_int

    def test_a_float_bound_gives_a_float_range(self):
        space = build_search_space("svc", {"C": {"low": 0.01, "high": 10}})
        assert isinstance(space["C"], _Range)
        assert not space["C"].is_int

    def test_log_and_step_are_carried_through(self):
        space = build_search_space(
            "svc",
            {
                "C": {"low": 1e-3, "high": 1e3, "log": True},
                "coef0": {"low": 0.0, "high": 1.0, "step": 0.25},
            },
        )
        assert space["C"].log is True
        assert space["coef0"].step == 0.25

    def test_unsupplied_hyperparameters_are_dropped(self):
        """Agrees with ``build_param_grid``: None and [] both mean "not tuned"."""
        space = build_search_space(
            "rf", {"n_estimators": [10], "max_depth": None, "max_features": []}
        )
        assert list(space) == ["n_estimators"]

    def test_a_bare_string_is_not_searched_character_by_character(self):
        """A str is a Sequence, so ``max_features: sqrt`` could become 4 bad values."""
        space = build_search_space("rf", {"max_features": "sqrt"})
        assert space["max_features"].values == ["sqrt"]

    def test_a_bare_scalar_is_wrapped(self):
        space = build_search_space("rf", {"max_depth": 5})
        assert space["max_depth"].values == [5]

    def test_zero_is_kept(self):
        """0 is a legal hyperparameter value; only None and [] mean "not tuned"."""
        space = build_search_space("xgb", {"gamma": [0, 1]})
        assert space["gamma"].values == [0, 1]

    def test_the_two_builders_agree_on_which_hyperparameters_are_searched(self):
        """Switching ``tuner`` must not change *which* hyperparameters are tuned.

        Both engines are handed the same ``candidates`` dict, so if they disagreed
        here a config would quietly search a different set depending on the engine.
        """
        candidates = {
            "n_estimators": [10, 50],
            "max_features": "sqrt",
            "max_depth": None,
            "min_samples_leaf": [],
            "bootstrap": 5,
        }
        assert set(build_search_space("rf", candidates)) == set(build_param_grid("rf", candidates))

    def test_an_entirely_empty_space_names_the_config_not_a_parameter(self):
        """The message must point at the config block, as ``build_param_grid``'s does."""
        with pytest.raises(ValueError) as excinfo:
            build_search_space("rf", {"n_estimators": None, "max_depth": []})
        msg = str(excinfo.value)
        assert "gridsearch_rf_args" in msg, "must name the config block"
        assert "grid_search: False" in msg, "must say how to opt out of tuning"
        assert "n_estimators" in msg and "max_depth" in msg

    @pytest.mark.parametrize(
        "spec,expected",
        [
            ({"low": 10}, "'low' and 'high'"),
            ({"high": 10}, "'low' and 'high'"),
            ({"low": 10, "high": 1}, "empty range"),
            ({"low": 1, "high": 1}, "empty range"),
            ({"low": 0, "high": 10, "log": True}, "positive lower bound"),
            ({"low": 1, "high": 10, "log": True, "step": 2}, "cannot combine"),
            ({"low": 1, "high": 10, "scale": "log"}, "unrecognised"),
            ({"low": "a", "high": "z"}, "non-numeric"),
        ],
    )
    def test_a_malformed_range_names_the_model_and_the_hyperparameter(self, spec, expected):
        """Optuna would otherwise raise about a distribution the user never named."""
        with pytest.raises(ValueError) as excinfo:
            build_search_space("rf", {"n_estimators": spec})
        msg = str(excinfo.value)
        assert expected in msg
        assert "'rf'" in msg and "n_estimators" in msg


class TestRunStudy:
    """The budget, its reproducibility, and that a range is really continuous."""

    def test_the_same_seed_gives_the_same_answer(self, data):
        """QProfiler fills ``random_state`` from the run's seed; tuning must honour it."""
        X, y = data
        space = build_search_space(
            "rf", {"n_estimators": {"low": 5, "high": 50}, "max_depth": [2, 3, 4]}
        )
        kwargs = dict(cv=3, n_trials=8, seed=42, fixed={"random_state": 42})
        first = run_study(RandomForestClassifier, space, X, y, **kwargs)
        second = run_study(RandomForestClassifier, space, X, y, **kwargs)
        assert first == second

    def test_a_different_seed_may_explore_differently(self, data):
        """Not a correctness requirement -- just evidence the seed is actually wired."""
        X, y = data
        space = build_search_space("rf", {"n_estimators": {"low": 5, "high": 200}})
        runs = {
            seed: run_study(
                RandomForestClassifier,
                space,
                X,
                y,
                cv=3,
                n_trials=8,
                seed=seed,
                fixed={"random_state": 0},
            )["n_estimators"]
            for seed in (1, 2, 3, 4)
        }
        assert len(set(runs.values())) > 1, (
            f"four seeds all proposed the same value ({runs}), so the sampler seed "
            f"is probably not reaching Optuna"
        )

    def test_a_finite_space_is_not_searched_more_times_than_it_has_points(self, data):
        """``gridsearch_nb_args`` is 8 values; 50 trials would re-fit the same models."""
        X, y = data
        space = build_search_space("rf", {"max_depth": [2, 3, 4]})
        seen = []
        original = _Categorical.suggest

        def counting_suggest(self, trial, name):
            value = original(self, trial, name)
            seen.append(value)
            return value

        _Categorical.suggest = counting_suggest
        try:
            run_study(
                RandomForestClassifier,
                space,
                X,
                y,
                cv=3,
                n_trials=50,
                seed=0,
                fixed={"random_state": 0},
            )
        finally:
            _Categorical.suggest = original
        assert len(seen) == 3, f"3-point space searched {len(seen)} times"

    def test_a_range_is_searched_continuously(self, data):
        """The whole point of a range: values off the decade grid are reachable."""
        X, y = data
        space = build_search_space("svc", {"C": {"low": 0.1, "high": 100.0, "log": True}})
        proposed = []
        original = _Range.suggest

        def recording_suggest(self, trial, name):
            value = original(self, trial, name)
            proposed.append(value)
            return value

        _Range.suggest = recording_suggest
        try:
            best = run_study(SVC, space, X, y, cv=3, n_trials=10, seed=0, fixed={"random_state": 0})
        finally:
            _Range.suggest = original

        assert 0.1 <= best["C"] <= 100.0, f"best C={best['C']} outside the range"
        assert any(not float(v).is_integer() for v in proposed), (
            f"every proposed C was a whole number ({proposed}), which suggests the "
            f"range collapsed to a categorical choice"
        )

    def test_fixed_keywords_reach_the_estimator_without_being_searched(self, data):
        """``random_state`` is passed to every trial but must not appear in the result."""
        X, y = data
        space = build_search_space("rf", {"max_depth": [2, 3]})
        best = run_study(
            RandomForestClassifier, space, X, y, cv=3, n_trials=2, seed=0, fixed={"random_state": 7}
        )
        assert set(best) == {"max_depth"}


def test_model_run_rejects_an_unknown_tuner():
    """A misspelt tuner would otherwise fall through and silently run Optuna."""
    from qbiocode.evaluation.model_run import model_run

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = (X[:, 0] > 0).astype(int)
    args = {
        "model": ["dt"],
        "grid_search": True,
        "tuner": "bogus",
        "seed": 1,
        "n_jobs": 1,
        "cross_validation": 3,
    }
    with pytest.raises(ValueError) as excinfo:
        model_run(X[:30], X[30:], y[:30], y[30:], "key", args)
    msg = str(excinfo.value)
    assert "bogus" in msg
    assert "optuna" in msg and "grid" in msg


def test_a_range_is_rejected_by_the_grid_tuner():
    """A grid can only enumerate, so a range must be refused in our own terms.

    A dict is not a Sequence, so it used to be wrapped into a one-element list and
    passed to the estimator as a value, surfacing as ``InvalidParameterError: The 'C'
    parameter of SVC must be a float in the range (0.0, inf]. Got {'low': 0.001,
    ...}`` -- an error about the estimator that named neither the config entry nor
    the tuner that would have accepted it.
    """
    with pytest.raises(ValueError) as excinfo:
        build_param_grid("svc", {"C": {"low": 0.001, "high": 100, "log": True}})
    msg = str(excinfo.value)
    assert "tuner: optuna" in msg, "must name the tuner that accepts a range"
    assert "'svc'" in msg and "'C'" in msg, "must name the model and hyperparameter"


class TestTheBugsFoundAfterTheFirstPass:
    """Regressions found by probing the finished feature, each a wrong-blame message."""

    def test_a_list_valued_choice_does_not_make_optuna_warn(self, data):
        """The shipped ``gridsearch_mlp_args`` is the case that exposed this.

        Its ``hidden_layer_sizes`` is ``[[20], [50], [100]]``. Optuna's categorical
        distribution warns once per trial for a choice it cannot store -- "should be a
        tuple of None, bool, int, float and str ... but contains [20] which is of type
        list" -- so a *default* config buried its own output in warnings. The values are
        coerced to tuples, which sklearn treats identically.
        """
        from sklearn.neural_network import MLPClassifier

        X, y = data
        space = build_search_space("mlp", {"hidden_layer_sizes": [[20], [50]]})
        # reported back as configured, not coerced
        assert space["hidden_layer_sizes"].values == [[20], [50]]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            best = run_study(
                MLPClassifier,
                space,
                X,
                y,
                cv=3,
                n_trials=2,
                seed=0,
                fixed={"random_state": 0, "max_iter": 50},
            )
        categorical = [w for w in caught if "categorical distribution" in str(w.message)]
        assert not categorical, f"optuna still warns: {[str(w.message)[:120] for w in categorical]}"
        assert best["hidden_layer_sizes"] in [[20], [50]]

    @pytest.mark.parametrize("n_trials", [0, -5, 1.5, None])
    def test_a_budget_below_one_trial_is_named(self, n_trials, data):
        """``n_trials: 0`` used to surface as Optuna's "No trials are completed yet"."""
        from sklearn.ensemble import RandomForestClassifier

        X, y = data
        space = build_search_space("rf", {"max_depth": [2, 3]})
        with pytest.raises(ValueError) as excinfo:
            run_study(RandomForestClassifier, space, X, y, cv=3, n_trials=n_trials, seed=0)
        assert "at least one trial" in str(excinfo.value)
        assert repr(n_trials) in str(excinfo.value)
