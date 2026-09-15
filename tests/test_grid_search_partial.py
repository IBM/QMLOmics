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

"""Tuning a subset of a model's hyperparameters has to work.

Every ``compute_*_opt`` function takes one keyword per tunable hyperparameter and
handed all of them to ``GridSearchCV``. Each defaulted to ``[]``, so a config that
named a subset -- which is every realistic config, and every demo-sized one --
failed inside sklearn on whichever unmentioned parameter came first::

    ValueError: Parameter grid for parameter 'colsample_bytree' need to be a
    non-empty sequence, got: []

That names a parameter the user never wrote and points at sklearn rather than at
the config. It also made a small grid inexpressible, which is what stalled the QPL
tutorial: its XGBoost baseline shipped a 2430-combination grid because trimming it
was indistinguishable from breaking it.

A hyperparameter nobody asked to tune should be left at the estimator's default.
"""

import warnings

import numpy as np
import pytest

from qbiocode.learning._grid import build_param_grid

# The `*_opt` signature keyword, the grid entry it feeds, and a small legal value.
# One entry per model so a partial grid is exercised through the real call path.
PARTIAL_GRIDS = [
    ("catboost", {"iterations": [10, 20]}),
    ("dt", {"max_depth": [2, 3]}),
    ("lr", {"C": [0.1, 1.0]}),
    ("mlp", {"alpha": [1e-4, 1e-3]}),
    ("nb", {"var_smoothing": [1e-9, 1e-8]}),
    ("rf", {"n_estimators": [10, 20]}),
    ("svc", {"C": [0.1, 1.0]}),
    ("xgb", {"n_estimators": [10, 20]}),
    # tabpfn is here so the partial-grid contract covers it too, but it is the one
    # learner that cannot be assumed runnable: it needs the optional [tabpfn] extra
    # and a checkpoint it downloads on first use. The test skips it when either is missing --
    # see the `tabpfn_skip_reason` fixture in conftest.py.
    ("tabpfn", {"n_estimators": [1, 2]}),
]

#: Learners whose dependencies are not guaranteed present, and the fixture that says so.
_OPTIONAL_LEARNERS = {"tabpfn"}


@pytest.fixture
def data():
    """A separable binary problem small enough for a 5-fold search to be quick."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 6))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    return X[:40], X[40:], y[:40], y[40:]


class TestBuildParamGrid:
    """The helper itself, where the semantics are easiest to pin down."""

    def test_unsupplied_hyperparameters_are_dropped(self):
        grid = build_param_grid("rf", {"n_estimators": [10], "max_depth": None, "max_features": []})
        assert grid == {"n_estimators": [10]}, (
            "None and [] both mean 'not tuned' and must leave the estimator default "
            "in force rather than reaching sklearn as an empty axis"
        )

    def test_a_bare_string_is_not_searched_character_by_character(self):
        """``max_features: sqrt`` in YAML is a string, and a string is a sequence.

        Left alone it becomes four one-character values -- no error, and a
        ``best_params_`` that means nothing.
        """
        assert build_param_grid("rf", {"max_features": "sqrt"}) == {"max_features": ["sqrt"]}

    def test_a_bare_scalar_is_wrapped(self):
        assert build_param_grid("rf", {"max_depth": 5}) == {"max_depth": [5]}

    def test_an_entirely_empty_grid_names_the_config_not_a_parameter(self):
        with pytest.raises(ValueError) as exc:
            build_param_grid("rf", {"n_estimators": [], "max_depth": None})
        msg = str(exc.value)
        assert "grid_search: False" in msg, "must say how to opt out of tuning"
        assert "gridsearch_rf_args" in msg, "must name the config block to add"
        assert "n_estimators" in msg and "max_depth" in msg, "must list what can be tuned"

    def test_zero_is_kept(self):
        """``0`` is falsy but a legitimate value to search; only [] and None drop."""
        assert build_param_grid("mlp", {"alpha": [0, 1]}) == {"alpha": [0, 1]}


@pytest.mark.parametrize("tuner", ["optuna", "grid"])
@pytest.mark.parametrize("model,grid", PARTIAL_GRIDS, ids=[m for m, _ in PARTIAL_GRIDS])
def test_every_opt_learner_accepts_a_one_parameter_grid(
    model, grid, tuner, data, tabpfn_skip_reason
):
    """The regression guard: one hyperparameter is a complete, valid config.

    Run against both search engines. Optuna is the default, but ``tuner: grid``
    exists so a number published against the exhaustive sweep stays reproducible,
    and an escape hatch nothing exercises is not an escape hatch.
    """
    import sys

    if model in _OPTIONAL_LEARNERS and tabpfn_skip_reason is not None:
        pytest.skip(tabpfn_skip_reason)

    import qbiocode  # noqa: F401  -- orders the OpenMP runtimes before xgboost fits

    fn = getattr(sys.modules[f"qbiocode.learning.compute_{model}"], f"compute_{model}_opt")
    X_train, X_test, y_train, y_test = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fn(
            X_train, X_test, y_train, y_test, {"seed": 42, "grid_search": True},
            cv=3, tuner=tuner, n_trials=5, **grid,
        )

    # modeleval returns a one-row frame with a `results_<model label>` column
    # holding the metrics dict, and the label is the model's display name rather
    # than the dispatch key -- so find the column instead of assuming it.
    assert not result.empty, f"compute_{model}_opt returned an empty frame"
    (results_col,) = [c for c in result.columns if c.startswith("results_")]
    metrics = result[results_col].iloc[0]
    assert np.isfinite(metrics["accuracy"]), f"non-finite accuracy: {metrics['accuracy']}"

    tuned = next(iter(grid))
    best = metrics["BestParams_Tuned"]
    assert set(best) == {tuned}, (
        f"{tuned!r} was the only hyperparameter given, so it must be the only one "
        f"searched -- everything else stays at the estimator default. Got {sorted(best)}"
    )
    assert best[tuned] in grid[tuned], (
        f"best value {best[tuned]!r} is not one of the searched values {grid[tuned]}"
    )


def test_xgboost_says_so_when_given_a_parameter_it_ignores(data):
    """``bootstrap`` is not an XGBoost parameter, and its wrapper accepts it anyway.

    Left unreported it doubles the number of fits and every duplicate returns the
    same model -- the shipped tutorial config did exactly this.
    """
    import qbiocode  # noqa: F401

    from qbiocode.learning.compute_xgb import compute_xgb_opt

    X_train, X_test, y_train, y_test = data
    with pytest.warns(UserWarning, match="bootstrap"):
        compute_xgb_opt(
            X_train, X_test, y_train, y_test, {"seed": 42, "grid_search": True},
            cv=3, n_trials=3, n_estimators=[10], bootstrap=[True, False],
        )
