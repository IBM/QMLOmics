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

"""Turning a partly-filled config block into a grid ``GridSearchCV`` accepts.

Every ``compute_*_opt`` function takes one keyword per tunable hyperparameter and
used to hand all of them to ``GridSearchCV`` unconditionally. Each defaulted to
``[]``, so any config that did not enumerate *every* hyperparameter of the chosen
model died inside sklearn::

    ValueError: Parameter grid for parameter 'colsample_bytree' need to be a
    non-empty sequence, got: []

The message names a parameter the user never mentioned and says nothing about the
config, which is the opposite of the useful direction. It also made a deliberately
small grid impossible to express: trimming a demo config down to two parameters
was indistinguishable from corrupting it.

A hyperparameter nobody asked to tune should simply be left at the estimator's own
default, which is what dropping it from the grid does.
"""

import warnings
from collections.abc import Mapping, Sequence


def build_param_grid(model, candidates):
    """Build a ``GridSearchCV`` ``param_grid`` from the values actually supplied.

    Args:
        model (str): Model name, used only to make the error message specific.
        candidates (dict): Maps hyperparameter name to the values to search. A
            value of ``None`` or an empty sequence means "not tuned" and is
            dropped, leaving the estimator's own default in force. A bare scalar
            or string is wrapped into a one-element list.

    Returns:
        dict: Only the entries worth searching, each a non-empty list.

    Raises:
        ValueError: If nothing at all was supplied. Grid search over an empty grid
            is a config mistake, not a one-point search, so it is worth saying so
            here rather than letting sklearn report it against an arbitrary
            parameter name.
    """
    grid = {}
    for name, values in candidates.items():
        if values is None:
            continue
        # A `{low, high}` range is meaningful to the Optuna tuner but not to a grid,
        # which can only enumerate. A dict is not a Sequence, so it used to be wrapped
        # into a one-element list and handed to the estimator as a *value*, surfacing
        # as `InvalidParameterError: The 'C' parameter of SVC must be a float ... Got
        # {'low': 0.001, ...}` -- an error about the estimator, naming neither the
        # config entry nor the tuner that would accept it.
        if isinstance(values, Mapping):
            raise ValueError(
                f"{model!r} hyperparameter {name!r} is written as a range "
                f"({dict(values)!r}), which only the Optuna tuner can sample. Either "
                f"set tuner: optuna, or write {name!r} as a list of values for the "
                f"grid to enumerate."
            )
        # A string is a sequence, so `max_features: sqrt` would otherwise be
        # searched as ['s', 'q', 'r', 't'] -- four invalid values, no error, and a
        # best_params_ that means nothing.
        if isinstance(values, str) or not isinstance(values, (Sequence, set, frozenset)):
            values = [values]
        values = list(values)
        if not values:
            continue
        grid[name] = values

    if not grid:
        raise ValueError(
            f"Grid search was requested for {model!r} but no hyperparameter values "
            f"were given, so there is nothing to search. Either add a "
            f"'gridsearch_{model}_args' block to the config naming at least one "
            f"hyperparameter and the values to try, or set grid_search: False to "
            f"run {model!r} at its default hyperparameters. "
            f"Recognised hyperparameters for this model: "
            f"{', '.join(sorted(candidates))}."
        )
    return grid


def warn_ignored_hyperparameter(model, name, reason):
    """Flag a hyperparameter the estimator will accept and then disregard.

    XGBoost's sklearn wrapper takes unknown keyword arguments without complaint,
    so a grid entry it does not implement is not an error -- it just multiplies the
    number of fits while every duplicate returns the same model.
    """
    warnings.warn(
        f"{model!r} was given a grid for {name!r}, which {reason} Every value will "
        f"be searched and will produce the same model, multiplying the run time for "
        f"nothing. Remove {name!r} from the grid.",
        UserWarning,
        stacklevel=3,
    )
