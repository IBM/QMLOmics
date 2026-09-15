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

"""Gradient boosting with CatBoost, as a QProfiler classical learner.

CatBoost sits alongside :mod:`qbiocode.learning.compute_xgb` as the second
gradient-boosting baseline. Three things about it differ from XGBoost in ways that
matter to the rest of the pipeline, and each is handled here rather than left to
surprise a caller:

* **It writes to disk by default.** Every ``fit`` drops a ``catboost_info/``
  directory into the *current working directory* holding training logs. QProfiler
  fans models out over joblib workers that share a CWD, so a run with CatBoost in
  ``model`` would have every worker writing the same directory at once and leave the
  litter behind afterwards. ``allow_writing_files=False`` is therefore not optional
  here.

* **It is loud by default.** CatBoost prints a line per boosting iteration. That is
  the same noise ``_tuning`` already suppresses for Optuna and the quantum learners,
  so training output is silenced independently of QBioCode's own ``verbose`` flag --
  which controls the one-line result summary, not the fit.

* **Two of its hyperparameters conflict, and which way they conflict depends on the
  configured loss.** ``subsample`` and ``bagging_temperature`` belong to mutually
  exclusive bootstrap schemes, and CatBoost picks the default scheme from the loss:
  ``MVS`` under ``Logloss``, ``Bayesian`` under ``MultiClass``. A config naming
  ``subsample`` therefore trains fine at the inferred default and dies as soon as the
  loss changes::

      CatBoostError: catboost/private/libs/options/catboost_options.cpp:795:
      Error: default bootstrap type is Bayesian, which does not support subsample

  QProfiler targets binary classification, where the inferred loss is ``Logloss`` and
  the default bootstrap is ``MVS``, so this is not reached by varying the data. It is
  reached by ``loss_function``, which both learners expose: ``'MultiClass'`` is legal
  on a two-class target and selects the Bayesian bootstrap, breaking ``subsample`` --
  and separately making ``predict`` return an (n, 1) column rather than a flat array.
  (The same switch happens implicitly if a target with more than two levels is ever
  passed, though ``modeleval`` cannot score one.) :func:`_resolve_bootstrap` pins the
  scheme up front rather than leaving validity contingent on the loss, and rejects a
  genuinely contradictory block with a message naming the config key.

No ``OneVsOneClassifier`` wrapper, unlike :func:`compute_rf` and
:func:`compute_xgb`: CatBoost selects ``MultiClass`` as its loss automatically when
the target has more than two levels, so wrapping it would fit n(n-1)/2 boosters to
reach the same answer the native loss already gives. That the wrapper is absent is
also why ``random_state`` is recorded unprefixed rather than as
``estimator__random_state``; see ``tests/test_split_reproducibility.py``.
"""

# ====== Base class imports ======

import time
from collections.abc import Mapping, Sequence

# ====== CatBoost imports ======

# Deliberately broad, and for the same reason as compute_xgb's guard: a missing
# catboost raises ImportError, but a catboost whose native extension cannot load
# raises OSError, and narrowing to ImportError would let that escape untranslated at
# package-import time. The reason is kept so the message below can quote the real
# failure rather than guess at it.
try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
    _CATBOOST_ERROR = None
except Exception as exc:  # noqa: BLE001 -- see above
    CATBOOST_AVAILABLE = False
    _CATBOOST_ERROR = str(exc)
    CatBoostClassifier = None  # type: ignore

# ====== Scikit-learn imports ======

from sklearn.model_selection import GridSearchCV

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import extract_binary_scores, modeleval
from qbiocode.learning._grid import build_param_grid
from qbiocode.learning._tuning import build_search_space, run_study

# ====== Module constants ======

#: Settings every CatBoost estimator built here is given, for the two reasons in the
#: module docstring. ``verbose`` is CatBoost's own training chatter and is unrelated
#: to QBioCode's ``verbose`` argument, which selects the result summary.
_QUIET = {"verbose": False, "allow_writing_files": False}

#: The bootstrap scheme each hyperparameter requires. CatBoost validates these at
#: ``fit`` time; pinning the scheme is what keeps the requirement from depending on
#: whichever loss CatBoost inferred or was handed.
_BAYESIAN_ONLY = "bagging_temperature"
_NON_BAYESIAN_ONLY = "subsample"

# ====== Begin functions ======


def _require_catboost():
    """Raise an actionable ImportError when CatBoost is unusable.

    CatBoost is a base dependency, so reaching this means a broken environment
    rather than a missing extra -- the message says so instead of suggesting an
    extra that does not exist.
    """
    if CATBOOST_AVAILABLE:
        return
    raise ImportError(
        "CatBoost is not properly installed or configured.\n"
        f"Error: {_CATBOOST_ERROR}\n\n"
        "CatBoost is a core QBioCode dependency, so this is a broken install rather "
        "than a missing extra. Reinstall it with:\n"
        "  pip install --force-reinstall catboost\n\n"
        "If the import itself succeeds outside QBioCode, check that the wheel matches "
        "this interpreter's platform and Python version."
    )


def _resolve_bootstrap(candidates, fixed, block="gridsearch_catboost_args"):
    """Pin ``bootstrap_type`` so the search cannot depend on the configured loss.

    ``bagging_temperature`` is accepted only under the ``Bayesian`` bootstrap and
    ``subsample`` only under a non-Bayesian one (``Bernoulli``, ``MVS``, ``Poisson``).
    CatBoost's *default* scheme is chosen from the loss -- ``MVS`` for ``Logloss``,
    ``Bayesian`` for ``MultiClass`` -- so leaving it unset makes the validity of a
    config block a property of ``loss_function`` (or of the target's class count)
    rather than of the block itself.

    Args:
        candidates (dict): The hyperparameters the config asked for, before the empty
            entries are dropped. Values may be lists (a tuned block) or bare scalars
            (an untuned one); both paths call this so they cannot disagree about which
            combinations are legal.
        fixed (dict): Estimator settings that are passed but not searched. Mutated
            in place to carry the resolved ``bootstrap_type``.
        block (str): Config block to name in error messages -- ``'catboost_args'`` for
            the untuned path, ``'gridsearch_catboost_args'`` for the tuned one.

    Raises:
        ValueError: If the block asks for both schemes at once, either by naming
            both hyperparameters or by searching ``bootstrap_type`` across values
            that cannot all satisfy the one that was named. Optuna would otherwise
            sample the bad corner partway through the study and abort the run.
    """
    tuned = {name for name, values in candidates.items() if values is not None and values != []}
    wants_bayesian = _BAYESIAN_ONLY in tuned
    wants_other = _NON_BAYESIAN_ONLY in tuned

    if wants_bayesian and wants_other:
        raise ValueError(
            f"{block!r} names both {_NON_BAYESIAN_ONLY!r} and "
            f"{_BAYESIAN_ONLY!r}, which belong to mutually exclusive CatBoost bootstrap "
            f"schemes -- {_NON_BAYESIAN_ONLY!r} needs a Bernoulli/MVS/Poisson bootstrap "
            f"and {_BAYESIAN_ONLY!r} needs the Bayesian one. No single trial can honour "
            f"both. Drop whichever you care about less."
        )

    explicit = candidates.get("bootstrap_type")
    # A range is meaningless for a string enum, and left alone it is destructive rather
    # than merely wrong: `list({'low': 1, 'high': 3})` is `['low', 'high']`, which passes
    # the compatibility checks below and then reaches build_search_space, which reads the
    # mapping as an *integer* range and proposes `bootstrap_type=2`. CatBoost then fails
    # with `Can't parse parameter "type" with value: 2` -- naming neither the config entry
    # nor the real mistake.
    if isinstance(explicit, Mapping):
        raise ValueError(
            f"{block!r} gives 'bootstrap_type' a range ({dict(explicit)!r}), but it names "
            f"a bootstrap scheme rather than a number, so there is nothing to sample "
            f"between. Write it as a list, e.g. bootstrap_type: ['Bernoulli', 'MVS']."
        )
    if explicit is not None and explicit != []:
        # The user is searching the scheme itself. Every value has to be compatible
        # with whichever of the two hyperparameters was also named, or some fraction
        # of the trials is guaranteed to raise.
        values = [explicit] if isinstance(explicit, str) else list(explicit)
        if wants_bayesian and [v for v in values if v != "Bayesian"]:
            raise ValueError(
                f"{block!r} sets {_BAYESIAN_ONLY!r}, which CatBoost "
                f"accepts only under the Bayesian bootstrap, but its 'bootstrap_type' "
                f"values are {values}. Trials landing on "
                f"{[v for v in values if v != 'Bayesian']} would fail. Either set "
                f"bootstrap_type: ['Bayesian'] or drop {_BAYESIAN_ONLY!r}."
            )
        if wants_other and "Bayesian" in values:
            raise ValueError(
                f"{block!r} sets {_NON_BAYESIAN_ONLY!r}, which the "
                f"Bayesian bootstrap does not support, but 'bootstrap_type' includes "
                f"'Bayesian'. Those trials would fail. Either remove 'Bayesian' from "
                f"bootstrap_type or drop {_NON_BAYESIAN_ONLY!r}."
            )
        return

    # Not searched, so it is ours to pin -- and it has to be pinned, or the scheme is
    # decided at fit time by whichever loss CatBoost inferred or was handed.
    if wants_bayesian:
        fixed["bootstrap_type"] = "Bayesian"
    elif wants_other:
        fixed["bootstrap_type"] = "Bernoulli"


def compute_catboost(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="catboost",
    data_key="",
    *,
    iterations=None,
    learning_rate=None,
    depth=None,
    l2_leaf_reg=None,
    border_count=None,
    random_strength=None,
    bagging_temperature=None,
    subsample=None,
    bootstrap_type=None,
    grow_policy=None,
    min_data_in_leaf=None,
    rsm=None,
    one_hot_max_size=None,
    loss_function=None,
    thread_count=None,
    random_state=None,
):
    """
    This function generates a model using a Gradient Boosting Classifier method as implemented in
    `CatBoost <https://catboost.ai/docs/en/references/training-parameters/common>`__. It takes in parameter
    arguments specified in the config.yaml file, but will leave any parameter not named at CatBoost's own
    default if none are passed.
    The model is trained on the training dataset and validated on the test dataset. The function returns the evaluation of the model
    on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Every hyperparameter defaults to ``None``, meaning "not specified, leave CatBoost's
    own default in force". That is deliberate rather than lazy: CatBoost *derives*
    several defaults from the data and from each other -- it auto-selects
    ``learning_rate`` for ``Logloss`` and ``MultiClass`` unless ``l2_leaf_reg`` is set,
    and picks ``bootstrap_type`` from the inferred loss -- so writing its documented
    default into the call is not a no-op and would quietly disable that logic. This
    mirrors how :mod:`qbiocode.learning._grid` treats an unnamed hyperparameter.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Additional arguments, typically from a configuration file.
        verbose (bool): If True, prints the one-line QBioCode result summary. It does
            not enable CatBoost's own per-iteration training output, which is always
            off -- see the module docstring.
        model (str): Name of the model being used, default is 'catboost'.
        data_key (str): Key for identifying the dataset, default is an empty string.
        iterations (int or None): Maximum number of trees to build. CatBoost's default is 1000.
        learning_rate (float or None): Gradient step shrinkage. CatBoost auto-selects this for
            ``Logloss``/``MultiClass`` from the iteration count, but only while ``l2_leaf_reg``
            is also unset; its documented fallback is 0.03.
        depth (int or None): Depth of each tree, up to 16 on CPU. CatBoost's default is 6.
        l2_leaf_reg (float or None): L2 coefficient on the cost function. CatBoost's default is
            3.0. Setting it disables the ``learning_rate`` auto-selection described above.
        border_count (int or None): Number of splits considered per numerical feature.
        random_strength (float or None): Variance multiplier on the noise added to split scores,
            used to counter overfitting. CatBoost's default is 1. CPU only.
        bagging_temperature (float or None): Aggressiveness of the Bayesian bootstrap's weighting;
            0 gives every object weight 1. Accepted only under ``bootstrap_type='Bayesian'``.
        subsample (float or None): Fraction of objects sampled per tree. Accepted only under a
            Bernoulli, MVS or Poisson bootstrap -- *not* under the Bayesian one, which is
            CatBoost's default for multiclass targets.
        bootstrap_type (str or None): One of ``'Bayesian'``, ``'Bernoulli'``, ``'MVS'``,
            ``'Poisson'`` (GPU only) or ``'No'``. Left unset, CatBoost chooses from the inferred
            loss, which makes the two parameters above data-dependent; pass it explicitly
            whenever either is used.
        grow_policy (str or None): ``'SymmetricTree'`` (CatBoost's default), ``'Depthwise'`` or
            ``'Lossguide'``.
        min_data_in_leaf (int or None): Minimum objects in a leaf. Intended for the ``Depthwise``
            and ``Lossguide`` policies.
        rsm (float or None): Share of features considered at each split, in (0, 1]. CPU only.
        one_hot_max_size (int or None): Categorical features with at most this many distinct
            values are one-hot encoded instead of target-statistic encoded.
        loss_function (str or None): Objective. Left unset, CatBoost picks ``Logloss`` for a
            two-class target and ``MultiClass`` for more, which is why no one-vs-one wrapper is
            needed here.
        thread_count (int or None): Threads CatBoost may use. Left unset it takes every core,
            which oversubscribes when QProfiler is already running one model per worker.
        random_state (int or None): Seed for the estimator's own randomness, accepted by CatBoost
            as an alias of ``random_seed``. QProfiler fills this in from the run's ``seed`` so two
            runs at one seed agree; None leaves the estimator drawing from the global RNG.

    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model, including accuracy, AUC, F1 score, and the time taken for training and validation.

    Raises:
        ImportError: If CatBoost is not properly installed or configured.
    """

    _require_catboost()

    beg_time = time.time()
    # Only what was actually named. CatBoost rejects `None` for several of these
    # (`bootstrap_type=None` among them) rather than reading it as "use the default",
    # so an unset hyperparameter has to be left out of the call entirely.
    params = {
        name: value
        for name, value in (
            ("iterations", iterations),
            ("learning_rate", learning_rate),
            ("depth", depth),
            ("l2_leaf_reg", l2_leaf_reg),
            ("border_count", border_count),
            ("random_strength", random_strength),
            ("bagging_temperature", bagging_temperature),
            ("subsample", subsample),
            ("bootstrap_type", bootstrap_type),
            ("grow_policy", grow_policy),
            ("min_data_in_leaf", min_data_in_leaf),
            ("rsm", rsm),
            ("one_hot_max_size", one_hot_max_size),
            ("loss_function", loss_function),
            ("thread_count", thread_count),
        )
        if value is not None
    }
    # The same guard the tuned path uses, rather than a local copy of the pinning. The
    # local copy pinned Bernoulli whenever `subsample` was set and then left CatBoost to
    # raise about `bagging_temperature` if both were given -- so `catboost_args` got a
    # bare CatBoostError for a combination `gridsearch_catboost_args` rejected up front
    # with a message naming the key. Sharing it makes that divergence impossible.
    _resolve_bootstrap(
        {
            "subsample": subsample,
            "bagging_temperature": bagging_temperature,
            "bootstrap_type": bootstrap_type,
        },
        params,
        block="catboost_args",
    )

    catboost = CatBoostClassifier(**params, random_state=random_state, **_QUIET)  # type: ignore
    # Fit the training datset
    model_fit = catboost.fit(X_train, y_train)
    # CatBoost's get_params() reports only what was explicitly set, so this is the
    # block above rather than the full parameter surface -- which is the more useful
    # record anyway: it says what the config asked for.
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = catboost.predict(X_test)
    # Under the MultiClass loss CatBoost returns an (n, 1) column rather than a flat
    # array. scikit-learn 1.9 squeezes it when scoring, so this is not what stops a
    # crash -- it is what keeps `y_predicted_<model>` in the results frame the same
    # shape every other learner puts there, instead of resting on that squeezing, which
    # is not a documented guarantee.
    y_predicted = y_predicted.ravel()
    # `auc` is computed from these probabilities alone. CatBoost is fitted unwrapped, so
    # predict_proba is available -- unlike the seven OneVsOneClassifier-wrapped learners,
    # where the absence of it is what made the old label-based `auc` look unavoidable.
    # Under the MultiClass loss on a two-class target predict_proba still returns two
    # columns, so this route survives the same loss_function corner as the ravel above.
    y_score = extract_binary_scores(catboost, X_test)
    return modeleval(
        y_test,
        y_predicted,
        beg_time,
        model_params,
        args,
        model=model,
        verbose=verbose,
        y_score=y_score,
    )


def compute_catboost_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    cv=5,
    model="catboost",
    iterations=None,
    learning_rate=None,
    depth=None,
    l2_leaf_reg=None,
    border_count=None,
    random_strength=None,
    bagging_temperature=None,
    subsample=None,
    bootstrap_type=None,
    grow_policy=None,
    min_data_in_leaf=None,
    rsm=None,
    one_hot_max_size=None,
    loss_function=None,
    thread_count=None,
    random_state=None,
    *,
    tuner="optuna",
    n_trials=50,
):
    """
    This function also generates a model using a Gradient Boosting Classifier method as implemented in
    `CatBoost <https://catboost.ai/docs/en/references/training-parameters/common>`__.
    The difference here is that this function tunes the model's hyperparameters.
    The values or ranges searched for each parameter are specified in the config.yaml file,
    and ``tuner`` selects the search engine (Optuna by default). The
    combination of parameters that led to the best performance is saved and returned as best_params, which can then be used on similar
    datasets, without having to repeat the search.
    The model is trained on the training dataset and validated on the test dataset. The function returns the evaluation of the model
    on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model across the search.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Additional arguments, typically from a configuration file.
        verbose (bool): If True, prints additional information during execution.
        cv (int): Number of cross-validation folds, default is 5.
        model (str): Name of the model being used, default is 'catboost'.
        iterations (list or dict): Values or ``{low, high}`` range of tree counts to search.
        learning_rate (list or dict): Values or range of gradient step shrinkages to search.
        depth (list or dict): Values or range of tree depths to search.
        l2_leaf_reg (list or dict): Values or range of L2 leaf regularisation coefficients.
        border_count (list or dict): Values or range of numerical split counts.
        random_strength (list or dict): Values or range of split-score noise multipliers.
        bagging_temperature (list or dict): Values or range for the Bayesian bootstrap's
            weighting. Naming it pins ``bootstrap_type`` to ``'Bayesian'``, because CatBoost
            accepts it under no other scheme.
        subsample (list or dict): Values or range of per-tree object fractions. Naming it pins
            ``bootstrap_type`` to ``'Bernoulli'`` for the same reason in reverse -- the Bayesian
            bootstrap CatBoost defaults to on multiclass targets rejects it outright.
        bootstrap_type (list): Bootstrap schemes to search. Searching it yourself overrides the
            pinning above, and is then checked for compatibility with whichever of the two
            parameters above was named.
        grow_policy (list): Tree growing policies to search.
        min_data_in_leaf (int or None): Minimum objects in a leaf, passed to every trial
            rather than searched -- CatBoost honours it only under ``grow_policy``
            ``'Depthwise'`` or ``'Lossguide'``, so searching it at the default
            ``'SymmetricTree'`` spends fits on identical models. Several values are refused
            with a message saying so. Set ``grow_policy`` alongside it to make it bite.
        rsm (list or dict): Values or range of per-split feature shares.
        one_hot_max_size (list or dict): Values or range of one-hot encoding cutoffs.
        loss_function (str or None): Objective, passed to every trial rather than searched --
            it specifies the problem rather than tuning it. Accepted here for parity with
            :func:`compute_catboost`, so that a block naming it is not a ``TypeError``.
            Note that it participates in the bootstrap conflict below: ``'MultiClass'``
            selects the Bayesian bootstrap even on a two-class target.
        thread_count (int or None): Threads CatBoost may use per fit. Passed to every trial
            rather than searched.
        random_state (int or None): Seed for the estimator's own randomness. QProfiler fills this
            in from the run's ``seed`` so two runs at one seed agree; None leaves the estimator
            drawing from the global RNG.

        tuner (str): Which search to run. ``'optuna'`` (default) spends ``n_trials`` on
            Optuna's TPE sampler, which also allows a hyperparameter to be given as a
            ``{low, high}`` range rather than a list. ``'grid'`` restores the exhaustive
            ``GridSearchCV`` sweep over every combination.
        n_trials (int): Trial budget when ``tuner='optuna'``, default is 50. Lowered
            automatically when the configured values describe fewer distinct
            combinations than that, so a small block does not re-evaluate the same
            models.

    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model, including accuracy, AUC, F1 score, and the time taken for training and validation.

    Raises:
        ImportError: If CatBoost is not properly installed or configured.
        ValueError: If the configured block asks for both bootstrap schemes at once;
            see :func:`_resolve_bootstrap`.
    """

    _require_catboost()

    beg_time = time.time()
    # Only the hyperparameters actually supplied. Passing all of them meant a
    # config that named a subset died in sklearn on the first one it left at its
    # `[]` default; see qbiocode.learning._grid.
    candidates = {
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "l2_leaf_reg": l2_leaf_reg,
        "border_count": border_count,
        "random_strength": random_strength,
        "bagging_temperature": bagging_temperature,
        "subsample": subsample,
        "bootstrap_type": bootstrap_type,
        "grow_policy": grow_policy,
        "rsm": rsm,
        "one_hot_max_size": one_hot_max_size,
    }

    # Every estimator the search builds, and the final refit, need these. Resolved
    # before the search so a contradictory block fails now, with a message naming the
    # config key, rather than on whichever trial first samples the bad corner.
    fixed = {"random_state": random_state, **_QUIET}
    if thread_count is not None:
        fixed["thread_count"] = thread_count
    if loss_function is not None:
        fixed["loss_function"] = loss_function
    # `min_data_in_leaf` is passed to every trial rather than searched. CatBoost accepts it
    # under any grow policy and honours it under only two, so searching it at the default
    # `SymmetricTree` multiplied the number of fits while every value returned an identical
    # model -- measured: the predictions at 1 and at 60 are the same under SymmetricTree and
    # differ under Depthwise. It is still *accepted* here, rather than dropped from the
    # signature, so that a config naming it does not die on an unexpected keyword argument
    # inside a joblib worker -- the asymmetry that `loss_function` had.
    if min_data_in_leaf is not None and min_data_in_leaf != []:
        if isinstance(min_data_in_leaf, (Sequence, set, frozenset, Mapping)) and not isinstance(
            min_data_in_leaf, str
        ):
            raise ValueError(
                f"'gridsearch_catboost_args' gives 'min_data_in_leaf' several values "
                f"({min_data_in_leaf!r}), but it is not searched: CatBoost honours it only "
                f"under grow_policy 'Depthwise' or 'Lossguide', so at the default "
                f"'SymmetricTree' every value produces an identical model and the extra fits "
                f"buy nothing. Give it a single value instead, and set 'grow_policy' if you "
                f"want it to take effect -- or search 'grow_policy' and leave this out."
            )
        fixed["min_data_in_leaf"] = min_data_in_leaf
    _resolve_bootstrap(candidates, fixed)

    # Optuna by default; the exhaustive grid stays reachable so a number published
    # against it can still be reproduced. Both engines are handed the same
    # `candidates`, so switching `tuner` never changes *which* hyperparameters are
    # searched -- only how the search spends its fits.
    if tuner == "grid":
        search = GridSearchCV(
            CatBoostClassifier(**fixed),  # type: ignore
            param_grid=build_param_grid("catboost", candidates),
            cv=cv,
        )
        search.fit(X_train, y_train)
        best_params = search.best_params_
    else:
        best_params = run_study(
            CatBoostClassifier,
            build_search_space("catboost", candidates),
            X_train,
            y_train,
            cv=cv,
            n_trials=n_trials,
            seed=random_state,
            fixed=fixed,
        )
    best_catboost = CatBoostClassifier(**best_params, **fixed)  # type: ignore
    best_catboost.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_catboost.predict(X_test).ravel()
    # See compute_catboost: fitted unwrapped, so `auc` comes from predict_proba.
    y_score = extract_binary_scores(best_catboost, X_test)
    return modeleval(
        y_test,
        y_predicted,
        beg_time,
        best_params,
        args,
        model=model,
        verbose=verbose,
        y_score=y_score,
        # This function IS the tuned branch, so it states so rather than letting
        # modeleval infer it from the label: a DIRECT call leaves `model` at its
        # display-name default ('Decision Tree'), which carries no _opt marker.
        tuned=True,
    )
