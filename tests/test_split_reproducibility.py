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

"""Splits must be reproducible, and must not reach for global random state.

QProfiler called ``train_test_split`` with no ``random_state``, so every iteration
drew a different split from the global RNG: ``--seed`` was accepted and silently
ignored for splitting, and no reported number could be reproduced. It now derives
``random_state = seed + iter``, which is distinct per iteration yet deterministic
across reruns and independent of any other RNG consumer that happens to run first.

The AST check below is the general guard: it fails on *any* unseeded scikit-learn
splitter added anywhere in the package, not just the two call sites that were wrong.
"""

from __future__ import annotations

import ast
import pathlib
from importlib import import_module

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

# Imported directly, not through ``pytest.importorskip``. The guard this replaces
# claimed the module "requires the [quvine] extra", which is false: it imports on
# a bare install -- networkx and scipy are enough, and the optional python-louvain
# import inside it is already handled locally. The guard therefore protected
# nothing while silently converting any real ImportError into six skips.
link_prediction = import_module("qbiocode.apps.quvine.evaluation.link_prediction")

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "qbiocode"

#: scikit-learn splitters whose output is random unless ``random_state`` is pinned.
#: Deliberately excludes ``KFold``/``GroupKFold``, which are deterministic unless
#: ``shuffle=True`` -- those are handled separately below.
SEEDED_SPLITTERS = {
    "train_test_split",
    "StratifiedShuffleSplit",
    "ShuffleSplit",
    "GroupShuffleSplit",
}

#: Deterministic unless shuffling is requested.
SHUFFLE_SPLITTERS = {"KFold", "StratifiedKFold", "GroupKFold"}

#: Estimator classes whose ``fit`` is randomized unless ``random_state`` is pinned:
#: feature permutation when splits tie, bootstrap sampling, weight initialisation,
#: row and column subsampling. ``SVC`` and ``LogisticRegression`` are only random
#: under some settings (``probability=True``, the saga/liblinear solvers), but they
#: are listed anyway -- pinning a seed that turns out not to matter costs nothing,
#: and leaving them out means the guard depends on reading the other arguments.
RANDOMIZED_ESTIMATORS = {
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "LogisticRegression",
    "MLPClassifier",
    "RandomForestClassifier",
    "SVC",
    "XGBClassifier",
}


def _calls():
    """Every call node in the package, with its file and line."""
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a syntax error is its own test failure
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                yield path, node, name


def _keywords(node):
    return {kw.arg for kw in node.keywords if kw.arg is not None}


def _has_double_star(node):
    """``f(**config)`` may well supply random_state; do not claim otherwise."""
    return any(kw.arg is None for kw in node.keywords)


class TestNoSplitterRunsUnseeded:
    def test_every_random_splitter_pins_random_state(self):
        offenders = []
        for path, node, name in _calls():
            if name not in SEEDED_SPLITTERS:
                continue
            if "random_state" in _keywords(node) or _has_double_star(node):
                continue
            rel = path.relative_to(PACKAGE_ROOT.parent)
            offenders.append(f"{rel}:{node.lineno} {name}()")
        assert not offenders, (
            "splitters called without random_state -- these splits cannot be "
            "reproduced, and --seed does not control them:\n  " + "\n  ".join(offenders)
        )

    def test_every_shuffling_cross_validator_pins_random_state(self):
        """``KFold(shuffle=True)`` without a seed is as irreproducible as the above."""
        offenders = []
        for path, node, name in _calls():
            if name not in SHUFFLE_SPLITTERS or _has_double_star(node):
                continue
            keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            shuffle = keywords.get("shuffle")
            shuffles = isinstance(shuffle, ast.Constant) and shuffle.value is True
            if shuffles and "random_state" not in keywords:
                rel = path.relative_to(PACKAGE_ROOT.parent)
                offenders.append(f"{rel}:{node.lineno} {name}(shuffle=True)")
        assert not offenders, "shuffling cross-validators without a seed:\n  " + "\n  ".join(
            offenders
        )


class TestNoEstimatorRunsUnseeded:
    """The companion to the splitter guard, for the models themselves.

    A seeded split reproduces the *data*; it does nothing about an estimator that
    permutes features, bootstraps rows or initialises weights from a global RNG.
    That gap is what made two QProfiler runs at seed 7 disagree, and it is not
    visible in any single file: each ``compute_*`` looked fine on its own, with
    ``random_state=None`` as a documented default that nothing ever filled in.
    Checking every construction site at once is the only way to see it.
    """

    def test_every_randomized_estimator_pins_random_state(self):
        offenders = []
        for path, node, name in _calls():
            if name not in RANDOMIZED_ESTIMATORS:
                continue
            if "random_state" in _keywords(node) or _has_double_star(node):
                continue
            rel = path.relative_to(PACKAGE_ROOT.parent)
            offenders.append(f"{rel}:{node.lineno} {name}()")
        assert not offenders, (
            "estimators constructed without random_state -- their fits draw from the "
            "global RNG, so two runs at the same seed can disagree:\n  "
            + "\n  ".join(offenders)
        )


class TestTheSeedPlusIterationContract:
    """QProfiler derives ``random_state = seed + iter`` for iteration ``iter``."""

    @staticmethod
    def _split_indices(seed):
        X = np.arange(100).reshape(50, 2)
        y = np.array([0, 1] * 25)
        train, test = train_test_split(X, y, test_size=0.3, random_state=seed)[:2]
        return tuple(train[:, 0].tolist()), tuple(test[:, 0].tolist())

    def test_the_same_seed_reproduces_the_same_split(self):
        assert self._split_indices(11) == self._split_indices(11)

    def test_a_different_seed_gives_a_different_split(self):
        assert self._split_indices(11) != self._split_indices(12)

    def test_iterations_differ_from_each_other_but_the_sequence_is_stable(self):
        seed = 42
        first_run = [self._split_indices(seed + i) for i in range(1, 6)]
        second_run = [self._split_indices(seed + i) for i in range(1, 6)]

        assert first_run == second_run, "the per-iteration sequence is not reproducible"
        assert len({split[1] for split in first_run}) == 5, (
            "two iterations drew the same test set: seed + iter is colliding"
        )

    def test_the_source_derives_the_split_seed_from_the_configured_seed(self):
        """Guards the derivation itself, which no unit test can reach directly."""
        source = (PACKAGE_ROOT / "apps" / "qprofiler" / "qprofiler.py").read_text(
            encoding="utf-8"
        )
        assert "split_seed = args['seed'] + iter" in source
        assert source.count("random_state=split_seed") == 2, (
            "both the stratified and unstratified train_test_split calls must be seeded"
        )


class TestLinkPredictionSplitsLeaveGlobalStateAlone:
    """``split_edges``/``sample_negative_edges`` seeded ``np.random`` process-wide.

    They were reproducible for a given seed, but only by reseeding the global RNG --
    so calling either one silently changed the random stream every later consumer in
    the process would draw from. They now use a local generator.
    """

    @pytest.fixture
    def _module(self):
        return link_prediction

    @staticmethod
    def _graph():
        import networkx as nx

        return nx.karate_club_graph()

    def test_the_same_seed_gives_the_same_edge_split(self, _module):
        first = _module.split_edges(self._graph(), seed=7)
        second = _module.split_edges(self._graph(), seed=7)
        assert first[1] == second[1] and first[2] == second[2]

    def test_a_different_seed_gives_a_different_edge_split(self, _module):
        assert _module.split_edges(self._graph(), seed=7)[2] != (
            _module.split_edges(self._graph(), seed=8)[2]
        )

    def test_splitting_does_not_reseed_the_callers_global_rng(self, _module):
        np.random.seed(1234)
        expected = np.random.rand(3).tolist()

        np.random.seed(1234)
        _module.split_edges(self._graph(), seed=999)
        actual = np.random.rand(3).tolist()

        assert actual == expected, (
            "split_edges perturbed the global numpy RNG: any later random draw in the "
            "calling process now depends on whether this function was called"
        )

    def test_negative_sampling_does_not_reseed_the_callers_global_rng(self, _module):
        graph = self._graph()
        existing = set(graph.edges())

        np.random.seed(555)
        expected = np.random.rand(3).tolist()

        np.random.seed(555)
        _module.sample_negative_edges(graph, 10, existing, "random", 999)
        actual = np.random.rand(3).tolist()

        assert actual == expected, "sample_negative_edges perturbed the global numpy RNG"

    def test_negative_sampling_is_still_reproducible(self, _module):
        graph = self._graph()
        existing = set(graph.edges())
        first = _module.sample_negative_edges(graph, 10, existing, "random", 3)
        second = _module.sample_negative_edges(graph, 10, existing, "random", 3)
        assert first == second


class TestEstimatorsAreSeededAtDispatch:
    """A seeded split is only half of reproducibility -- the estimators need one too.

    ``qprofiler`` calls ``np.random.seed(args['seed'])`` in the parent process, but
    ``model_run`` fans the models out with joblib, whose loky workers are fresh
    interpreters seeded from OS entropy. Every estimator left at
    ``random_state=None`` therefore drew a different random state on every run.
    ``DecisionTreeClassifier`` permutes the features before choosing a split, so a
    tie between two equally-good splits broke either way; that is how two
    end-to-end runs at seed 7 came back with 0.889 and 0.944 accuracy on the same
    row, and it is what these tests pin down in seconds rather than minutes.

    Asserting on the *recorded parameters* rather than on the metrics is
    deliberate. Whether an unseeded estimator actually changes its answer depends
    on there being a tie to break, which varies with the split -- a metric
    comparison would pass or fail by luck. The seed reaching the estimator is the
    property that matters, and it is exact.
    """

    ARGS = {"model": ["dt"], "seed": 7, "n_jobs": 1, "grid_search": False}

    @staticmethod
    def _dataset():
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 5))
        y = (X[:, 0] + 0.3 * rng.normal(size=60) > 0).astype(int)
        return train_test_split(X, y, stratify=y, test_size=0.3, random_state=9)

    @classmethod
    def _results(cls, model, **overrides):
        from qbiocode.evaluation.model_run import model_run

        X_train, X_test, y_train, y_test = cls._dataset()
        args = {**cls.ARGS, "model": [model], **overrides}
        raw = model_run(X_train, X_test, y_train, y_test, "tiny", args)
        return raw[f"results_{model}"][0]

    # Every classical estimator in the dispatch table whose scikit-learn class takes
    # a random_state *and* is wrapped in OneVsOneClassifier, which is what puts the
    # ``estimator__`` prefix on the recorded name. ``xgb`` is absent only because
    # xgboost is an optional install.
    @pytest.mark.parametrize("model", ["dt", "lr", "rf", "svc"])
    def test_the_configured_seed_reaches_every_estimator_that_takes_one(self, model):
        params = self._results(model)["Model_Parameters"]
        assert params["estimator__random_state"] == self.ARGS["seed"], (
            f"{model} ran with random_state="
            f"{params['estimator__random_state']!r}, so its randomness does not "
            f"follow the configured seed"
        )

    # CatBoost is fitted directly rather than through OneVsOneClassifier -- it selects
    # a multiclass loss on its own -- so its parameters are recorded unprefixed. The
    # property being checked is identical; only the key differs, which is why this
    # cannot simply join the list above.
    #
    # ``tabpfn`` is fitted directly too but is absent here: it needs the optional
    # [tabpfn] extra and a downloaded checkpoint, so it cannot be assumed runnable.
    # That its signature accepts ``random_state`` at all -- the property ``_seeded_kwargs``
    # actually depends on -- is asserted in tests/test_catboost_tabpfn.py.
    @pytest.mark.parametrize("model", ["catboost"])
    def test_the_seed_reaches_an_unwrapped_estimator_too(self, model):
        params = self._results(model, **{f"{model}_args": {"iterations": 10}})[
            "Model_Parameters"
        ]
        assert params["random_state"] == self.ARGS["seed"], (
            f"{model} ran with random_state={params.get('random_state')!r}, so its "
            f"randomness does not follow the configured seed. CatBoost samples rows "
            f"whenever a bootstrap is active, so this is not cosmetic."
        )

    def test_an_estimator_without_a_random_state_is_left_alone(self):
        """The signature check is what keeps this from being a TypeError.

        ``GaussianNB`` has no ``random_state``; injecting one unconditionally would
        turn a reproducibility fix into a crash for naive Bayes.
        """
        params = self._results("nb")["Model_Parameters"]
        assert "estimator__random_state" not in params

    def test_a_random_state_in_the_config_still_wins(self):
        """Filling the gap must not override a choice the user made explicitly."""
        params = self._results("dt", dt_args={"random_state": 3})["Model_Parameters"]
        assert params["estimator__random_state"] == 3

    def test_two_runs_agree_whatever_the_ambient_global_rng(self):
        """The behavioural half, stated as a property rather than a fixed value.

        Re-seeding the global RNG differently before each run stands in for the
        worker entropy that broke this, and does not depend on joblib's backend or
        its batching.
        """
        np.random.seed(1234)
        first = self._results("dt")
        np.random.seed(999_983)
        second = self._results("dt")

        def signature(row):
            return row["accuracy"], row["f1_score"], row["auc"]

        assert signature(first) == signature(second)
