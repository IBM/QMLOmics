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

"""The two learners added alongside XGBoost, and the traps each of them brings.

CatBoost and TabPFN are both "just another classical classifier" from the config's
point of view, and neither behaves like one underneath. These tests pin the specific
things that were wrong, or would have been wrong, when they were wired in.

**CatBoost's bootstrap schemes depend on the configured loss.** ``subsample`` and
``bagging_temperature`` belong to mutually exclusive bootstrap schemes, and CatBoost
chooses the default scheme from the loss -- ``MVS`` for ``Logloss``, ``Bayesian`` for
``MultiClass``. QProfiler is a binary-classification tool, so the inferred loss is
``Logloss`` and ``subsample`` is fine at the default; the reachable route in is
``loss_function``, which ``'MultiClass'`` sets legally even on a two-class target::

    CatBoostError: default bootstrap type is Bayesian, which does not support subsample

Optuna's objective in ``run_study`` does not catch exceptions, so one such corner aborts
the whole study rather than costing a trial. Most of the CatBoost tests here are about
that one bug; ``TestCatBoostSurvivesTheMultiClassLoss`` has the measurements.

Everything here uses a **binary** target, deliberately. ``qprofiler.py`` warns on any
other class count and ``modeleval``'s ``roc_auc_score`` call is binary-only for every
model in the package, so a three-class fixture would be testing outside the supported
envelope -- and would assert a downstream failure rather than a real score.

**CatBoost writes to the working directory and narrates to stdout.** Both defaults are
actively harmful under a joblib fan-out that shares a CWD.

**TabPFN must not be imported eagerly.** ``import tabpfn`` imports ``torch``, and
``qbiocode`` deliberately arranges for xgboost's ``libomp`` to initialise first
(see ``qbiocode.utils._openmp`` and ``tests/test_openmp_import_order.py``). An eager
import here would both undo that and require an optional extra at package-import time.

**TabPFN's weights are gated**, so anything that fits it skips unless the
``tabpfn_ready`` fixture says otherwise. Everything that does *not* need a fit --
the guards, the error messages, the registry wiring -- is tested unconditionally, and
that is most of it.
"""

import importlib
import ast
import inspect
import os
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

import qbiocode
from qbiocode.learning.compute_catboost import (
    _resolve_bootstrap,
    compute_catboost,
    compute_catboost_opt,
)
from qbiocode.learning.compute_tabpfn import (
    TABPFN_DEFAULT_VERSION,
    TABPFN_MAX_CLASSES,
    _cap_openmp_threads,
    normalise_model_version,
    resolve_model_path,
    _check_class_count,
    _explain_weight_access_failure,
    _load_tabpfn_classifier,
    compute_tabpfn,
    compute_tabpfn_opt,
    tabpfn_is_available,
)

# The modules, not the same-named functions. `qbiocode/learning/__init__.py` does
# `from .compute_tabpfn import compute_tabpfn`, which rebinds the attribute
# `qbiocode.learning.compute_tabpfn` from the submodule to the function -- so
# `from qbiocode.learning import compute_tabpfn` hands back the function and
# `monkeypatch.setattr` on it fails with a confusing AttributeError. import_module
# always returns the module.
catboost_module = importlib.import_module("qbiocode.learning.compute_catboost")
tabpfn_module = importlib.import_module("qbiocode.learning.compute_tabpfn")

UNTUNED = {"seed": 42, "grid_search": False}
TUNED = {"seed": 42, "grid_search": True}


@pytest.fixture
def binary():
    """A separable two-class problem; CatBoost infers ``Logloss``, so bootstrap is MVS."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 5))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    return X[:60], X[60:], y[:60], y[60:]


def metrics_of(frame):
    """The metrics dict out of a ``modeleval`` frame, without assuming the column name."""
    assert not frame.empty, "modeleval returned an empty frame"
    (column,) = [c for c in frame.columns if c.startswith("results_")]
    return frame[column].iloc[0]


# ======================================================================================
# CatBoost: the bootstrap-scheme trap
# ======================================================================================


class TestCatBoostBootstrapSchemes:
    """``subsample`` and ``bagging_temperature`` must not depend on the configured loss."""

    def test_subsample_is_pinned_to_a_compatible_scheme(self):
        fixed = {}
        _resolve_bootstrap({"subsample": [0.8]}, fixed)
        assert fixed["bootstrap_type"] == "Bernoulli", (
            "subsample must pin a non-Bayesian bootstrap; left unpinned CatBoost picks "
            "Bayesian for any multiclass target and rejects subsample outright"
        )

    def test_bagging_temperature_is_pinned_to_bayesian(self):
        fixed = {}
        _resolve_bootstrap({"bagging_temperature": [0.5]}, fixed)
        assert fixed["bootstrap_type"] == "Bayesian", (
            "bagging_temperature is accepted under no other scheme"
        )

    def test_nothing_is_pinned_when_neither_is_searched(self):
        """A block that avoids the area must keep CatBoost's own default behaviour."""
        fixed = {}
        _resolve_bootstrap({"depth": [4, 6], "iterations": [50]}, fixed)
        assert "bootstrap_type" not in fixed

    def test_an_explicit_bootstrap_search_is_left_alone(self):
        fixed = {}
        _resolve_bootstrap({"subsample": [0.8], "bootstrap_type": ["Bernoulli", "MVS"]}, fixed)
        assert "bootstrap_type" not in fixed, (
            "the user is searching the scheme themselves; overriding it would silently "
            "discard part of their grid"
        )

    def test_asking_for_both_schemes_is_rejected_with_both_names(self):
        with pytest.raises(ValueError) as exc:
            _resolve_bootstrap({"subsample": [0.8], "bagging_temperature": [0.5]}, {})
        message = str(exc.value)
        assert "subsample" in message and "bagging_temperature" in message
        assert "gridsearch_catboost_args" in message, "must name the config block to edit"

    def test_a_bayesian_value_alongside_subsample_is_rejected_before_the_search(self):
        """Optuna would otherwise abort mid-study on the first bad corner it samples."""
        with pytest.raises(ValueError, match="Bayesian"):
            _resolve_bootstrap({"subsample": [0.8], "bootstrap_type": ["Bayesian", "MVS"]}, {})

    def test_a_non_bayesian_value_alongside_bagging_temperature_is_rejected(self):
        with pytest.raises(ValueError) as exc:
            _resolve_bootstrap(
                {"bagging_temperature": [0.5], "bootstrap_type": ["Bernoulli", "Bayesian"]}, {}
            )
        assert "Bernoulli" in str(exc.value), "must name the offending value"

    def test_empty_and_none_entries_do_not_count_as_searched(self):
        """``[]`` and ``None`` mean 'not tuned' everywhere else; they must here too."""
        fixed = {}
        _resolve_bootstrap({"subsample": None, "bagging_temperature": []}, fixed)
        assert "bootstrap_type" not in fixed


class TestCatBoostSurvivesTheMultiClassLoss:
    """The same conflict, reached the way a binary-only pipeline can actually reach it.

    QProfiler targets binary classification -- ``qprofiler.py`` warns when a dataset has
    any other class count, and ``modeleval``'s ``roc_auc_score`` call is binary-only for
    every model in the package. So the class-count route into the bootstrap conflict is
    out of scope here.

    ``loss_function`` is not, and it is exposed on both learners. ``'MultiClass'`` is
    legal on a two-class target, and setting it moves CatBoost onto the Bayesian
    bootstrap exactly as three classes would -- which breaks ``subsample`` and changes
    the shape of ``predict()``. Measured on binary data:

    ===========================  ==================  ==============  ==============
    loss_function                default bootstrap   subsample       predict shape
    ===========================  ==================  ==============  ==============
    unset                        MVS                 ok              (n,)
    ``'Logloss'``                MVS                 ok              (n,)
    ``'MultiClass'``             Bayesian            **raises**      **(n, 1)**
    ===========================  ==================  ==============  ==============

    The two columns fail differently, and only the first is fatal. The pinning prevents a
    real ``CatBoostError``. The ``.ravel()`` prevents an *inconsistency*: scikit-learn
    1.9 silently squeezes a column vector, so ``modeleval`` scores it correctly either
    way -- but the array is stored in the results frame as ``y_predicted_<model>``, and
    every other learner puts a flat one there. Normalising the shape keeps that contract
    from depending on undocumented squeezing behaviour, so it is tested as a shape
    contract rather than by expecting an exception that does not occur.
    """

    MULTICLASS_LOSS = {"loss_function": "MultiClass"}

    def test_untuned_subsample_survives_the_multiclass_loss(self, tmp_path, monkeypatch, binary):
        """Unpinned this raised CatBoostError before the estimator was ever fitted."""
        monkeypatch.chdir(tmp_path)
        X_train, X_test, y_train, y_test = binary
        frame = compute_catboost(
            X_train, X_test, y_train, y_test, UNTUNED,
            iterations=20, subsample=0.8, random_state=42, **self.MULTICLASS_LOSS,
        )
        metrics = metrics_of(frame)
        assert np.isfinite(metrics["accuracy"])
        assert metrics["Model_Parameters"]["bootstrap_type"] == "Bernoulli", (
            "the multiclass loss must still get a subsample-compatible bootstrap pinned"
        )

    def test_tuned_subsample_survives_the_multiclass_loss(self, tmp_path, monkeypatch, binary):
        """Same guarantee on the tuned path, where a failure would abort the whole study.

        ``run_study``'s Optuna objective does not catch exceptions, so one unbuildable
        corner takes the run down rather than costing a trial.
        """
        monkeypatch.chdir(tmp_path)
        X_train, X_test, y_train, y_test = binary
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = compute_catboost_opt(
                X_train, X_test, y_train, y_test, TUNED,
                cv=3, n_trials=3, iterations=[10, 20], subsample=[0.7, 0.9],
                random_state=42, **self.MULTICLASS_LOSS,
            )
        metrics = metrics_of(frame)
        assert np.isfinite(metrics["accuracy"])
        assert "subsample" in metrics["BestParams_Tuned"]

    def test_loss_function_is_accepted_by_both_learners(self, binary):
        """``model_run`` splats the config block, so a name only one twin takes is a TypeError.

        ``loss_function`` specifies the problem rather than tuning it, so ``_opt`` passes
        it through fixed -- but it still has to *accept* it, or a block naming it dies
        with ``unexpected keyword argument`` inside a joblib worker.
        """
        for function in (compute_catboost, compute_catboost_opt):
            assert "loss_function" in inspect.signature(function).parameters, (
                f"{function.__name__} does not accept loss_function"
            )

    def test_the_raw_estimator_returns_a_column_vector_under_this_loss(self):
        """The premise the ``.ravel()`` rests on. If CatBoost stops doing this, drop it."""
        from catboost import CatBoostClassifier

        rng = np.random.default_rng(0)
        X = rng.normal(size=(90, 5))
        y = (X[:, 0] > 0).astype(int)
        raw = (
            CatBoostClassifier(
                iterations=10, random_state=42, verbose=False,
                allow_writing_files=False, **self.MULTICLASS_LOSS,
            )
            .fit(X, y)
            .predict(X)
        )
        assert raw.ndim == 2, (
            "CatBoost no longer returns a column vector under the MultiClass loss; "
            "the .ravel() in compute_catboost is then redundant and can go"
        )

    @pytest.mark.parametrize("loss", [None, "MultiClass"])
    def test_the_stored_prediction_is_always_flat(self, tmp_path, monkeypatch, binary, loss):
        """The contract the ``.ravel()`` actually buys, and the one worth pinning.

        ``modeleval`` stores the prediction in the frame as ``y_predicted_<model>``, and
        every other learner stores a 1-D array. Under the ``MultiClass`` loss CatBoost
        hands back an (n, 1) column; scikit-learn happens to squeeze it when scoring, so
        the metrics are right either way and nothing raises -- which is exactly why this
        needs asserting on the *shape* rather than left to a metric to catch.
        """
        monkeypatch.chdir(tmp_path)
        X_train, X_test, y_train, y_test = binary
        extra = {} if loss is None else {"loss_function": loss}
        frame = compute_catboost(
            X_train, X_test, y_train, y_test, UNTUNED,
            iterations=20, random_state=42, **extra,
        )
        stored = frame["y_predicted_catboost"].iloc[0]
        assert np.asarray(stored).ndim == 1, (
            f"loss_function={loss!r} put a {np.asarray(stored).shape} prediction in the "
            f"results frame; every other learner stores a flat array"
        )


class TestCatBoostDoesNotDisturbItsSurroundings:
    """Two defaults that are wrong under a joblib fan-out sharing a CWD."""

    def test_no_catboost_info_directory_is_left_behind(self, tmp_path, monkeypatch, binary):
        """Every default CatBoost fit writes training logs into the current directory."""
        monkeypatch.chdir(tmp_path)
        X_train, X_test, y_train, y_test = binary
        compute_catboost(
            X_train, X_test, y_train, y_test, UNTUNED, iterations=10, random_state=42
        )
        leftovers = sorted(p.name for p in tmp_path.iterdir())
        assert leftovers == [], (
            f"compute_catboost littered the working directory with {leftovers}; "
            "allow_writing_files=False has regressed. Under QProfiler every joblib "
            "worker shares this directory."
        )

    def test_the_fit_is_silent(self, tmp_path, monkeypatch, binary, capfd):
        """CatBoost prints a line per boosting iteration unless told not to."""
        monkeypatch.chdir(tmp_path)
        X_train, X_test, y_train, y_test = binary
        capfd.readouterr()
        compute_catboost(
            X_train, X_test, y_train, y_test, UNTUNED, iterations=50, random_state=42
        )
        captured = capfd.readouterr()
        assert captured.out == "", (
            f"CatBoost training output leaked to stdout: {captured.out[:300]!r}"
        )

    def test_qbiocode_verbose_is_not_catboosts_verbose(self, tmp_path, monkeypatch, binary, capfd):
        """``verbose=True`` must print the one-line summary, not 50 boosting lines."""
        monkeypatch.chdir(tmp_path)
        X_train, X_test, y_train, y_test = binary
        capfd.readouterr()
        compute_catboost(
            X_train, X_test, y_train, y_test, UNTUNED,
            iterations=50, random_state=42, verbose=True,
        )
        out = capfd.readouterr().out
        assert out, "verbose=True printed nothing at all"
        assert "learn:" not in out and "total:" not in out, (
            f"QBioCode's verbose flag was passed through to CatBoost: {out[:300]!r}"
        )


class TestCatBoostParameterHandling:
    """Unset hyperparameters must reach CatBoost's own defaults, not ``None``."""

    def test_an_unset_hyperparameter_is_not_passed_at_all(self, tmp_path, monkeypatch, binary):
        """``CatBoostClassifier(bootstrap_type=None)`` is an error, not a default.

        So a default-everything call has to leave the names out of the constructor
        rather than forward ``None`` -- which is also why the signature defaults are
        ``None`` rather than CatBoost's documented values.
        """
        monkeypatch.chdir(tmp_path)
        X_train, X_test, y_train, y_test = binary
        frame = compute_catboost(X_train, X_test, y_train, y_test, UNTUNED, random_state=42)
        reported = metrics_of(frame)["Model_Parameters"]
        assert "bootstrap_type" not in reported
        assert "grow_policy" not in reported
        assert reported.get("random_state") == 42

    def test_named_hyperparameters_are_reported(self, tmp_path, monkeypatch, binary):
        monkeypatch.chdir(tmp_path)
        X_train, X_test, y_train, y_test = binary
        frame = compute_catboost(
            X_train, X_test, y_train, y_test, UNTUNED,
            iterations=20, depth=4, l2_leaf_reg=2.5, random_state=42,
        )
        reported = metrics_of(frame)["Model_Parameters"]
        assert reported["iterations"] == 20
        assert reported["depth"] == 4
        assert reported["l2_leaf_reg"] == 2.5

    def test_the_same_seed_gives_the_same_score(self, tmp_path, monkeypatch, binary):
        monkeypatch.chdir(tmp_path)
        X_train, X_test, y_train, y_test = binary
        scores = [
            metrics_of(
                compute_catboost(
                    X_train, X_test, y_train, y_test, UNTUNED,
                    iterations=30, subsample=0.7, random_state=7,
                )
            )["accuracy"]
            for _ in range(2)
        ]
        assert scores[0] == scores[1], f"two runs at seed 7 disagreed: {scores}"

    def test_tuning_reports_the_tuned_parameters(self, tmp_path, monkeypatch, binary):
        monkeypatch.chdir(tmp_path)
        X_train, X_test, y_train, y_test = binary
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = compute_catboost_opt(
                X_train, X_test, y_train, y_test, TUNED,
                cv=3, n_trials=4, iterations=[10, 20], depth=[3, 4], random_state=42,
            )
        best = metrics_of(frame)["BestParams_Tuned"]
        assert set(best) == {"iterations", "depth"}, (
            f"tuned parameters should be exactly what was searched, got {best}"
        )
        assert best["iterations"] in (10, 20)
        assert best["depth"] in (3, 4)

    def test_a_range_is_accepted_by_optuna_and_refused_by_the_grid(
        self, tmp_path, monkeypatch, binary
    ):
        """The two engines must agree on which hyperparameters are searched."""
        monkeypatch.chdir(tmp_path)
        X_train, X_test, y_train, y_test = binary
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = compute_catboost_opt(
                X_train, X_test, y_train, y_test, TUNED,
                cv=3, n_trials=3, iterations=[10],
                learning_rate={"low": 0.01, "high": 0.3, "log": True}, random_state=42,
            )
        assert 0.01 <= metrics_of(frame)["BestParams_Tuned"]["learning_rate"] <= 0.3

        with pytest.raises(ValueError, match="only the Optuna tuner can sample"):
            compute_catboost_opt(
                X_train, X_test, y_train, y_test, TUNED,
                cv=3, tuner="grid", learning_rate={"low": 0.01, "high": 0.3},
                random_state=42,
            )

    def test_an_empty_block_names_catboost_and_how_to_opt_out(self, binary):
        X_train, X_test, y_train, y_test = binary
        with pytest.raises(ValueError) as exc:
            compute_catboost_opt(X_train, X_test, y_train, y_test, TUNED, cv=3)
        message = str(exc.value)
        assert "gridsearch_catboost_args" in message
        assert "grid_search: False" in message


# ======================================================================================
# TabPFN: laziness, gates and limits
# ======================================================================================


class TestTabPFNStaysLazy:
    """The import-order invariant that ``qbiocode.utils._openmp`` exists to maintain."""

    def test_importing_the_module_does_not_import_torch_or_tabpfn(self):
        """Run in a subprocess: import order cannot be undone within one interpreter."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import sys
                    import qbiocode.learning.compute_tabpfn  # noqa: F401
                    print("torch" in sys.modules, "tabpfn" in sys.modules)
                    """
                ),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert completed.stdout.strip() == "False False", (
            "importing compute_tabpfn pulled in tabpfn or torch. A module-level "
            "`from tabpfn import TabPFNClassifier` would do that, and it maps torch's "
            "OpenMP runtime into every QBioCode process -- see "
            "tests/test_openmp_import_order.py and qbiocode.utils._openmp."
        )

    def test_importing_the_package_does_not_import_torch_either(self):
        """``qbiocode/__init__.py`` re-exports the TabPFN functions; same guarantee."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import sys
                    import qbiocode
                    assert callable(qbiocode.compute_tabpfn)
                    assert callable(qbiocode.compute_tabpfn_opt)
                    print("torch" in sys.modules)
                    """
                ),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert completed.stdout.strip() == "False"

    def test_availability_can_be_probed_without_importing(self):
        """``tabpfn_is_available`` uses ``find_spec``, so it must not import anything."""
        source = inspect.getsource(tabpfn_module.tabpfn_is_available)
        assert "find_spec" in source
        assert isinstance(tabpfn_is_available(), bool)


class TestTabPFNClassLimit:
    """The one pretraining limit that ``ignore_pretraining_limits`` cannot waive."""

    def test_more_classes_than_the_head_supports_is_rejected(self):
        with pytest.raises(ValueError) as exc:
            _check_class_count(np.arange(TABPFN_MAX_CLASSES + 1), "tabpfn")
        message = str(exc.value)
        assert str(TABPFN_MAX_CLASSES) in message, "must state the limit"
        assert str(TABPFN_MAX_CLASSES + 1) in message, "must state the actual class count"
        assert "ignore_pretraining_limits" in message, "must say the flag will not help"

    def test_exactly_the_limit_is_allowed(self):
        _check_class_count(np.arange(TABPFN_MAX_CLASSES), "tabpfn")

    def test_the_limit_is_checked_before_the_weights_are_touched(self):
        """An 11-class dataset must fail on its class count, not on a license error.

        Order matters: the class check is cheap and local, the weight load is neither.
        Reversed, a user with no token would be told to accept a license for a dataset
        TabPFN could never handle anyway.
        """
        y = np.arange(TABPFN_MAX_CLASSES + 1)
        X = np.zeros((len(y), 3))
        if not tabpfn_is_available():
            pytest.skip("the [tabpfn] extra is not installed")
        with pytest.raises(ValueError, match="at most"):
            compute_tabpfn(X, X, y, y, UNTUNED)


class TestTabPFNErrorMessages:
    """The three ways TabPFN can be unusable, each named precisely."""

    def test_a_missing_extra_names_the_extra(self, monkeypatch):
        monkeypatch.setattr(tabpfn_module, "tabpfn_is_available", lambda: False)
        monkeypatch.setattr(tabpfn_module, "_TABPFN_CLASSIFIER", None)
        with pytest.raises(ImportError) as exc:
            _load_tabpfn_classifier()
        message = str(exc.value)
        assert 'pip install "qbiocode[tabpfn]"' in message
        assert "requirements-tabpfn.txt" in message
        assert "model" in message, "must say how to run without it"

    @pytest.mark.parametrize(
        "name", ["TabPFNLicenseError", "TabPFNHuggingFaceGatedRepoError"]
    )
    def test_a_gated_weight_failure_is_translated(self, name):
        gated = type(name, (Exception,), {})
        explained = _explain_weight_access_failure(gated("accept the license"), "tabpfn")
        assert isinstance(explained, ImportError)
        message = str(explained)
        assert "accept the license" in message, "upstream's own instructions must survive"
        assert "TABPFN_TOKEN" in message
        assert name in message, "must name the underlying error type"

    def test_an_unrelated_error_is_not_translated(self):
        """Swallowing every exception here would hide real bugs as licensing problems."""
        assert _explain_weight_access_failure(ValueError("bad shape"), "tabpfn") is None
        assert _explain_weight_access_failure(MemoryError(), "tabpfn") is None

    @pytest.mark.skipif(
        not tabpfn_is_available(), reason="the [tabpfn] extra is not installed"
    )
    def test_an_unreachable_checkpoint_surfaces_as_our_error(self, binary):
        """Whatever this environment's state, a fit either works or explains itself.

        This is the test that would have caught the raw ``TabPFNLicenseError`` escaping
        from six frames inside ``model_loading`` with no mention of QBioCode.
        """
        X_train, X_test, y_train, y_test = binary
        try:
            frame = compute_tabpfn(
                X_train, X_test, y_train, y_test, UNTUNED, n_estimators=1,
                device="cpu", random_state=42,
            )
        except ImportError as error:
            assert "TABPFN_TOKEN" in str(error) or "pip install" in str(error), (
                f"a weights failure must explain itself, got: {error}"
            )
        else:
            assert np.isfinite(metrics_of(frame)["accuracy"])


class TestTabPFNFits:
    """Everything that needs a real forward pass, skipped unless the weights load."""

    def test_untuned(self, binary, tabpfn_ready):
        X_train, X_test, y_train, y_test = binary
        frame = compute_tabpfn(
            X_train, X_test, y_train, y_test, UNTUNED,
            n_estimators=1, device="cpu", random_state=42,
        )
        assert np.isfinite(metrics_of(frame)["accuracy"])

    def test_tuned_reports_only_what_was_searched(self, binary, tabpfn_ready):
        X_train, X_test, y_train, y_test = binary
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = compute_tabpfn_opt(
                X_train, X_test, y_train, y_test, TUNED,
                cv=2, n_trials=2, n_estimators=[1, 2], device="cpu", random_state=42,
            )
        best = metrics_of(frame)["BestParams_Tuned"]
        assert set(best) == {"n_estimators"}, (
            f"only n_estimators was searched, but best_params holds {best}. The fixed "
            "settings (device, model_path) must not be reported as tuned."
        )

    def test_the_same_seed_gives_the_same_score(self, binary, tabpfn_ready):
        X_train, X_test, y_train, y_test = binary
        scores = [
            metrics_of(
                compute_tabpfn(
                    X_train, X_test, y_train, y_test, UNTUNED,
                    n_estimators=2, device="cpu", random_state=11,
                )
            )["accuracy"]
            for _ in range(2)
        ]
        assert scores[0] == scores[1], f"two runs at seed 11 disagreed: {scores}"


# ======================================================================================
# Registry wiring: the package surface, the dispatcher and the config
# ======================================================================================


class TestBothModelsAreProperlyRegistered:
    """A learner that exists but is not reachable from a config is not integrated."""

    @pytest.mark.parametrize("name", ["catboost", "tabpfn"])
    def test_the_dispatcher_knows_both_names_and_their_opt_twins(self, name):
        """``model_run`` builds its table locally, so read it back out of the source."""
        source = inspect.getsource(sys.modules["qbiocode.evaluation.model_run"])
        assert f'"{name}": compute_{name},' in source
        assert f'"{name}_opt": compute_{name}_opt,' in source

    @pytest.mark.parametrize(
        "attribute",
        [
            "compute_catboost",
            "compute_catboost_opt",
            "compute_tabpfn",
            "compute_tabpfn_opt",
        ],
    )
    def test_exported_from_the_package_root_and_declared_in_all(self, attribute):
        assert callable(getattr(qbiocode, attribute))
        assert attribute in qbiocode.__all__

    @pytest.mark.parametrize(
        "attribute",
        [
            "compute_catboost",
            "compute_catboost_opt",
            "compute_tabpfn",
            "compute_tabpfn_opt",
        ],
    )
    def test_exported_from_the_learning_subpackage(self, attribute):
        import qbiocode.learning as learning

        assert callable(getattr(learning, attribute))
        assert attribute in learning.__all__

    @pytest.mark.parametrize("name", ["catboost", "tabpfn"])
    def test_both_accept_random_state_so_qprofiler_can_seed_them(self, name):
        """``_seeded_kwargs`` fills ``random_state`` only if the signature has it.

        A learner missing the parameter is silently left unseeded, which is how the
        decision-tree rows used to differ between two runs at the same seed.
        """
        for suffix in ("", "_opt"):
            function = getattr(qbiocode, f"compute_{name}{suffix}")
            parameters = inspect.signature(function).parameters
            assert "random_state" in parameters, (
                f"compute_{name}{suffix} takes no random_state, so QProfiler cannot "
                f"seed it and two runs at one seed will disagree"
            )

    @pytest.mark.parametrize("name", ["catboost", "tabpfn"])
    def test_an_unknown_model_error_advertises_them(self, name, binary):
        """The message lists what is available; a new learner must appear in it."""
        from qbiocode.evaluation.model_run import model_run

        X_train, X_test, y_train, y_test = binary
        with pytest.raises(ValueError) as exc:
            model_run(
                X_train, X_test, y_train, y_test, "k",
                {"model": ["not_a_model"], "grid_search": False},
            )
        assert name in str(exc.value)

    @pytest.mark.parametrize("name", ["catboost", "tabpfn"])
    def test_grid_search_validation_accepts_them(self, name, binary):
        """``model_run`` rejects a classical model with no ``_opt`` twin.

        So this both proves the twin is registered and guards the branch that would
        otherwise refuse the model outright.
        """
        from qbiocode.evaluation.model_run import model_run

        X_train, X_test, y_train, y_test = binary
        with pytest.raises(ValueError) as exc:
            model_run(
                X_train, X_test, y_train, y_test, "k",
                {"model": [name, "not_a_model"], "grid_search": True},
            )
        assert "no '_opt' implementation" not in str(exc.value), (
            f"{name} was reported as lacking an _opt twin"
        )


class TestTheShippedConfigBlocks:
    """The defaults ship in two config files, and both must be usable as written."""

    @staticmethod
    def configs():
        import pathlib

        import yaml

        root = pathlib.Path(__file__).resolve().parents[1]
        for relative in (
            "qbiocode/apps/qprofiler/configs/config.yaml",
            "tutorial/QProfiler/configs/config.yaml",
        ):
            with open(root / relative) as handle:
                yield relative, yaml.safe_load(handle)

    @pytest.mark.parametrize("key", [
        "catboost_args", "gridsearch_catboost_args",
        "tabpfn_args", "gridsearch_tabpfn_args",
    ])
    def test_every_config_defines_the_block(self, key):
        for relative, config in self.configs():
            assert key in config, f"{relative} has no {key} block"
            assert config[key], f"{relative}'s {key} block is empty"

    def test_the_shipped_catboost_block_avoids_the_bootstrap_trap(self):
        """Defaults must not need the pinning rescue in the first place.

        ``_resolve_bootstrap`` handles ``subsample`` correctly, but a shipped default
        that relies on being rescued teaches the wrong thing by example.
        """
        for relative, config in self.configs():
            block = config["gridsearch_catboost_args"]
            overlap = {"subsample", "bagging_temperature", "bootstrap_type"} & set(block)
            assert not overlap, (
                f"{relative}'s gridsearch_catboost_args names {sorted(overlap)}. Those "
                "are the data-dependent bootstrap parameters; keep the shipped default "
                "clear of them."
            )

    def test_the_shipped_blocks_are_accepted_by_the_real_functions(
        self, tmp_path, monkeypatch, binary
    ):
        """A config block nothing has ever executed is a guess, not a default."""
        monkeypatch.chdir(tmp_path)
        X_train, X_test, y_train, y_test = binary
        _, config = next(iter(self.configs()))

        frame = compute_catboost(
            X_train, X_test, y_train, y_test, UNTUNED,
            **config["catboost_args"], random_state=42,
        )
        assert np.isfinite(metrics_of(frame)["accuracy"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = compute_catboost_opt(
                X_train, X_test, y_train, y_test, TUNED,
                cv=2, n_trials=2, random_state=42,
                **config["gridsearch_catboost_args"],
            )
        assert np.isfinite(metrics_of(frame)["accuracy"])

    def test_the_tabpfn_blocks_only_name_real_parameters(self):
        """Caught early: a typo'd key becomes an unexpected-keyword TypeError at run time."""
        for relative, config in self.configs():
            for key, function in (
                ("tabpfn_args", compute_tabpfn),
                ("gridsearch_tabpfn_args", compute_tabpfn_opt),
            ):
                allowed = set(inspect.signature(function).parameters)
                unknown = set(config[key]) - allowed
                assert not unknown, (
                    f"{relative}'s {key} names {sorted(unknown)}, which "
                    f"{function.__name__} does not accept"
                )

    def test_the_catboost_blocks_only_name_real_parameters(self):
        for relative, config in self.configs():
            for key, function in (
                ("catboost_args", compute_catboost),
                ("gridsearch_catboost_args", compute_catboost_opt),
            ):
                allowed = set(inspect.signature(function).parameters)
                unknown = set(config[key]) - allowed
                assert not unknown, (
                    f"{relative}'s {key} names {sorted(unknown)}, which "
                    f"{function.__name__} does not accept"
                )


class TestTheQplHeads:
    """CatBoost and TabPFN as classical heads on a quantum projection."""

    def test_catboost_is_a_default_head_and_tabpfn_is_opt_in(self):
        """An optional extra must not be a default head.

        ``compute_qpl`` warns and drops an unavailable head, so defaulting TabPFN on
        would make every ordinary QPL run emit a warning about an extra the user never
        asked for.
        """
        source = inspect.getsource(sys.modules["qbiocode.learning.compute_qpl"])
        assert '["rf", "mlp", "svc", "lr", "xgb", "catboost"]' in source
        assert '"tabpfn"' in source, "tabpfn must still be selectable"

    def test_the_catboost_head_pins_its_bootstrap_scheme(self):
        """Its grid varies ``subsample``, so an unpinned scheme breaks on multiclass."""
        from qbiocode.learning.compute_qpl import create_catboost_model

        model = create_catboost_model(42)
        params = model.estimator.get_params()
        assert params["bootstrap_type"] == "Bernoulli"
        assert params["allow_writing_files"] is False
        assert params["verbose"] is False
        assert "subsample" in model.param_distributions, (
            "if subsample is no longer searched, the pinning above can be reconsidered"
        )

    def test_the_catboost_head_is_seeded(self):
        from qbiocode.learning.compute_qpl import create_catboost_model

        model = create_catboost_model(42)
        assert model.estimator.get_params()["random_state"] == 42
        assert model.random_state == 42

    def test_the_tabpfn_head_is_not_wrapped_in_a_search(self):
        """40 x 5 transformer passes per projection is not a reasonable default.

        ``compute_qpl`` reads ``best_params_`` off each head, so this head is exactly
        why that read falls back to ``get_params()``.
        """
        from qbiocode.learning.compute_qpl import create_tabpfn_model

        if not tabpfn_is_available():
            pytest.skip("the [tabpfn] extra is not installed")
        model = create_tabpfn_model(42)
        assert not hasattr(model, "best_params_")
        assert not hasattr(model, "param_distributions")
        assert model.get_params()["random_state"] == 42

    def test_the_qpl_head_report_tolerates_an_unsearched_head(self):
        """The ``best_params`` fallback must be a real fallback, not a crash.

        A bare estimator has no ``best_params_``, and reading it unguarded raised
        AttributeError for the TabPFN head alone -- after the expensive quantum projection
        had already been computed.

        Asserted on the AST rather than on a source substring. The substring version named
        the variable (``getattr(model, ...)``) and so broke the moment that variable was
        renamed: ``compute_qpl`` used to rebind ``model`` -- its label parameter -- to each
        head's estimator, destroying the label, which is why a tuned QPL run was
        indistinguishable from an untuned one. Fixing that renamed the estimator to
        ``estimator`` and this test failed, having pinned the spelling instead of the
        behaviour. What matters is that the attribute is reached through a defaulting
        ``getattr`` with a fallback, whatever the variable is called.
        """
        module = sys.modules["qbiocode.learning.compute_qpl"]
        tree = ast.parse(inspect.getsource(module))
        guarded = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) == 3
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "best_params_"
        ]
        assert guarded, (
            "compute_qpl no longer reads best_params_ through a 3-argument getattr, so an "
            "estimator without one (the TabPFN head) would raise AttributeError after the "
            "projection had been computed"
        )
        # And the fallback must actually be used, not discarded: the getattr sits on the
        # left of a `or`, whose right side supplies the unsearched head's own parameters.
        assert any(
            isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.Or)
            and any(value is call for value in parent.values)
            for call in guarded
            for parent in ast.walk(tree)
            if isinstance(parent, ast.BoolOp)
        ), "the getattr default is not feeding an `or` fallback"

    @pytest.mark.parametrize("head", ["catboost", "tabpfn"])
    def test_an_unavailable_head_is_dropped_rather_than_fatal(self, head):
        """One unusable head must not waste a projection that already cost real time."""
        source = inspect.getsource(sys.modules["qbiocode.learning.compute_qpl"])
        assert f'classical_models = [m for m in classical_models if m != "{head}"]' in source
        assert f'elif method == "{head}":' in source


class TestTheSageSurrogate:
    """CatBoost as a QuantumSage regression surrogate, alongside the XGBoost one."""

    def test_catboost_optuna_is_a_valid_sage_type(self):
        from qbiocode.apps.sage.sage import QuantumSage

        source = inspect.getsource(QuantumSage.train_sub_sages)
        assert "'catboost_optuna'" in source
        assert hasattr(QuantumSage, "_sage_catboost_optuna")

    def test_the_cli_accepts_it_and_maps_the_alias(self):
        source = inspect.getsource(sys.modules["qbiocode.apps.sage.sage"])
        assert "'catboost', 'catboost_optuna'" in source, "CLI --model-type choices"
        assert "'catboost': 'catboost_optuna'" in source, "alias map"

    def test_the_surrogate_returns_the_same_keys_as_its_xgboost_twin(self):
        """Downstream reporting reads these keys positionally across sage types."""
        import pandas as pd

        from qbiocode.apps.sage.sage import QuantumSage

        rng = np.random.default_rng(0)
        X_train = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))
        X_test = pd.DataFrame(rng.normal(size=(12, 4)), columns=list("abcd"))
        y_train = pd.Series(X_train["a"] * 2 + rng.normal(scale=0.1, size=40))
        y_test = pd.Series(X_test["a"] * 2)

        class Stub:
            _seed = 42

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = QuantumSage._sage_catboost_optuna(
                Stub(), X_train, X_test, y_train, y_test, n_iter=3, cv=3
            )
        assert set(result) == {
            "fit_model", "preds", "y_test", "params", "mae", "mse", "rmse", "r2", "study",
        }
        assert np.isfinite(result["r2"])
        # The settings that keep n_iter*cv fits quiet and litter-free.
        assert result["params"]["allow_writing_files"] is False
        assert result["params"]["verbose"] is False
        assert result["params"]["bootstrap_type"] == "Bernoulli"

    def test_the_surrogate_leaves_no_catboost_info_behind(self, tmp_path, monkeypatch):
        import pandas as pd

        from qbiocode.apps.sage.sage import QuantumSage

        monkeypatch.chdir(tmp_path)
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame(rng.normal(size=(30, 3)), columns=list("abc"))
        X_test = pd.DataFrame(rng.normal(size=(10, 3)), columns=list("abc"))
        y_train = pd.Series(X_train["a"])
        y_test = pd.Series(X_test["a"])

        class Stub:
            _seed = 42

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            QuantumSage._sage_catboost_optuna(
                Stub(), X_train, X_test, y_train, y_test, n_iter=2, cv=2
            )
        assert sorted(p.name for p in tmp_path.iterdir()) == []


class TestThePackagingDeclarations:
    """catboost is a base dependency; tabpfn is an extra. Neither by accident."""

    @staticmethod
    def read_pyproject():
        import pathlib

        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10
            tomllib = pytest.importorskip("tomli")
        root = pathlib.Path(__file__).resolve().parents[1]
        with open(root / "pyproject.toml", "rb") as handle:
            return tomllib.load(handle)

    @staticmethod
    def read_requirements(name):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        specs = []
        for line in (root / "requirements" / name).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line and not line.startswith("-"):
                specs.append(line)
        return specs

    def test_catboost_is_a_base_dependency(self):
        """``compute_catboost`` is imported unconditionally, so it must be declared."""
        assert "catboost" in self.read_requirements("requirements-base.txt")

    def test_tabpfn_is_not_a_base_dependency(self):
        """It brings torch's ecosystem plus mlx/lightgbm/hf-hub, which a bare install
        must not carry -- and `import qbiocode` must not pull torch in."""
        base = self.read_requirements("requirements-base.txt")
        assert not [spec for spec in base if spec.lower().startswith("tabpfn")], (
            "tabpfn reached requirements-base.txt; a default install would then carry "
            "a model that cannot fit without a manual license step"
        )

    def test_the_tabpfn_extra_exists_and_matches_its_requirements_file(self):
        extras = self.read_pyproject()["project"]["optional-dependencies"]
        assert sorted(extras["tabpfn"]) == sorted(self.read_requirements("requirements-tabpfn.txt"))

    def test_the_tabpfn_extra_is_part_of_all(self):
        extras = self.read_pyproject()["project"]["optional-dependencies"]
        assert "tabpfn" in extras["all"][0]

    def test_the_dev_environment_installs_the_tabpfn_tier(self):
        """Otherwise the tabpfn tests skip in the one environment meant to run them."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        text = (root / "requirements" / "requirements.txt").read_text()
        assert "-r requirements-tabpfn.txt" in text

class TestTheBugsFoundInReview:
    """Defects in the first pass of this integration, each with the shape it failed in.

    Named after ``test_optuna_tuning.TestTheBugsFoundAfterFirstPass``, and for the same
    reason: every one of these passed the original suite, so the tests are the record of
    what the suite was not looking at.
    """

    # ---------------------------------------------------------------- guard divergence

    def test_the_untuned_path_rejects_both_bootstrap_parameters(self, binary):
        """``catboost_args`` naming both raised a bare CatBoostError.

        The tuned path rejected the combination up front with a message naming the config
        key; the untuned path had its own copy of the pinning, which set ``Bernoulli``
        whenever ``subsample`` appeared and then left CatBoost to complain about
        ``bagging_temperature``::

            CatBoostError: bagging temperature available for bayesian bootstrap only

        Both now call ``_resolve_bootstrap``, so they cannot disagree again.
        """
        X_train, X_test, y_train, y_test = binary
        with pytest.raises(ValueError) as exc:
            compute_catboost(
                X_train, X_test, y_train, y_test, UNTUNED,
                iterations=10, subsample=0.8, bagging_temperature=0.5, random_state=42,
            )
        message = str(exc.value)
        assert "catboost_args" in message, "must name the untuned block, not the tuned one"
        assert "gridsearch_catboost_args" not in message, (
            "an untuned config must not be told to edit the tuned block"
        )
        assert "subsample" in message and "bagging_temperature" in message

    def test_the_untuned_path_rejects_subsample_under_an_explicit_bayesian_bootstrap(
        self, binary
    ):
        """A check the untuned path did not have at all, only the tuned one."""
        X_train, X_test, y_train, y_test = binary
        with pytest.raises(ValueError, match="Bayesian"):
            compute_catboost(
                X_train, X_test, y_train, y_test, UNTUNED,
                iterations=10, subsample=0.8, bootstrap_type="Bayesian", random_state=42,
            )

    def test_both_paths_share_one_guard(self):
        """The structural half: a second copy of the pinning is how they diverged.

        Asserted on the source because the behavioural tests above can only catch the
        cases someone thought to write; a reintroduced local copy would pass them and
        drift again on the next case.
        """
        source = inspect.getsource(catboost_module)
        assert source.count('params["bootstrap_type"] = ') == 0, (
            "compute_catboost is pinning bootstrap_type itself again instead of "
            "delegating to _resolve_bootstrap"
        )
        assert source.count("_resolve_bootstrap(") == 3, (
            "expected exactly three references: the definition, the untuned call and "
            "the tuned call"
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"subsample": 0.8},
            {"bagging_temperature": 0.5},
            {"subsample": 0.8, "bootstrap_type": "Bernoulli"},
            {"bagging_temperature": 0.5, "bootstrap_type": "Bayesian"},
        ],
    )
    def test_the_legal_combinations_still_fit(self, tmp_path, monkeypatch, binary, kwargs):
        """Tightening the guard must not start refusing configurations that worked."""
        monkeypatch.chdir(tmp_path)
        X_train, X_test, y_train, y_test = binary
        frame = compute_catboost(
            X_train, X_test, y_train, y_test, UNTUNED,
            iterations=10, random_state=42, **kwargs,
        )
        assert np.isfinite(metrics_of(frame)["accuracy"])

    # ------------------------------------------------------- a range for a string enum

    def test_a_range_for_bootstrap_type_is_rejected_by_name(self, binary):
        """It was silently destroyed rather than refused.

        ``list({'low': 1, 'high': 3})`` is ``['low', 'high']``, which satisfied the
        compatibility checks; ``build_search_space`` then read the mapping as an *integer*
        range and proposed ``bootstrap_type=2``, so CatBoost failed with
        ``Can't parse parameter "type" with value: 2`` -- naming neither the config entry
        nor the actual mistake.
        """
        X_train, X_test, y_train, y_test = binary
        with pytest.raises(ValueError) as exc:
            compute_catboost_opt(
                X_train, X_test, y_train, y_test, TUNED,
                cv=2, n_trials=2, iterations=[10], subsample=[0.8],
                bootstrap_type={"low": 1, "high": 3}, random_state=42,
            )
        message = str(exc.value)
        assert "bootstrap_type" in message, "must name the offending key"
        assert "list" in message, "must say what to write instead"
        assert "parse parameter" not in message, "the CatBoost error must not be what surfaces"

    # ------------------------------------------------- a searched dimension that is inert

    def test_min_data_in_leaf_is_not_searchable(self, tmp_path, monkeypatch, binary):
        """It was searchable and inert; several values are now refused, not warned about.

        CatBoost accepts it under every grow policy and honours it under only two, so at the
        default ``SymmetricTree`` every value produces an identical model. A warning left the
        wasted fits to happen anyway; refusing does not.
        """
        monkeypatch.chdir(tmp_path)
        X_train, X_test, y_train, y_test = binary
        with pytest.raises(ValueError) as exc:
            compute_catboost_opt(
                X_train, X_test, y_train, y_test, TUNED,
                cv=2, n_trials=2, iterations=[10], min_data_in_leaf=[1, 40],
                random_state=42,
            )
        message = str(exc.value)
        assert "min_data_in_leaf" in message
        assert "grow_policy" in message, "must say what would make it take effect"
        assert "gridsearch_catboost_args" in message, "must name the config block"

    def test_a_single_min_data_in_leaf_value_is_applied_to_every_trial(
        self, tmp_path, monkeypatch, binary
    ):
        """Still usable -- it is passed fixed rather than dropped, and does not warn."""
        monkeypatch.chdir(tmp_path)
        X_train, X_test, y_train, y_test = binary
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            frame = compute_catboost_opt(
                X_train, X_test, y_train, y_test, TUNED,
                cv=2, n_trials=2, iterations=[10], min_data_in_leaf=5,
                grow_policy=["Depthwise"], random_state=42,
            )
        assert not [w for w in caught if "min_data_in_leaf" in str(w.message)]
        best = metrics_of(frame)["BestParams_Tuned"]
        assert "min_data_in_leaf" not in best, "it is fixed, so it is not a tuned result"

    @pytest.mark.parametrize("name", ["compute_catboost", "compute_catboost_opt"])
    def test_both_twins_still_accept_it(self, name):
        """Dropping it from ``_opt`` alone would recreate the ``loss_function`` asymmetry:
        ``model_run`` splats the config block, so a name only one twin takes is a
        ``TypeError`` raised inside a joblib worker."""
        assert "min_data_in_leaf" in inspect.signature(getattr(qbiocode, name)).parameters

    def test_it_is_absent_from_the_searchable_candidates(self):
        source = inspect.getsource(catboost_module.compute_catboost_opt)
        candidates = source.split("candidates = {")[1].split("}")[0]
        assert "min_data_in_leaf" not in candidates

    def test_the_sage_surrogate_does_not_search_the_inert_dimension(self):
        """It did, so every CatBoost Sage trial spent a dimension on nothing."""
        from qbiocode.apps.sage.sage import QuantumSage

        source = inspect.getsource(QuantumSage._sage_catboost_optuna)
        assert "suggest_int('min_data_in_leaf'" not in source, (
            "the Sage CatBoost surrogate searches min_data_in_leaf at the default "
            "SymmetricTree policy, where CatBoost ignores it"
        )
        # And the dimensions it does search must all be ones CatBoost honours by default.
        assert "suggest_float('subsample'" in source
        assert "bootstrap_type" in source, "subsample needs the scheme pinned"

    # --------------------------------------------- a survivable failure that was fatal

    def test_a_gated_tabpfn_qpl_head_is_dropped_rather_than_fatal(self):
        """It killed the run *after* the quantum projection had been computed.

        The availability filtering cannot catch this one: the extra is installed and the
        import succeeds, and only the fit discovers that the checkpoint is gated. So the
        head has to be dropped at fit time, which is what this function already promises
        for an unusable xgboost or catboost.
        """
        source = inspect.getsource(sys.modules["qbiocode.learning.compute_qpl"])
        assert "_explain_weight_access_failure(error, method_qpl)" in source, (
            "compute_qpl no longer translates a weights failure per head, so a gated "
            "TabPFN takes the whole projection down with it"
        )
        assert "could not run and was skipped" in source

    def test_qpl_reports_something_useful_when_every_head_is_dropped(self):
        """``pd.concat([])`` raises "No objects to concatenate", which explains nothing.

        Reachable now that a gated head is skipped instead of fatal, and reachable before
        via the unknown-model-name branch.
        """
        source = inspect.getsource(sys.modules["qbiocode.learning.compute_qpl"])
        assert "could be fitted" in source, "the empty-result case is unguarded again"
        index_guard = source.index("if not model_res:")
        index_concat = source.index("model_res = pd.concat(model_res)")
        assert index_guard < index_concat, "the guard must precede the concat it protects"


class TestTheModelVersionIsPinned:
    """Which checkpoint TabPFN loads is a licensing decision, not only a modelling one.

    TabPFN's *code* is Apache 2.0 plus an attribution clause. Its *weights* are licensed
    per version, and they diverge sharply:

    ==========  ============================================  ===============
    version     weights licence                               commercial use
    ==========  ============================================  ===============
    ``v2``      Prior Labs License v1.1 (Apache 2.0 + attr)   permitted
    ``v2.5``    TABPFN-2.5 Non-Commercial License             no
    ``v2.6``    TABPFN-2.6 Non-Commercial License             no
    ``v3``      TABPFN-3 Non-Commercial License               no
    ==========  ============================================  ===============

    Upstream's constructor defaults to ``v3``. QBioCode is Apache-2.0 software whose users
    include companies, so inheriting that default would quietly impose a non-commercial,
    non-production licence on them -- and it also cannot be downloaded unattended, because
    reaching it requires an interactive licence acceptance. So ``v2`` is pinned, and the
    others are reachable but announce themselves.
    """

    def test_the_default_is_the_commercially_usable_version(self):
        assert TABPFN_DEFAULT_VERSION == "v2", (
            "the pinned default must stay v2: it is the only version whose weights permit "
            "commercial use, and the only one that needs no licence acceptance"
        )

    @pytest.mark.parametrize("name", ["v2", "V2", "2", "v2.5", "V2_5", "2.5", "v3", "V3"])
    def test_the_spellings_people_write_are_accepted(self, name):
        """A config is written by hand; rejecting a spelling is an obstacle, not a guard."""
        assert normalise_model_version(name) in {"v2", "v2.5", "v2.6", "v3"}

    def test_an_enum_member_is_accepted(self):
        """``tabpfn.constants.ModelVersion.V2_5`` is the natural thing to reach for."""
        pytest.importorskip("tabpfn")
        from tabpfn.constants import ModelVersion

        assert normalise_model_version(ModelVersion.V2) == "v2"
        assert normalise_model_version(ModelVersion.V2_5) == "v2.5"

    def test_an_unknown_version_lists_the_choices_and_names_the_safe_one(self):
        with pytest.raises(ValueError) as exc:
            normalise_model_version("v9")
        message = str(exc.value)
        assert "v2" in message and "v3" in message, "must list what is available"
        assert "commercial" in message, "must say which one is the safe default and why"

    @pytest.mark.parametrize("version", ["v2.5", "v2.6", "v3"])
    def test_selecting_a_non_commercial_version_warns_and_names_the_licence(self, version):
        """It must not be possible to end up on a non-commercial checkpoint silently."""
        pytest.importorskip("tabpfn")
        with pytest.warns(UserWarning) as caught:
            resolve_model_path("auto", version)
        message = "\n".join(str(w.message) for w in caught)
        assert "Non-Commercial" in message, "must name the licence"
        assert "non-production" in message, "non-production is the half people miss"
        assert TABPFN_DEFAULT_VERSION in message, "must point at the unrestricted default"

    def test_the_default_version_does_not_warn(self):
        """A warning on the safe path would train people to ignore it."""
        pytest.importorskip("tabpfn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_model_path("auto", TABPFN_DEFAULT_VERSION)
        licence_warnings = [w for w in caught if "Non-Commercial" in str(w.message)]
        assert not licence_warnings

    def test_the_default_resolves_to_the_v2_checkpoint(self):
        pytest.importorskip("tabpfn")
        path = resolve_model_path("auto", TABPFN_DEFAULT_VERSION)
        assert "v2" in str(path), f"expected a v2 checkpoint, got {path}"
        assert "v3" not in str(path)

    def test_an_explicit_model_path_wins_over_the_version(self):
        """Pointing at a local checkpoint is more specific, and is how to run offline."""
        assert resolve_model_path("/tmp/my-checkpoint.ckpt", "v3") == "/tmp/my-checkpoint.ckpt"

    @pytest.mark.parametrize("function", ["compute_tabpfn", "compute_tabpfn_opt"])
    def test_both_learners_take_model_version(self, function):
        parameters = inspect.signature(getattr(qbiocode, function)).parameters
        assert "model_version" in parameters
        assert parameters["model_version"].default == TABPFN_DEFAULT_VERSION

    def test_the_version_is_not_searchable(self):
        """Searching it would compare different models -- under different licences -- and
        report the winner as though it were a hyperparameter."""
        source = inspect.getsource(tabpfn_module.compute_tabpfn_opt)
        candidates = source.split("candidates = {")[1].split("}")[0]
        assert "model_version" not in candidates
        assert "model_path" not in candidates

    @pytest.mark.parametrize("key", ["tabpfn_args", "gridsearch_tabpfn_args"])
    def test_the_shipped_configs_pin_the_version_explicitly(self, key):
        """Implicit is not good enough for a licensing choice -- it should be readable."""
        for relative, config in TestTheShippedConfigBlocks.configs():
            assert config[key].get("model_version") == TABPFN_DEFAULT_VERSION, (
                f"{relative}'s {key} does not pin model_version to "
                f"{TABPFN_DEFAULT_VERSION!r}"
            )


class TestTheOpenMpGuard:
    """A TabPFN fit used to kill the process outright, with no traceback.

    ``qbiocode/__init__.py`` initialises xgboost's ``libomp`` first, deliberately. Importing
    ``tabpfn`` brings torch's copy in as a second LLVM OpenMP runtime under the same install
    name, and the second one to open a parallel region dies below Python: exit 139, no
    exception, and a notebook front end reporting only "kernel died". Measured on this tree,
    a ``compute_tabpfn`` call in a process that imported ``qbiocode`` exited 139 without
    ``OMP_NUM_THREADS`` set and returned a score with it.
    """

    def test_the_variable_is_set_when_absent(self, monkeypatch):
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.setattr(sys, "platform", "darwin")
        with pytest.warns(UserWarning, match="OMP_NUM_THREADS"):
            assert _cap_openmp_threads() is True
        assert os.environ["OMP_NUM_THREADS"] == "1"

    def test_an_existing_value_is_respected(self, monkeypatch):
        """Overriding a deliberate choice silently would be worse than the crash."""
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        monkeypatch.setattr(sys, "platform", "darwin")
        with pytest.warns(UserWarning, match="SIGSEGV"):
            assert _cap_openmp_threads() is False
        assert os.environ["OMP_NUM_THREADS"] == "8", "must not overwrite the caller's value"

    def test_a_value_of_one_is_respected_without_a_warning(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        monkeypatch.setattr(sys, "platform", "darwin")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert _cap_openmp_threads() is False
        assert not caught, "already correct; nothing to say"

    def test_it_does_nothing_off_macos(self, monkeypatch):
        """The duplicate-runtime crash is macOS-specific; capping threads elsewhere is a
        needless performance ceiling."""
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.setattr(sys, "platform", "linux")
        assert _cap_openmp_threads() is False
        assert "OMP_NUM_THREADS" not in os.environ

    def test_the_loader_calls_it_before_importing_tabpfn(self):
        """Ordering is the whole point: torch reads the variable as its runtime starts."""
        source = inspect.getsource(tabpfn_module._load_tabpfn_classifier)
        assert "_cap_openmp_threads()" in source
        assert source.index("_cap_openmp_threads()") < source.index('import_module("tabpfn")')

    def test_a_fit_in_a_process_that_imported_qbiocode_does_not_crash(self):
        """The regression itself, in a fresh interpreter: exit 139 is the failure."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import warnings
                    warnings.simplefilter("ignore")
                    import numpy as np
                    import qbiocode                      # xgboost's OpenMP runtime first
                    from qbiocode.learning.compute_tabpfn import compute_tabpfn
                    rng = np.random.default_rng(0)
                    X = rng.normal(size=(40, 3))
                    y = (X[:, 0] > 0).astype(int)
                    try:
                        compute_tabpfn(X[:28], X[28:], y[:28], y[28:],
                                       {"grid_search": False}, n_estimators=1,
                                       device="cpu", random_state=0)
                        print("FITTED")
                    except ImportError:
                        print("UNAVAILABLE")   # no extra, or weights unreachable
                    """
                ),
            ],
            capture_output=True,
            text=True,
            timeout=1800,
            env={k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"},
        )
        assert completed.returncode >= 0, (
            f"child killed by signal {-completed.returncode} (SIGSEGV is 11): the OpenMP "
            f"guard has regressed.\nstdout: {completed.stdout!r}"
        )
        assert completed.returncode == 0, (
            f"exit {completed.returncode}\nstderr: {completed.stderr[-2000:]}"
        )
        assert completed.stdout.strip() in {"FITTED", "UNAVAILABLE"}, completed.stdout
