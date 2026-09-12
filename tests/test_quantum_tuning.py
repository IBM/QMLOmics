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

"""Optuna tuning for the quantum classifiers, and the guards that keep it affordable.

The quantum learners have no estimator to hand to ``cross_val_score`` -- ``encoding``
and ``reps`` select a feature map, which selects a kernel or an ansatz, and
``compute_qsvc`` and friends build that chain internally. So their ``_opt`` wrappers
tune the *function*, scoring each trial on one stratified holdout carved out of the
training data.

Three things here are cost guards rather than correctness checks, and they matter more
than usual: tuning is off unless asked for twice, a quantum fit is scored once rather
than k times, and tuning against a real device is refused outright. Each trial on
hardware would be a queued job billed to the user's instance, and the failure mode is
silent -- the run simply never appears to finish.
"""

import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture
def data():
    """Small, scaled and separable. Scaling matters: the ZZ feature map encodes
    magnitudes as rotation angles, so unscaled features land anywhere on the circle.

    Split before scaling, with the scaler fitted on the training rows only. Fitting on
    the whole matrix would leak test-set statistics into training -- harmless to what
    these tests assert, but a fixture is example code, and the leak would contradict
    ``scale_train_test``, which is the protocol the library itself uses.
    """
    from qbiocode import scale_train_test

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test = scale_train_test(X[:30], X[30:], scaling="MinMaxScaler")
    return X_train, X_test, y[:30], y[30:]


@pytest.fixture
def sim_args(tmp_path):
    """The keys the quantum stack reads. `shots` is required for the sampler.

    `pqk_projection_dir` points into tmp_path so PQK's projection cache never lands in
    the repository root, and so one test's projections cannot be mistaken for another's:
    the cache name is built from `data_key` and a feature-map fingerprint but not the
    row count, so a stale file from a differently-sized split trips PQK's row-count guard.
    """
    return {
        "backend": "simulator",
        "shots": 512,
        "seed": 42,
        "grid_search": True,
        "pqk_projection_dir": str(tmp_path / "pqk_projections"),
        "qpl_projection_dir": str(tmp_path / "qpl_projections"),
    }


# One cheap block per model: `reps: [1]` keeps every circuit shallow, and each space is
# finite and tiny so the budget cap makes the studies exhaustive and quick.
BLOCKS = [
    ("qsvc", {"encoding": ["Z", "ZZ"], "reps": [1]}),
    ("vqc", {"encoding": ["Z"], "reps": [1], "maxiter": [10, 20]}),
    ("qnn", {"encoding": ["Z"], "reps": [1], "maxiter": [10, 20]}),
    ("pqk", {"encoding": ["Z", "ZZ"], "reps": [1]}),
    ("qpl", {"encoding": ["Z"], "reps": [1, 2]}),
]


@pytest.mark.parametrize("model,block", BLOCKS, ids=[m for m, _ in BLOCKS])
def test_every_quantum_learner_tunes_and_reports_what_it_chose(model, block, data, sim_args):
    """The regression guard for the five ``compute_q*_opt`` wrappers."""
    import sys

    import qbiocode  # noqa: F401  -- orders the OpenMP runtimes
    from qbiocode.learning._tuning import _metric_dicts

    fn = getattr(sys.modules[f"qbiocode.learning.compute_{model}"], f"compute_{model}_opt")
    X_train, X_test, y_train, y_test = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = fn(X_train, X_test, y_train, y_test, sim_args, n_trials=2, **block)

    metrics = _metric_dicts(frame, model)
    assert metrics, f"compute_{model}_opt reported no metrics"
    for entry in metrics:
        assert np.isfinite(entry["accuracy"])
        tuned = entry["BestParams_Tuned"]
        for name, values in block.items():
            assert name in tuned, f"{name!r} was searched but is not reported in {sorted(tuned)}"
            assert tuned[name] in values, f"{name}={tuned[name]!r} is not one of {values}"


def test_qpl_reports_every_classical_head(data, sim_args):
    """QPL fits one classical head per model on the same projection.

    ``pd.concat`` of a frame per head gives several ``results_`` columns, each populated
    on its own row. A helper that assumed exactly one column scored only the first head
    and left the rest with an uncorrected time and parameter dict.
    """
    import qbiocode  # noqa: F401
    from qbiocode.learning._tuning import _metric_dicts
    from qbiocode.learning.compute_qpl import compute_qpl_opt

    X_train, X_test, y_train, y_test = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = compute_qpl_opt(
            X_train,
            X_test,
            y_train,
            y_test,
            sim_args,
            n_trials=1,
            encoding=["Z"],
            reps=[1],
        )
    metrics = _metric_dicts(frame, "qpl")
    assert len(metrics) > 1, f"expected one entry per classical head, got {len(metrics)}"
    # every head must carry the tuned parameters and the whole-search time
    assert all("BestParams_Tuned" in m for m in metrics)
    assert all(m["BestParams_Tuned"].get("encoding") == "Z" for m in metrics)
    assert (
        len({round(m["time"], 3) for m in metrics}) == 1
    ), "each head should report the same whole-search wall clock"


class TestTheCostGuards:
    """Quantum tuning is expensive, so it must never happen by accident."""

    def test_tuning_is_off_unless_asked_for_twice(self, data, sim_args):
        """``grid_search: True`` alone must not start tuning quantum models.

        Every config already naming a quantum model would otherwise get an order of
        magnitude slower on upgrade with no change on the user's part.
        """
        import qbiocode  # noqa: F401
        from qbiocode.evaluation.model_run import model_run

        X_train, X_test, y_train, y_test = data
        args = {
            **sim_args,
            "model": ["qsvc"],
            "n_jobs": 1,
            "q_seed": 42,
            "cross_validation": 3,
            "gridsearch_qsvc_args": {"reps": [1]},
            "qsvc_args": {"reps": 1},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = model_run(X_train, X_test, y_train, y_test, "k", args)
        # the untuned function is dispatched, so the column is not the `_opt` one
        assert "results_qsvc" in out, f"expected the untuned path, got {sorted(out)}"
        assert "results_qsvc_opt" not in out

    def test_hardware_tuning_is_refused_by_default(self):
        """One queued job per trial, billed to the user's instance."""
        from qbiocode.learning._tuning import ensure_tuning_is_affordable

        with pytest.raises(ValueError) as excinfo:
            ensure_tuning_is_affordable({"backend": "ibm_cleveland"}, "qsvc")
        msg = str(excinfo.value)
        assert "ibm_cleveland" in msg
        assert "allow_hardware_tuning" in msg, "must name the opt-out"

    def test_hardware_tuning_is_allowed_when_asked_for(self):
        from qbiocode.learning._tuning import ensure_tuning_is_affordable

        ensure_tuning_is_affordable(
            {"backend": "ibm_cleveland", "allow_hardware_tuning": True}, "qsvc"
        )

    @pytest.mark.parametrize("backend", ["simulator", "simulator_aer"])
    def test_simulators_are_free(self, backend):
        from qbiocode.learning._tuning import ensure_tuning_is_affordable

        ensure_tuning_is_affordable({"backend": backend}, "qsvc")

    def test_tune_quantum_without_grid_search_is_rejected(self, data, sim_args):
        """The two keys together mean something; alone, tune_quantum is a silent no-op."""
        from qbiocode.evaluation.model_run import model_run

        X_train, X_test, y_train, y_test = data
        args = {
            **sim_args,
            "grid_search": False,
            "tune_quantum": True,
            "model": ["qsvc"],
            "n_jobs": 1,
        }
        with pytest.raises(ValueError, match="grid_search"):
            model_run(X_train, X_test, y_train, y_test, "k", args)

    def test_a_quantum_model_with_nothing_to_search_is_named(self, data, sim_args):
        from qbiocode.evaluation.model_run import model_run

        X_train, X_test, y_train, y_test = data
        args = {
            **sim_args,
            "tune_quantum": True,
            "model": ["qsvc", "vqc"],
            "n_jobs": 1,
            "gridsearch_qsvc_args": {"reps": [1]},
        }
        with pytest.raises(ValueError) as excinfo:
            model_run(X_train, X_test, y_train, y_test, "k", args)
        msg = str(excinfo.value)
        assert "vqc" in msg and "qsvc" not in msg.split("have no")[0].split("[")[1]
        assert "gridsearch_<model>_args" in msg


class TestTheSearchItself:
    def test_a_failing_combination_costs_a_trial_not_the_run(self, data, sim_args):
        """Not every (encoding, entanglement, primitive) triple is constructible."""
        import qbiocode  # noqa: F401
        from qbiocode.learning._tuning import build_search_space, run_function_study
        from qbiocode.learning.compute_qsvc import compute_qsvc

        X_train, _, y_train, _ = data
        space = build_search_space("qsvc", {"encoding": ["Z", "nonsense-encoding"]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best = run_function_study(
                compute_qsvc,
                space,
                X_train,
                y_train,
                sim_args,
                model="qsvc",
                n_trials=2,
                seed=0,
            )
        assert best["encoding"] == "Z", "the one buildable option must win"

    def test_every_trial_failing_says_so(self, data, sim_args):
        """A study where nothing completed has no best configuration to report."""
        import qbiocode  # noqa: F401
        from qbiocode.learning._tuning import build_search_space, run_function_study
        from qbiocode.learning.compute_qsvc import compute_qsvc

        X_train, _, y_train, _ = data
        space = build_search_space("qsvc", {"encoding": ["nope", "also-nope"]})
        with pytest.raises(ValueError) as excinfo:
            run_function_study(
                compute_qsvc, space, X_train, y_train, sim_args, model="qsvc", n_trials=2, seed=0
            )
        msg = str(excinfo.value)
        assert "Every tuning trial" in msg
        assert "gridsearch_qsvc_args" in msg, "must point at the config block"

    def test_a_class_too_small_to_split_is_named(self, sim_args):
        """Stratifying a holdout needs at least two samples of every class."""
        from qbiocode.learning._tuning import build_search_space, run_function_study
        from qbiocode.learning.compute_qsvc import compute_qsvc

        X = np.random.default_rng(0).normal(size=(10, 3))
        y = np.array([0] * 9 + [1])
        with pytest.raises(ValueError) as excinfo:
            run_function_study(
                compute_qsvc,
                build_search_space("qsvc", {"reps": [1]}),
                X,
                y,
                sim_args,
                model="qsvc",
                n_trials=1,
            )
        assert "only 1 training sample" in str(excinfo.value)

    def test_the_search_never_sees_the_test_set(self, data, sim_args):
        """The holdout is carved out of X_train, so the reported score stays honest."""
        import qbiocode  # noqa: F401
        from qbiocode.learning._tuning import build_search_space, run_function_study

        X_train, X_test, y_train, y_test = data
        seen = []

        def spy(Xa, Xb, ya, yb, args, **params):
            seen.append((len(Xa), len(Xb)))
            import pandas as pd

            return pd.DataFrame({"results_spy": [{"accuracy": 0.5, "time": 0.0}]})

        run_function_study(
            spy,
            build_search_space("qsvc", {"reps": [1, 2]}),
            X_train,
            y_train,
            sim_args,
            model="qsvc",
            n_trials=2,
            seed=0,
            validation_split=0.25,
        )
        for n_inner, n_val in seen:
            assert n_inner + n_val == len(X_train), (
                f"the trial saw {n_inner}+{n_val} samples but X_train has {len(X_train)}; "
                f"the test set must not reach the search"
            )
            assert n_val == pytest.approx(len(X_train) * 0.25, abs=1)


class TestTheQplProjectionCache:
    """QPL caches its projected features; the key has to describe the circuit.

    The key used to be ``data_key`` alone, in a hardcoded ``qpl_projections``
    directory. Changing ``encoding``, ``entanglement``, ``reps`` or ``primitive`` and
    rerunning therefore loaded the projection computed for the *previous* settings --
    the new circuit was never run, and the reported result described the old one. Under
    tuning that is fatal rather than merely wrong: every trial after the first would
    score the same cached projection, so the search would compare a hyperparameter
    against itself and report whichever value it happened to try first.
    """

    @staticmethod
    def _projections(directory):
        return {p.name for p in directory.glob("qpl_projection_*.npy")}

    def _run(self, tmp_path, data, encoding="Z", reps=1):
        import qbiocode  # noqa: F401
        from qbiocode.learning.compute_qpl import compute_qpl

        X_train, X_test, y_train, y_test = data
        args = {
            "backend": "simulator",
            "shots": 512,
            "seed": 42,
            "grid_search": False,
            "qpl_projection_dir": str(tmp_path),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return compute_qpl(
                X_train,
                X_test,
                y_train,
                y_test,
                args,
                encoding=encoding,
                reps=reps,
                classical_models=["lr"],
            )

    def test_changing_the_feature_map_writes_a_separate_cache_entry(self, tmp_path, data):
        self._run(tmp_path, data, encoding="Z", reps=1)
        before = self._projections(tmp_path)
        assert len(before) == 2, f"baseline run wrote no train/test pair: {before}"

        self._run(tmp_path, data, encoding="Z", reps=2)
        after = self._projections(tmp_path)
        assert len(after - before) == 2, (
            f"changing reps reused the cached projection instead of writing a new pair; "
            f"new files: {after - before}"
        )

    def test_identical_settings_reuse_the_cache(self, tmp_path, data):
        self._run(tmp_path, data, encoding="Z", reps=1)
        before = self._projections(tmp_path)
        self._run(tmp_path, data, encoding="Z", reps=1)
        assert self._projections(tmp_path) == before, "a cache hit wrote new files"

    def test_a_wrong_row_count_is_reported_with_both_numbers(self, tmp_path, data):
        """The old failure surfaced as sklearn complaining about sample counts."""
        import numpy as np

        import qbiocode  # noqa: F401
        from qbiocode.learning.compute_qpl import compute_qpl

        self._run(tmp_path, data, encoding="Z", reps=1)
        train_file = next(p for p in tmp_path.glob("qpl_projection_*_train.npy"))
        np.save(train_file, np.load(train_file, allow_pickle=False)[:-3])

        X_train, X_test, y_train, y_test = data
        args = {
            "backend": "simulator",
            "shots": 512,
            "seed": 42,
            "grid_search": False,
            "qpl_projection_dir": str(tmp_path),
        }
        with pytest.raises(ValueError) as excinfo:
            compute_qpl(
                X_train,
                X_test,
                y_train,
                y_test,
                args,
                encoding="Z",
                reps=1,
                classical_models=["lr"],
            )
        msg = str(excinfo.value)
        assert str(len(X_train) - 3) in msg and str(len(X_train)) in msg
        assert "qpl_projection_dir" in msg, "must say how to get out of it"


class TestTheBugsFoundAfterTheFirstPass:
    """Regressions found by probing the finished feature."""

    @pytest.mark.parametrize("n_trials", [0, -3])
    def test_a_budget_below_one_trial_is_not_reported_as_every_trial_failing(
        self, n_trials, data, sim_args
    ):
        """The all-trials-failed branch used to catch this and blame the search space.

        ``n_trials: 0`` ran nothing, so the "every tuning trial failed ... check the
        gridsearch_qsvc_args block" error fired -- pointing at a config block that was
        perfectly fine.
        """
        from qbiocode.learning._tuning import build_search_space, run_function_study
        from qbiocode.learning.compute_qsvc import compute_qsvc

        X_train, _, y_train, _ = data
        with pytest.raises(ValueError) as excinfo:
            run_function_study(
                compute_qsvc,
                build_search_space("qsvc", {"reps": [1]}),
                X_train,
                y_train,
                sim_args,
                model="qsvc",
                n_trials=n_trials,
            )
        message = str(excinfo.value)
        assert "at least one trial" in message
        assert "failed" not in message, "must not blame the search space"

    @pytest.mark.parametrize("split", [0.0, 1.0, -0.2, 3, "0.25"])
    def test_an_invalid_validation_split_is_named_in_the_users_own_terms(
        self, split, data, sim_args
    ):
        """It used to surface as sklearn complaining about `test_size`.

        `test_size` appears nowhere in a QBioCode config, so the message named a
        parameter the user had never written.
        """
        from qbiocode.learning._tuning import build_search_space, run_function_study
        from qbiocode.learning.compute_qsvc import compute_qsvc

        X_train, _, y_train, _ = data
        with pytest.raises(ValueError) as excinfo:
            run_function_study(
                compute_qsvc,
                build_search_space("qsvc", {"reps": [1]}),
                X_train,
                y_train,
                sim_args,
                model="qsvc",
                n_trials=1,
                validation_split=split,
            )
        message = str(excinfo.value)
        assert "validation_split" in message
        assert "test_size" not in message

    def test_grid_tuner_with_quantum_tuning_says_optuna_is_used_anyway(self, data, sim_args):
        """`tuner: grid` cannot reach the quantum models, and silence would hide that.

        Their `_opt` wrappers score a whole compute function rather than an estimator
        GridSearchCV could drive, so there is no exhaustive engine for them. Accepting
        `tuner: grid` and quietly running Optuna makes the result uninterpretable later.
        """
        import qbiocode  # noqa: F401
        from qbiocode.evaluation.model_run import model_run

        X_train, X_test, y_train, y_test = data
        args = {
            **sim_args,
            "model": ["qsvc"],
            "n_jobs": 1,
            "q_seed": 42,
            "tune_quantum": True,
            "tuner": "grid",
            "n_trials_quantum": 1,
            "cross_validation": 3,
            "gridsearch_qsvc_args": {"reps": [1]},
        }
        with pytest.warns(UserWarning, match="applies to the classical models only"):
            out = model_run(X_train, X_test, y_train, y_test, "k", args)
        assert "results_qsvc_opt" in out, "the quantum model should still be tuned"
