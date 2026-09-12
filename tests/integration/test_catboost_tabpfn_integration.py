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

"""CatBoost and TabPFN through the whole pipeline, the way a user reaches them.

The unit tests in ``tests/test_catboost_tabpfn.py`` call the learners directly. That
cannot see the defects these two bring, because those live in the seams:

* **CatBoost writes ``catboost_info/`` into the current working directory.** A direct
  call does that too, but only a real QProfiler run has *several joblib workers sharing
  one CWD*, which is what turns litter into a race. The unit test proves the flag is
  set; this tier proves it survives the fan-out.

* **TabPFN must be genuinely optional.** Every unit test runs in an environment where
  the extra happens to be installed, so none of them can tell whether ``import
  qbiocode`` would survive without it. Only a subprocess with ``tabpfn`` made
  unimportable answers that -- and that is the state of every default install.

* **The shipped config must be executable as written.** The blocks were added to two
  config files; a run driven by the real CLI is what proves the keys reach the
  estimators rather than merely parsing as YAML.

Everything here goes through ``run_qprofiler`` -- a real subprocess, the real config
directory, a real CSV on disk -- for the reason the tier exists: ``_resolve_scaling``
once accepted a plain ``['True']`` in every unit test and rejected the
``ListConfig(['True'])`` the shipped config actually produces.

TabPFN's pinned ``v2`` weights need no token, but the first fit downloads a checkpoint
and the extra may be absent, so anything that fits it defers to the ``tabpfn_skip_reason``
fixture from ``tests/conftest.py``.
"""

from __future__ import annotations

import textwrap

import pandas as pd
import pytest

from .conftest import (
    KEY_COLUMNS,
    METRIC_COLUMNS,
    metric_signature,
    run_python,
    run_qprofiler,
)

# One embedding and one iteration: this tier is about whether the seams hold, not
# about coverage of the embedding matrix, which test_qprofiler_end_to_end.py owns.
BASE_OVERRIDES = [
    "embeddings=[none]",
    "iter=1",
    "n_components=2",
    "n_jobs=2",
]


@pytest.fixture(scope="session")
def catboost_run(tmp_path_factory):
    """CatBoost untuned, beside another model so the joblib fan-out is real."""
    return run_qprofiler(
        tmp_path_factory.mktemp("catboost-untuned"),
        BASE_OVERRIDES + ["model=[catboost,rf]", "seed=7"],
    )


@pytest.fixture(scope="session")
def catboost_run_dir(tmp_path_factory):
    """A run kept with its working directory, for inspecting what it left behind."""
    work_dir = tmp_path_factory.mktemp("catboost-litter")
    run_qprofiler(work_dir, BASE_OVERRIDES + ["model=[catboost,rf]", "seed=7"])
    return work_dir


@pytest.fixture(scope="session")
def catboost_tuned_run(tmp_path_factory):
    """CatBoost through the Optuna tuner, driven by the shipped gridsearch block."""
    return run_qprofiler(
        tmp_path_factory.mktemp("catboost-tuned"),
        BASE_OVERRIDES
        + ["model=[catboost]", "seed=7", "grid_search=True", "n_trials=4", "cross_validation=3"],
    )


class TestCatBoostThroughQProfiler:
    """The untuned path: dispatch, parallelism, results file, config block."""

    def test_it_appears_in_the_results_file(self, catboost_run):
        assert set(catboost_run["model"]) == {"catboost", "rf"}, (
            f"expected both models in ModelResults.csv, got {sorted(set(catboost_run['model']))}"
        )

    def test_it_carries_the_documented_columns(self, catboost_run):
        expected = set(KEY_COLUMNS) | set(METRIC_COLUMNS) | {"time", "Model_Parameters"}
        missing = expected - set(catboost_run.columns)
        assert not missing, f"ModelResults.csv is missing {sorted(missing)}"

    @pytest.mark.parametrize("column", METRIC_COLUMNS)
    def test_its_metrics_are_finite_and_in_range(self, catboost_run, column):
        rows = catboost_run[catboost_run["model"] == "catboost"]
        values = pd.to_numeric(rows[column])
        assert values.notna().all(), f"catboost produced nan in {column}"
        assert ((values >= 0.0) & (values <= 1.0)).all(), f"{column} outside [0, 1]"

    def test_it_learned_something(self, catboost_run):
        """The fixture dataset is separable; a boosted model at chance means a broken wiring."""
        rows = catboost_run[catboost_run["model"] == "catboost"]
        assert pd.to_numeric(rows["accuracy"]).max() > 0.6

    def test_the_shipped_config_block_reached_the_estimator(self, catboost_run):
        """``catboost_args`` must be applied, not merely parsed.

        ``model_run._model_args`` falls back to ``{}`` and logs when a block is absent,
        so a mis-named block is silent: the model still runs, at CatBoost's defaults.
        The shipped block sets ``iterations: 200``, which is not CatBoost's own 1000.
        """
        rows = catboost_run[catboost_run["model"] == "catboost"]
        parameters = rows["Model_Parameters"].iloc[0]
        assert "'iterations': 200" in parameters, (
            f"catboost_args did not reach the estimator; parameters were {parameters}"
        )

    def test_it_is_reproducible_at_one_seed(self, catboost_run, tmp_path_factory):
        """A second independent run at the same seed must agree row for row.

        CatBoost samples rows when a bootstrap is active, so an unseeded estimator
        drifts between runs -- the same defect that moved the decision-tree rows.
        """
        again = run_qprofiler(
            tmp_path_factory.mktemp("catboost-untuned-again"),
            BASE_OVERRIDES + ["model=[catboost,rf]", "seed=7"],
        )
        pd.testing.assert_frame_equal(
            metric_signature(catboost_run), metric_signature(again)
        )


class TestCatBoostLeavesNoLitter:
    """The seam the unit tests cannot reach: several workers, one working directory."""

    def test_no_catboost_info_directory_survives_a_full_run(self, catboost_run_dir):
        """Default CatBoost writes training logs into the CWD on every single fit.

        Under QProfiler that CWD is shared by every joblib worker, so the directory is
        both a race and litter left in the user's project. ``results/`` is the run's own
        output and is expected.
        """
        leftovers = sorted(
            p.name for p in catboost_run_dir.iterdir() if p.name not in {"data", "results"}
        )
        assert "catboost_info" not in leftovers, (
            "a QProfiler run left catboost_info/ behind; allow_writing_files=False has "
            f"regressed somewhere in the dispatch path. Working directory held: {leftovers}"
        )

    def test_nor_anywhere_beneath_it(self, catboost_run_dir):
        """A worker with a different CWD would put the directory somewhere else."""
        stray = [
            str(p.relative_to(catboost_run_dir))
            for p in catboost_run_dir.rglob("catboost_info")
        ]
        assert not stray, f"catboost_info written at {stray}"


class TestCatBoostTuning:
    """The tuned path, including the exhaustive engine that has to stay reachable."""

    def test_the_tuned_run_reports_tuned_parameters(self, catboost_tuned_run):
        assert "BestParams_Tuned" in catboost_tuned_run.columns, (
            f"tuned run wrote {sorted(catboost_tuned_run.columns)}"
        )
        best = catboost_tuned_run["BestParams_Tuned"].iloc[0]
        assert "iterations" in best and "learning_rate" in best, (
            f"the shipped gridsearch_catboost_args was not searched: {best}"
        )

    def test_the_tuned_model_is_labelled_as_tuned(self, catboost_tuned_run):
        assert set(catboost_tuned_run["model"]) == {"catboost_opt"}

    def test_its_metrics_are_finite(self, catboost_tuned_run):
        for column in METRIC_COLUMNS:
            values = pd.to_numeric(catboost_tuned_run[column])
            assert values.notna().all(), f"tuned catboost produced nan in {column}"

    def test_a_continuous_range_from_the_config_was_actually_sampled(self, catboost_tuned_run):
        """``learning_rate`` ships as a ``{low, high, log}`` range, not a list.

        A value landing exactly on a decade would suggest the range was collapsed to a
        categorical somewhere; anything strictly inside the interval proves Optuna
        sampled it.
        """
        best = catboost_tuned_run["BestParams_Tuned"].iloc[0]
        assert "'learning_rate': 0." in best, f"no sampled learning_rate in {best}"

    def test_the_exhaustive_grid_engine_still_runs_it(self, tmp_path_factory):
        """``tuner: grid`` exists so a published number stays reproducible.

        The shipped block writes ``learning_rate`` and ``l2_leaf_reg`` as ranges, which a
        grid cannot enumerate, so both are removed here with Hydra's ``~`` deletion --
        the two remaining list entries are then narrowed so the sweep stays small.

        Two Hydra details make this the only spelling that works. A whole-dict override
        (``gridsearch_catboost_args={iterations: [10]}``) *merges* into the existing node,
        leaving the ranges in place; and replacing a range in place
        (``gridsearch_catboost_args.learning_rate=[0.1]``) is refused outright, because
        the node is a dict and the new value is a list.
        """
        frame = run_qprofiler(
            tmp_path_factory.mktemp("catboost-grid"),
            BASE_OVERRIDES
            + [
                "model=[catboost]",
                "seed=7",
                "grid_search=True",
                "tuner=grid",
                "cross_validation=2",
                "~gridsearch_catboost_args.learning_rate",
                "~gridsearch_catboost_args.l2_leaf_reg",
                "gridsearch_catboost_args.iterations=[10,20]",
                "gridsearch_catboost_args.depth=[3]",
                "gridsearch_catboost_args.random_strength=[1.0]",
            ],
        )
        assert set(frame["model"]) == {"catboost_opt"}
        best = frame["BestParams_Tuned"].iloc[0]
        assert "iterations" in best and "depth" in best

    def test_the_shipped_ranges_are_refused_by_the_grid_with_an_actionable_message(
        self, tmp_path
    ):
        """A range under ``tuner: grid`` must name the parameter, the file and the fix.

        This is a real configuration a user will hit -- the shipped block is written for
        Optuna, and flipping ``tuner`` alone cannot work. Silently treating the range
        mapping as a single categorical *value* is what used to happen, and it surfaced
        as an InvalidParameterError about the estimator instead.
        """
        from .conftest import CONFIG_DIR, subprocess_env, write_dataset
        import subprocess
        import sys

        write_dataset(tmp_path / "data")
        argv = [
            "qprofiler",
            f"--config-dir={CONFIG_DIR}",
            "--config-name=config",
            "folder_path=data",
            "file_dataset=ALL",
            *BASE_OVERRIDES,
            "model=[catboost]",
            "grid_search=True",
            "tuner=grid",
            "cross_validation=2",
        ]
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys\nsys.argv = {argv!r}\n"
                "from qbiocode.apps.qprofiler.qprofiler import main\nmain()\n",
            ],
            cwd=str(tmp_path),
            env=subprocess_env(),
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert completed.returncode != 0, "a range under tuner: grid should not be accepted"
        combined = completed.stdout + completed.stderr
        assert "learning_rate" in combined, "the error must name the offending parameter"
        assert "tuner: optuna" in combined, "the error must name the fix"


class TestTabPFNIsGenuinelyOptional:
    """What a default ``pip install qbiocode`` gets, simulated by hiding the module.

    Nothing else in the suite can check this: every other test runs in an environment
    that happens to have the extra, so an accidental hard dependency on ``tabpfn``
    would pass everywhere and break on every user's machine.
    """

    #: Makes `tabpfn` unimportable in the child, including to `importlib.util.find_spec`,
    #: which is what `tabpfn_is_available` consults.
    BLOCKER = textwrap.dedent(
        """
        import sys

        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "tabpfn" or name.startswith("tabpfn."):
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return None

        sys.meta_path.insert(0, _Blocker())
        """
    )

    def test_the_package_imports_without_it(self, tmp_path):
        completed = run_python(
            self.BLOCKER
            + textwrap.dedent(
                """
                import qbiocode
                from qbiocode.learning.compute_tabpfn import tabpfn_is_available
                assert tabpfn_is_available() is False, "the blocker did not take effect"
                assert callable(qbiocode.compute_tabpfn)
                assert callable(qbiocode.compute_catboost)
                print("IMPORT-OK")
                """
            ),
            cwd=tmp_path,
        )
        assert completed.returncode == 0, (
            "import qbiocode fails when tabpfn is absent, so the [tabpfn] extra is not "
            f"optional after all.\n--- stderr ---\n{completed.stderr[-3000:]}"
        )
        assert "IMPORT-OK" in completed.stdout

    def test_selecting_it_without_the_extra_says_how_to_install_it(self, tmp_path):
        completed = run_python(
            self.BLOCKER
            + textwrap.dedent(
                """
                import numpy as np
                from qbiocode import compute_tabpfn

                rng = np.random.default_rng(0)
                X = rng.normal(size=(40, 3))
                y = (X[:, 0] > 0).astype(int)
                try:
                    compute_tabpfn(X[:28], X[28:], y[:28], y[28:], {"grid_search": False})
                except ImportError as error:
                    print("MESSAGE:", " ".join(str(error).split()))
                else:
                    print("NO-ERROR")
                """
            ),
            cwd=tmp_path,
        )
        assert completed.returncode == 0, completed.stderr[-3000:]
        assert "MESSAGE:" in completed.stdout, (
            f"expected an ImportError, got: {completed.stdout[-1000:]}"
        )
        message = completed.stdout
        assert 'pip install "qbiocode[tabpfn]"' in message, (
            f"the error does not name the extra: {message[-600:]}"
        )

    def test_other_models_still_run_without_it(self, tmp_path):
        """The failure mode worth guarding: one missing extra disabling everything."""
        completed = run_python(
            self.BLOCKER
            + textwrap.dedent(
                """
                import warnings
                warnings.filterwarnings("ignore")
                import numpy as np
                from qbiocode.evaluation.model_run import model_run

                rng = np.random.default_rng(0)
                X = rng.normal(size=(60, 4))
                y = (X[:, 0] > 0).astype(int)
                result = model_run(
                    X[:40], X[40:], y[:40], y[40:], "k",
                    {"model": ["catboost", "rf"], "grid_search": False,
                     "seed": 42, "n_jobs": 2, "catboost_args": {"iterations": 20}},
                )
                assert "results_catboost" in result, sorted(result)
                print("RUN-OK", result["results_catboost"][0]["accuracy"])
                """
            ),
            cwd=tmp_path,
        )
        assert completed.returncode == 0, (
            "a missing [tabpfn] extra broke an unrelated model run.\n"
            f"--- stderr ---\n{completed.stderr[-3000:]}"
        )
        assert "RUN-OK" in completed.stdout
        # And nothing wrote litter into the child's working directory either.
        assert not list(tmp_path.rglob("catboost_info"))


class TestTabPFNThroughQProfiler:
    """The real thing, when the license has been accepted and the weights are cached."""

    def test_it_appears_in_the_results_file(self, tmp_path_factory, tabpfn_skip_reason):
        if tabpfn_skip_reason is not None:
            pytest.skip(tabpfn_skip_reason)
        frame = run_qprofiler(
            tmp_path_factory.mktemp("tabpfn-untuned"),
            BASE_OVERRIDES + ["model=[tabpfn]", "seed=7"],
        )
        assert set(frame["model"]) == {"tabpfn"}
        values = pd.to_numeric(frame["accuracy"])
        assert values.notna().all()
        assert ((values >= 0.0) & (values <= 1.0)).all()

    def test_the_tuned_path_runs(self, tmp_path_factory, tabpfn_skip_reason):
        if tabpfn_skip_reason is not None:
            pytest.skip(tabpfn_skip_reason)
        frame = run_qprofiler(
            tmp_path_factory.mktemp("tabpfn-tuned"),
            BASE_OVERRIDES
            + [
                "model=[tabpfn]",
                "seed=7",
                "grid_search=True",
                "n_trials=2",
                "cross_validation=2",
                "gridsearch_tabpfn_args={n_estimators: [1, 2], device: cpu}",
            ],
        )
        assert set(frame["model"]) == {"tabpfn_opt"}
        assert "n_estimators" in frame["BestParams_Tuned"].iloc[0]


class TestTheQplHeadsEndToEnd:
    """A quantum projection with the new heads fitted on it."""

    def test_catboost_runs_as_a_qpl_head(self, tmp_path):
        """The projection is real, so a broken head wastes work that already cost time.

        Kept to two heads and one repetition: this is about whether ``qpl_catboost``
        appears with a finite score, which ``compute_qpl``'s per-head ``pd.concat`` and
        the ``best_params`` fallback both have to be right for.
        """
        completed = run_python(
            textwrap.dedent(
                """
                import warnings
                warnings.filterwarnings("ignore")
                import numpy as np
                from qbiocode.learning.compute_qpl import compute_qpl

                rng = np.random.default_rng(0)
                X = rng.normal(size=(40, 3))
                y = (X[:, 0] > 0).astype(int)
                frame = compute_qpl(
                    X[:28], X[28:], y[:28], y[28:],
                    {"grid_search": False, "seed": 42, "backend": "simulator",
                     "shots": 256},
                    classical_models=["rf", "catboost"], reps=1,
                )
                columns = sorted(c for c in frame.columns if c.startswith("results_"))
                print("COLUMNS:", columns)
                for column in columns:
                    metrics = [v for v in frame[column] if isinstance(v, dict)][0]
                    assert np.isfinite(metrics["accuracy"]), column
                    assert metrics["Model_Parameters"]["best_params"], column
                print("QPL-OK")
                """
            ),
            cwd=tmp_path,
            timeout=1800,
        )
        assert completed.returncode == 0, (
            f"QPL with a catboost head failed\n--- stderr ---\n{completed.stderr[-3000:]}"
        )
        assert "results_qpl_catboost" in completed.stdout, (
            f"the catboost head produced no results column: {completed.stdout[-1000:]}"
        )
        assert "QPL-OK" in completed.stdout
        assert not list(tmp_path.rglob("catboost_info")), (
            "the QPL catboost head littered the working directory"
        )

    def test_an_unavailable_tabpfn_head_warns_and_is_dropped(self, tmp_path):
        """The projection must still be used by the remaining heads.

        Reached with the same blocker as above, because in this environment the extra is
        installed -- so without hiding it, the warn-and-drop branch is never executed.
        """
        completed = run_python(
            TestTabPFNIsGenuinelyOptional.BLOCKER
            + textwrap.dedent(
                """
                import warnings
                import numpy as np
                from qbiocode.learning.compute_qpl import compute_qpl

                rng = np.random.default_rng(0)
                X = rng.normal(size=(40, 3))
                y = (X[:, 0] > 0).astype(int)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    frame = compute_qpl(
                        X[:28], X[28:], y[:28], y[28:],
                        {"grid_search": False, "seed": 42, "backend": "simulator",
                         "shots": 256},
                        classical_models=["rf", "tabpfn"], reps=1,
                    )
                messages = [str(w.message) for w in caught]
                assert any("TabPFN is not installed" in m for m in messages), messages
                columns = sorted(c for c in frame.columns if c.startswith("results_"))
                assert columns == ["results_qpl_rf"], columns
                print("DROPPED-OK")
                """
            ),
            cwd=tmp_path,
            timeout=1800,
        )
        assert completed.returncode == 0, (
            "an unavailable tabpfn head was fatal rather than dropped\n"
            f"--- stderr ---\n{completed.stderr[-3000:]}"
        )
        assert "DROPPED-OK" in completed.stdout


class TestTheSageCatBoostSurrogate:
    """The QuantumSage CLI has to accept the new sage type it advertises."""

    def test_the_cli_accepts_catboost_optuna(self, tmp_path):
        """``--model-type catboost`` must map to a real sage type, not ``KeyError``.

        The alias map and the ``choices`` list are maintained separately, so a value
        argparse accepts can still be missing from the map.
        """
        completed = run_python(
            textwrap.dedent(
                """
                import sys
                from qbiocode.apps.sage.sage import main
                sys.argv = ["qsage", "--help"]
                try:
                    main()
                except SystemExit:
                    pass
                """
            ),
            cwd=tmp_path,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert "catboost_optuna" in completed.stdout, (
            "qsage --help does not advertise catboost_optuna"
        )

    def test_the_alias_map_covers_every_advertised_choice(self, tmp_path):
        """Every value in ``choices`` must resolve through ``sage_type_map``."""
        completed = run_python(
            textwrap.dedent(
                """
                import inspect, re
                from qbiocode.apps.sage import sage as module

                source = inspect.getsource(module)
                choices = re.search(
                    r"choices=\\[([^\\]]*)\\],\\s*\\n\\s*help='Type of sub-sage", source
                )
                assert choices, "could not find the --model-type choices list"
                advertised = re.findall(r"'([a-z_]+)'", choices.group(1))
                mapped = re.findall(r"'([a-z_]+)': '([a-z_]+)'", source)
                mapping = dict(mapped)
                missing = [c for c in advertised if c not in mapping]
                assert not missing, f"choices not in sage_type_map: {missing}"
                print("ALIASES-OK", sorted(advertised))
                """
            ),
            cwd=tmp_path,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert "ALIASES-OK" in completed.stdout
        assert "catboost" in completed.stdout

class TestTheBugsFoundInReview:
    """The review defects, through the paths a user actually reaches them by.

    Each of these passed the first version of this suite. Two of them can only be seen
    from here: one is about what a *config file* does to a real run, and the other about
    what survives inside ``compute_qpl`` after an expensive projection.
    """

    def test_a_contradictory_catboost_args_block_fails_with_an_actionable_message(
        self, tmp_path
    ):
        """``catboost_args`` naming both bootstrap parameters raised a bare CatBoostError.

        The message a user saw mentioned neither the config file nor either parameter they
        had written -- only CatBoost's internal ``bootstrap_options.cpp``. Driven through
        the real CLI because that is the only way to prove the message survives Hydra and
        the joblib worker boundary rather than being swallowed or reformatted.
        """
        from .conftest import CONFIG_DIR, subprocess_env, write_dataset
        import subprocess
        import sys

        write_dataset(tmp_path / "data")
        argv = [
            "qprofiler",
            f"--config-dir={CONFIG_DIR}",
            "--config-name=config",
            "folder_path=data",
            "file_dataset=ALL",
            *BASE_OVERRIDES,
            "model=[catboost]",
            "+catboost_args.subsample=0.8",
            "+catboost_args.bagging_temperature=0.5",
        ]
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys\nsys.argv = {argv!r}\n"
                "from qbiocode.apps.qprofiler.qprofiler import main\nmain()\n",
            ],
            cwd=str(tmp_path),
            env=subprocess_env(),
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert completed.returncode != 0, "a contradictory block should not be accepted"
        combined = completed.stdout + completed.stderr
        assert "catboost_args" in combined, "the error must name the config block"
        assert "subsample" in combined and "bagging_temperature" in combined, (
            "the error must name both parameters the user wrote"
        )
        assert "bootstrap_options.cpp" not in combined, (
            "CatBoost's internal error is what surfaced before; it should be translated"
        )

    def test_a_gated_tabpfn_qpl_head_does_not_take_the_projection_down(self, tmp_path):
        """The other heads, and the quantum projection they were fitted on, must survive.

        Unreachable from the unit tier: the extra is installed here and imports fine, so
        only an actual fit discovers the checkpoint is gated. If the weights *are*
        available this asserts the head simply ran, which is the same contract from the
        other side.
        """
        completed = run_python(
            textwrap.dedent(
                """
                import warnings
                import numpy as np
                from qbiocode.learning.compute_qpl import compute_qpl

                rng = np.random.default_rng(0)
                X = rng.normal(size=(40, 3))
                y = (X[:, 0] > 0).astype(int)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    frame = compute_qpl(
                        X[:28], X[28:], y[:28], y[28:],
                        {"grid_search": False, "seed": 42, "backend": "simulator",
                         "shots": 256},
                        classical_models=["rf", "tabpfn"], reps=1,
                    )
                columns = sorted(c for c in frame.columns if c.startswith("results_"))
                skipped = [m for m in (str(w.message) for w in caught)
                           if "qpl_tabpfn" in m and "skipped" in m]
                print("COLUMNS:", columns)
                print("SKIPPED:", bool(skipped))
                for column in columns:
                    metrics = [v for v in frame[column] if isinstance(v, dict)][0]
                    assert np.isfinite(metrics["accuracy"]), column
                print("SURVIVED-OK")
                """
            ),
            cwd=tmp_path,
            timeout=1800,
        )
        assert completed.returncode == 0, (
            "a gated TabPFN head took the whole QPL run down with it\n"
            f"--- stderr ---\n{completed.stderr[-3000:]}"
        )
        assert "SURVIVED-OK" in completed.stdout
        assert "results_qpl_rf" in completed.stdout, (
            f"the rf head's results were lost: {completed.stdout[-600:]}"
        )
        # Whichever way this environment is configured, the outcome must be coherent:
        # either tabpfn was skipped with a warning, or it ran and produced a column.
        skipped = "SKIPPED: True" in completed.stdout
        ran = "results_qpl_tabpfn" in completed.stdout
        assert skipped != ran, (
            f"tabpfn neither ran nor was cleanly skipped: {completed.stdout[-600:]}"
        )

    def test_a_multi_valued_inert_hyperparameter_is_refused_by_a_real_run(
        self, tmp_path
    ):
        """The refusal has to survive Hydra and the joblib worker boundary.

        ``min_data_in_leaf`` used to be searchable and silently inert at the default grow
        policy. It is now refused when given several values -- and the message must reach the
        user rather than being swallowed by a worker.
        """
        from .conftest import CONFIG_DIR, subprocess_env, write_dataset
        import subprocess
        import sys

        write_dataset(tmp_path / "data")
        argv = [
            "qprofiler",
            f"--config-dir={CONFIG_DIR}",
            "--config-name=config",
            "folder_path=data",
            "file_dataset=ALL",
            *BASE_OVERRIDES,
            "model=[catboost]",
            "grid_search=True",
            "n_trials=2",
            "cross_validation=2",
            "+gridsearch_catboost_args.min_data_in_leaf=[1,40]",
        ]
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys\nsys.argv = {argv!r}\n"
                "from qbiocode.apps.qprofiler.qprofiler import main\nmain()\n",
            ],
            cwd=str(tmp_path),
            env=subprocess_env(),
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert completed.returncode != 0, "several values should be refused, not searched"
        combined = completed.stdout + completed.stderr
        assert "min_data_in_leaf" in combined
        assert "grow_policy" in combined, "must say what would make it take effect"
