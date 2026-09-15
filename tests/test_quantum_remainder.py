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

"""The quantum seams no other test file reaches: the winner finder, QPL's cache, the CLI.

``tests/test_quantum_models.py`` fits all five quantum models and pins the row they
produce. What it cannot see is everything on either side of that row, and five
regressions lived there: one is fixed and asserted below, and the other four are the
whole of what this file still pins -- one starred paragraph each, and between them the
eight ``xfail(strict=True)`` cases here (one, one, four and two respectively). Three of
the five are silent, producing a file or a number that looks exactly right; the other two
raise, but from somewhere other than the mistake, so the message names neither the
parameter that was wrong nor the key that was missing.

*The results table never reached the winner finder -- fixed; the pins are now
assertions.* ``qml_winner`` selected quantum rows by matching the ``model`` column
against ``["QSVC", "QNN", "VQC", "PQK"]`` with ``.isin()``, upper case and exact, while
every writer of that column spells it lower case -- ``model_run`` passes ``model=method``
straight from ``args['model']``, so a real ModelResults.csv carries ``qsvc``. The whole
branch was therefore dead on genuine output: it did not crash, it printed "QML methods
were outperformed by CML methods in all datasets" and returned ``None``, so
qml_winners.csv came back empty from every real quantum run.
``tests/integration/test_pymfe_model_integration.py`` reached the branch by constructing
``QSVC`` by hand and said in a docstring that this was a defect; nothing asserted it. That
workaround is retired -- it builds its table with ``qsvc`` now -- so no test outside this
file spells a quantum model the way the old list did. QPL was worse off still -- absent
from the list at any casing, and its rows are labelled ``qpl_<head>``, so even adding
``"QPL"`` would not have matched them.
``qc_winner_finder.py:98-101`` now matches the lower-cased first ``_``-token of the label
against ``{qsvc, qnn, vqc, pqk, qpl}``, which covers the lower case a run writes, the
upper case a hand-built table uses, QPL's ``qpl_<head>`` fan-out and the ``<name>_opt``
label a tuned run carries. The tests below assert all of that on the real table -- and,
because a case-insensitive prefix rule is exactly the kind of match that over-matches,
that the classical labels are still not quantum, ``svc`` against ``qsvc`` included.

*QPL's projection cache is keyed on a parameter it does not use -- still pinned, one
case.* ``primitive`` enters the cache fingerprint but never reaches the backend, which is
requested as ``"estimator"`` unconditionally; ``compute_pqk`` rejects any other value for
exactly this reason. So a run at ``primitive='sampler'`` recomputes and stores a second
projection pair whose bytes equal the first, doubling a cache that exists to be
expensive, while reporting a measurement that never happened.

*And a cached projection is validated on its row count but not its width -- still pinned,
one case.* The fingerprint covers the inputs, not the shape of the output, so a pair left
behind by a change to the observable set has a name the current run accepts. It is loaded,
reshaped to ``(n, -1)`` and fitted, and every downstream shape check agrees with itself:
the result is a complete row of metrics computed from the wrong number of observables.
``compute_pqk`` checks both the count and the width.

*``compute_qpl`` validates none of its arguments -- still pinned, four cases, the largest
of the pins.* It starts using them immediately, so each mistake is reported by whatever
library reaches it first: a test matrix of a different width by qiskit's bindings array, a
1-D training matrix as ``IndexError: tuple index out of range``, a non-mapping ``args`` as
``AttributeError: 'list' object has no attribute 'get'``, a non-string ``data_key`` by the
string concatenation that builds the cache filename. None of the four messages names the
parameter that was wrong, and none of them says ``compute_qpl``. Its twin
``compute_pqk`` refuses all four by name before it creates a directory or reads a cache.

*A config that lost its seed dies with ``KeyError: 'seed'`` -- still pinned, two cases,
one per projection model.* Both read ``args['seed']`` bare, after the projection is
loaded, where the message says neither which key nor which file. On a warm cache it is
instant, so nothing about it looks like a run at all. ``model_run`` treats the same class
of mistake as a contract violation worth a sentence (``Unknown model ...`` names every
valid choice); these do not.

*No quantum model had ever been through the real CLI.*
``tests/integration/test_qprofiler_end_to_end.py`` is ``lr`` and ``dt`` only, so the
seam from ``model_run``'s dict to a ModelResults.csv row was untested for a learner
whose results label is produced inside a compute function -- and that CSV is the input
the winner finder above is handed. One simulator-backed QSVC run through
``qprofiler.main()`` closes it, and its output is what the winner-finder tests read, so
the label matching is exercised on a real table rather than a fabricated one.

Also here: ``vqc`` and ``qnn`` reproducibility, asserted from both sides. ``model_run``
seeds both ``algorithm_globals`` singletons (qiskit-algorithms' and
qiskit-machine-learning's are separate objects, and the variational models read the
second), which made these two models reproducible for the first time.
``tests/test_quantum_models.py`` guards the same-seed half of that directly -- its
same-seed check is parametrized over ``pqk``, ``qsvc``, ``vqc`` and ``qnn``, so the two
variational models are no longer excluded from it. What it has no counterpart for is the
other half: that varying ``q_seed`` alone *changes* the answer. Without that, "two runs
agree" is satisfied just as well by a ``q_seed`` that never reaches the optimizer, since
``seed`` alone already seeds numpy. Both halves live here, over one set of memoized runs.

Everything except the CLI run is on the statevector simulator, at 14 rows (the
projection models) or 24 (the variational pair) and two features throughout.
``requires_quantum`` is for tests needing a real device; nothing here does.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The subprocess protocol for a real QProfiler run -- PYTHONPATH pinned to this
# checkout, the shipped config, a dataset written to disk -- is defined once, in the
# integration tier's conftest. Reimplementing it here would leave two definitions of
# "the actual CLI path" to keep in step, and the point of the CLI test below is that it
# is the same path the other end-to-end tests take.
from integration.conftest import run_qprofiler

# Imported at module scope, deliberately: `qbiocode` orders the OpenMP runtimes (see
# tests/test_openmp_import_order.py), and tests/test_suite_hygiene.py forbids reaching
# first-party or base-requirement modules through pytest.importorskip.
import qbiocode  # noqa: F401
import qbiocode.utils.qutils as qutils
from qbiocode import scale_train_test
from qbiocode.evaluation.model_run import model_run
from qbiocode.learning.compute_pqk import compute_pqk
from qbiocode.learning.compute_qpl import compute_qpl
from qbiocode.utils.qc_winner_finder import qml_winner

# QSVC's kernel warnings, the QPL head chatter and sklearn's undefined-metric warnings
# on a 4-row test set are noise here, not the subject.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

#: 10 training rows so the 5-fold stratified search inside a QPL head has five samples
#: per class, and 4 test rows so ``modeleval`` sees both classes. Two features keeps
#: every circuit at two qubits.
N_TRAIN, N_TEST, N_FEATURES = 10, 4, 2

#: One QPL head, not the default six. QPL fits one classical model per name on the same
#: projection and the projection is the subject here; ``svc``'s randomized search is the
#: cheapest of them.
HEAD = "svc"

#: The QPL results label that head produces. QPL is the one dispatch key whose label is
#: not the key -- ``compute_qpl`` fans out one row per classical head as
#: ``f"{model}_{head}"`` -- which is why the winner finder has to match on the label's
#: first token rather than on the whole name.
QPL_LABEL = f"qpl_{HEAD}"

#: The same two labels a *tuned* run writes. ``model_run`` passes ``model='<name>_opt'``
#: to the tuned twin of a quantum model, so the marker is a suffix for qsvc -- and an
#: infix for QPL, whose head is appended after it. Neither shape can be enumerated in a
#: list of exact names, and tuning a quantum model is reachable now that ``tune_quantum``
#: exists.
TUNED_QUANTUM_LABELS = ("qsvc_opt", f"qpl_opt_{HEAD}")

#: Classical labels, for the over-match side of the same rule. ``svc`` is the one that
#: matters: it is a classical dispatch key *and* a substring of ``qsvc``, so a matcher
#: written as a containment test in either direction reports every SVC run as a quantum
#: winner. ``rf`` is a QPL head, which makes ``qpl_rf`` quantum while ``rf`` alone is not,
#: and ``dt_opt`` checks that honouring the ``_opt`` suffix did not make tuned classical
#: rows quantum too.
CLASSICAL_LABELS = ("dt", "rf", "catboost", "svc", "dt_opt")


def _projection_dataset():
    """Features in [0, 1) and perfectly balanced labels.

    Already in the interval the feature maps encode as rotation angles, so no scaling
    step is needed; the labels are alternated rather than derived from the features
    because nothing here asserts on accuracy, and a balanced split is what the stratified
    search inside every head requires.
    """
    rng = np.random.default_rng(0)
    total = N_TRAIN + N_TEST
    X = rng.random((total, N_FEATURES))
    y = np.array([0, 1] * (total // 2))
    return X[:N_TRAIN], X[N_TRAIN:], y[:N_TRAIN], y[N_TRAIN:]


def _qpl_args(projection_dir, **extra):
    """The keys ``compute_qpl`` reads, with the projection cache under a temp directory.

    ``compute_qpl`` defaults its cache to a CWD-relative ``qpl_projections``, so a test
    that accepted the default would write into the repository root and could later be
    *served* by a file some unrelated run left there.
    """
    return {
        "backend": "simulator",  # StatevectorEstimator: offline, no credentials
        "seed": 7,
        "grid_search": False,
        "qpl_projection_dir": str(projection_dir),
        **extra,
    }


def _run_qpl(projection_dir, data_key="ds", **overrides):
    """One real ``compute_qpl`` run: quantum projection, then one classical head."""
    X_train, X_test, y_train, y_test = _projection_dataset()
    return compute_qpl(
        X_train,
        X_test,
        y_train,
        y_test,
        _qpl_args(projection_dir),
        data_key=data_key,
        classical_models=[HEAD],
        **overrides,
    )


def _projections(directory):
    """The projection file names in a directory."""
    return {path.name for path in Path(directory).glob("qpl_projection_*.npy")}


def _digests(directory):
    """file name -> content digest, for every projection in a directory."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(directory).glob("qpl_projection_*.npy"))
    }


# ======================================================================================
# A quantum model through the real CLI
# ======================================================================================

#: One dataset, one embedding, one iteration, one quantum model. ~8s in total, most of
#: it the pyMFE evaluation the CLI always runs; the QSVC fit itself is ~2s.
CLI_OVERRIDES = [
    "embeddings=[pca]",
    "model=[qsvc]",
    "iter=1",
    "n_components=2",
    "n_jobs=1",
    "seed=7",
    "q_seed=7",
    "shots=64",
]

#: What ``qprofiler`` must write for a quantum row. The metric names are the ones
#: ``qc_winner_finder`` and ``QuantumSage`` select on; ``Model_Parameters`` is the
#: untuned branch's parameter column, and its absence would mean ``modeleval`` took the
#: tuned branch for a run that searched nothing.
CLI_COLUMNS = (
    "Dataset",
    "embeddings",
    "iteration",
    "model",
    "accuracy",
    "f1_score",
    "auc",
    "time",
    "Model_Parameters",
)

#: What the shipped config's ``qsvc_args`` block asks for, spelled the way
#: ``compute_qsvc`` records it (compute_qsvc.py:110-115). ``encoding: ZZ`` selects a
#: ``ZZFeatureMap`` (qutils.py:356-369) and ``pegasos: False`` on the simulator branch a
#: ``FidelityQuantumKernel`` over ``ComputeUncompute``; ``C: 0.01`` is the config's own
#: value and *not* ``compute_qsvc``'s default of 1, so it reaches the row only if the
#: ``qsvc_args`` block reached the estimator. Hardcoded rather than recomputed through
#: ``qutils.get_feature_map``: an expectation derived from the same call the row is
#: derived from would agree with the row even when both describe the wrong circuit.
CLI_RECORDED_PARAMETERS = {
    "feature_map": "ZZFeatureMap",
    "quantum_kernel": "FidelityQuantumKernel",
    "C": 0.01,
    "gamma": "scale",
}

#: The rest of the config's circuit, under the names a recorder would use for it. Nothing
#: writes these today, so they are checked for *disagreement* only: a version that starts
#: recording them cannot satisfy the test with a value the run did not use, and a version
#: that keeps leaving them out stays green.
CLI_CIRCUIT_SHAPE = {"reps": 2, "entanglement": "linear"}


@pytest.fixture(scope="module")
def cli_quantum_run(tmp_path_factory):
    """A real ``qprofiler.main()`` run whose only model is a quantum one.

    Module-scoped: it is the most expensive thing in this file, and the winner-finder
    tests below read the same output. Returns ``(ModelResults, RawDataEvaluation)``,
    both as ``qml_winner`` receives them -- read back from the CSVs rather than taken
    in memory, because ``Model_Parameters`` holds a dict before it is written and
    grouping on a dict raises ``TypeError: unhashable type``.
    """
    work_dir = tmp_path_factory.mktemp("qsvc-cli")
    results = run_qprofiler(work_dir, CLI_OVERRIDES)
    raw_files = sorted(work_dir.glob("results/**/RawDataEvaluation.csv"))
    assert len(raw_files) == 1, f"expected one RawDataEvaluation.csv, found {raw_files}"
    return results, pd.read_csv(raw_files[0])


class TestAQuantumRowThroughTheRealCli:
    """The ``model_run`` dict -> ModelResults.csv seam, for a quantum learner."""

    def test_the_writer_produced_one_row_carrying_the_documented_columns(self, cli_quantum_run):
        """One requested model, one embedding, one iteration: exactly one row.

        A quantum model that started answering to a second results label -- which QPL
        already does -- would multiply this, and the row count is the only place a
        reader would see it.
        """
        results, _ = cli_quantum_run
        assert len(results) == 1, f"expected one results row, got {len(results)}"
        missing = [column for column in CLI_COLUMNS if column not in results.columns]
        assert not missing, f"ModelResults.csv is missing {missing}"

    def test_the_recorded_model_name_is_the_dispatch_key_in_lower_case(self, cli_quantum_run):
        """The label the winner finder has to match, established on real output.

        ``model_run`` passes ``model=method`` from ``args['model']``, so the column
        carries the config's own spelling -- and Hydra's config, the docs and
        ``compute_ml_dict`` all spell the quantum models in lower case. This is the
        spelling the upper-case-only list in ``qc_winner_finder`` could never match, and
        the reason its quantum branch was dead on real output. Still asserted rather than
        assumed by the tests below: if the writer ever started emitting ``QSVC``, they
        would be exercising a spelling no run produces and the fix they guard could rot
        underneath them unnoticed.
        """
        results, _ = cli_quantum_run
        assert list(results["model"]) == ["qsvc"]

    @pytest.mark.parametrize("metric", ("accuracy", "f1_score", "auc"))
    def test_the_metrics_survived_the_round_trip_through_the_csv(self, cli_quantum_run, metric):
        """Finite and in the unit interval, read back as numbers rather than strings.

        ``auc`` is a genuine ranking AUC now (QSVC inherits ``SVC.decision_function``,
        which ``extract_binary_scores`` uses), and NaN is its documented value when no
        score exists -- so a NaN arriving here would mean the score never reached
        ``modeleval`` through the CLI, which no unit test can see.
        """
        results, _ = cli_quantum_run
        values = pd.to_numeric(results[metric])
        assert values.notna().all(), f"{metric} came back as {list(results[metric])}"
        assert ((values >= 0.0) & (values <= 1.0)).all(), f"{metric} outside [0, 1]"

    def test_the_parameter_column_records_the_circuit_that_produced_the_row(self, cli_quantum_run):
        """``Model_Parameters`` is the only record of *which* quantum model this was.

        Two runs differing only in ``encoding`` produce identical column names, identical
        labels and identical metric keys; the feature map recorded here is the sole
        discriminator after the fact. Which is why the value is compared against the
        circuit the config asked for, key by key, and not searched for the substring
        ``"FeatureMap"``: every encoding QBioCode supports builds a class whose name
        contains that (``Z``/``ZZ``/``Pauli`` + ``FeatureMap``), so a row describing a
        circuit that was never run satisfies a substring check exactly as well as a
        correct one, and the discriminator discriminates nothing.

        It survives as a repr of a dict, which is what a reader of the CSV has to parse,
        so it is parsed here the same way -- a value ``literal_eval`` cannot read is
        already a row no reader can attribute to a circuit.
        """
        results, _ = cli_quantum_run
        recorded = results["Model_Parameters"].iloc[0]

        parsed = ast.literal_eval(recorded)
        assert isinstance(parsed, dict), f"Model_Parameters is not a dict repr: {recorded!r}"

        absent = "<absent>"
        pinned = {key: parsed.get(key, absent) for key in CLI_RECORDED_PARAMETERS}
        assert pinned == CLI_RECORDED_PARAMETERS, (
            "the row does not describe the circuit and estimator qsvc_args asked for: "
            f"recorded {pinned}, config asks for {CLI_RECORDED_PARAMETERS}"
        )

        contradicted = {
            key: {"recorded": parsed[key], "config": asked}
            for key, asked in CLI_CIRCUIT_SHAPE.items()
            if key in parsed and parsed[key] != asked
        }
        assert (
            not contradicted
        ), f"the row records circuit settings the run did not use: {contradicted}"


# ======================================================================================
# The winner finder, on the table QProfiler actually writes
# ======================================================================================

#: The files ``qml_winner`` writes *only* once its quantum branch has found a winner
#: (qc_winner_finder.py:110, :123, :125, :129), as templates on its tag. These four are
#: the diagnostic ones: measured against the old matching rule, a table whose only model
#: is ``qsvc`` left every one of them unwritten.
WINNER_OUTPUTS = (
    "{tag}_qml_winners.csv",
    "{tag}_winner_evals.csv",
    "{tag}_winner_score.csv",
    "{tag}_winner_eval_score.csv",
)

#: The two files it writes *before* reaching that branch (qc_winner_finder.py:69 and :84),
#: so they appear whichever way the matching goes. That is what makes them useful in the
#: negative test and pointless in the positive ones: a run that returned a winner wrote
#: them on the way in, so asserting them beside a winner pins nothing, while beside a
#: ``None`` they are the only thing separating "the matching refused this label" from
#: "the function died before the matching". These two and the four above are everything
#: ``qml_winner`` writes.
UNCONDITIONAL_OUTPUTS = (
    "{tag}_best_across_split.csv",
    "{tag}_best_permodel_summary.csv",
)


class TestTheWinnerFinderRecognisesTheLabelsARunWrites:
    """``qml_winner``'s quantum branch, on the labels QProfiler actually produces.

    It used to be dead on all of them. The branch selected quantum rows by comparing the
    ``model`` column with ``["QSVC", "QNN", "VQC", "PQK"]`` via ``.isin()`` -- upper case
    and exact -- so the lower-case ``qsvc`` a real run writes never matched, QPL was
    absent from the list at any casing, and a tuned ``<name>_opt`` label could not have
    matched either. ``qc_winner_finder.py:98-101`` now compares the lower-cased first
    ``_``-token against ``{qsvc, qnn, vqc, pqk, qpl}``.

    Every test here is handed the *same* real table from the CLI run above, differing only
    in how the ``model`` field is spelled. That isolates the outcome to the name matching:
    the grouping, the F1 selection and the file writing are identical across all of them,
    so a spelling that yields no winner has failed for exactly one reason.
    """

    def test_the_finder_fires_on_the_table_qprofiler_actually_writes(
        self, cli_quantum_run, tmp_path
    ):
        """The real frame, untouched: one dataset whose best -- only -- model is QSVC.

        This was an ``xfail(strict=True)``. The bug: the finder's list was upper case and
        compared exactly, so this table -- whose ``model`` column reads ``qsvc``, the
        spelling ``model_run`` passes through from ``args['model']`` -- selected no quantum
        rows at all, and ``qml_winner`` printed that CML won everywhere and returned
        ``None``. No genuine quantum run could ever produce a qml_winners.csv. Fixed by
        matching case-insensitively on the label's first token; this assertion is what
        keeps it fixed.

        The written files are checked as well as the return value, because the files are
        what a user reads. Only the four in ``WINNER_OUTPUTS`` are checked, though.
        Measured, by patching the old ``.isin`` rule back into ``qc_winner_finder`` and
        rerunning: this test reddens -- along with the four QPL and tuned cases below --
        and the output directory is left holding ``real_best_across_split.csv`` and
        ``real_best_permodel_summary.csv``. So the defect's shape was "the four winner
        files are missing", not an empty output directory, and the two files in
        ``UNCONDITIONAL_OUTPUTS`` are written before the quantum branch is reached: under
        the defect they were there, and here they are implied by ``found is not None``
        anyway, since the branch that returns a winner writes them on the way in.
        Asserting them here would pin nothing. Where they do discriminate is the negative
        case, ``test_a_classical_row_is_not_treated_as_a_quantum_winner``, which is where
        they are asserted.
        """
        results, raw = cli_quantum_run

        found = qml_winner(results, raw, str(tmp_path), "real")

        assert found is not None, (
            "qml_winner found no quantum winner in a real results table whose only "
            f"model is {list(results['model'])}, so no genuine run can ever produce a "
            "qml_winners.csv"
        )
        winners, winner_eval_score, _ = found
        assert set(winners["model"]) == {"qsvc"}
        assert set(winners["Dataset"]) == set(results["Dataset"])
        assert not winner_eval_score.empty
        for name in WINNER_OUTPUTS:
            written = tmp_path / name.format(tag="real")
            assert written.exists(), f"qml_winner reported a winner but wrote no {name}"

    def test_the_finder_still_fires_when_the_model_name_is_upper_cased(
        self, cli_quantum_run, tmp_path
    ):
        """Case-insensitive has to mean both directions.

        This was the control that established the rest of the function works while the
        real-table case was pinned: upper case was the only spelling the old list could
        match, so a winner here beside a ``None`` there localised the defect to the name
        matching and nothing else.

        It is kept because case-insensitivity is a property of the fix rather than a side
        effect of it, and this test is what a plausible wrong fix trips over. Measured, by
        patching each variant into ``qc_winner_finder`` and running this class: splitting
        the label on ``_`` but comparing the token case-sensitively -- the smallest change
        that satisfies the real table above -- reddens exactly the two upper-case
        spellings, this test and the ``QPL`` parametrization, and nothing else;
        lower-casing the old list while still comparing whole names reddens this test and
        every other positive case in the class except the plain lower-case one. Of the two
        upper-case cases this is the isolating one: it changes only the casing of the label
        the run really wrote, where ``QPL`` changes the name too.

        What no longer justifies it is a caller. ``tests/integration/`` did have one:
        ``test_pymfe_model_integration.py`` built its table with ``QSVC`` by hand as a
        workaround for this very defect. It spells it ``qsvc`` now
        (``TestTheWinnerFinder.QML_NAME``) and says in ``_scored_table``'s docstring that
        the workaround is retired, so the only upper case left in that directory is inside
        the historical note itself. Nothing produces this spelling today: what this test
        guards is the shape of the fix, not a writer.
        """
        results, raw = cli_quantum_run
        upper_cased = results.assign(model=results["model"].str.upper())

        found = qml_winner(upper_cased, raw, str(tmp_path), "control")

        assert found is not None, (
            "qml_winner found no quantum winner in a table whose only model is quantum "
            "and whose name is spelled the way its own list used to spell it"
        )
        winners, winner_eval_score, _ = found
        assert set(winners["model"]) == {"QSVC"}
        assert set(winners["Dataset"]) == set(results["Dataset"])
        assert not winner_eval_score.empty
        for name in WINNER_OUTPUTS:
            written = tmp_path / name.format(tag="control")
            assert written.exists(), f"qml_winner reported a winner but wrote no {name}"

    @pytest.mark.parametrize("spelling", (QPL_LABEL, "QPL"))
    def test_a_qpl_row_is_recognised_as_a_quantum_model(self, cli_quantum_run, tmp_path, spelling):
        """QPL is a first-class dispatch key, and its rows are quantum results.

        This was an ``xfail(strict=True)`` failing for two independent reasons, both now
        fixed: ``QPL`` was absent from the finder's list at any casing, and ``qpl_svc`` --
        the label a run really writes, since ``compute_qpl`` fans out one row per
        classical head as ``f"{model}_{head}"`` -- can never be covered by a list of exact
        names however it is spelled. Matching the first ``_``-token handles both. Both
        spellings stay parametrized: a future "fix" that merely appends ``"qpl"`` to a list
        of names would satisfy the ``QPL`` case and still lose every real QPL row, and the
        ``qpl_svc`` case is what catches that.
        """
        results, raw = cli_quantum_run
        relabelled = results.assign(model=spelling)

        found = qml_winner(relabelled, raw, str(tmp_path), "qpl")

        assert found is not None, (
            f"a results table whose only model is {spelling!r} produced no quantum "
            "winner, so QPL runs are invisible to the winner finder"
        )
        assert set(found[0]["model"]) == {spelling}

    @pytest.mark.parametrize("spelling", TUNED_QUANTUM_LABELS)
    def test_a_tuned_quantum_row_is_recognised_as_a_quantum_model(
        self, cli_quantum_run, tmp_path, spelling
    ):
        """Tuning a quantum model must not drop its row out of the winner table.

        Untested until now because tuning a quantum model is new: ``model_run`` dispatches
        to the ``_opt`` twin only when ``tune_quantum`` is set, and labels the row
        ``<name>_opt`` -- which QPL then turns into ``qpl_opt_<head>``, putting the marker
        in the middle rather than at the end. Neither shape was in the finder's old list,
        and neither is the shape the two pins above covered, so this was a second,
        independent way for a quantum row to be invisible to the finder: a tuned run --
        the expensive kind, and the kind whose results someone is most likely to go looking
        for -- would have produced an empty qml_winners.csv even after the casing fix.
        Matching the first token covers both without enumerating them.
        """
        results, raw = cli_quantum_run
        relabelled = results.assign(model=spelling)

        found = qml_winner(relabelled, raw, str(tmp_path), "tuned")

        assert found is not None, (
            f"a results table whose only model is {spelling!r} produced no quantum "
            "winner, so a tuned quantum run is invisible to the winner finder"
        )
        assert set(found[0]["model"]) == {spelling}

    @pytest.mark.parametrize("spelling", CLASSICAL_LABELS)
    def test_a_classical_row_is_not_treated_as_a_quantum_winner(
        self, cli_quantum_run, tmp_path, spelling
    ):
        """The other side of the fix: a loose match must not over-match.

        Widening an exact comparison into a case-insensitive prefix one is precisely the
        change that starts matching things it should not, and the failure would be silent
        in the expensive direction -- qml_winners.csv is read as evidence that a quantum
        method beat the classical ones on a dataset, so a classical row admitted to it
        turns the file into a claim about quantum advantage that the run does not support.
        ``svc`` is the case that matters: a classical dispatch key that is also a substring
        of ``qsvc``, so a matcher written as a containment test in either direction would
        admit it. ``dt_opt`` covers the same risk for the ``_opt`` handling.

        ``UNCONDITIONAL_OUTPUTS`` is asserted to exist so that a ``None`` here is known to
        come from the name matching rather than from the function failing before it: the
        two are indistinguishable in the return value, and those two files are exactly what
        the original defect left behind when it returned ``None``. This is the one place
        they discriminate, and they do -- an early ``return None`` patched in above the
        first ``to_csv`` leaves ``found is None`` satisfied and reddens only this loop.
        All four winner files are asserted absent, not just ``qml_winners.csv``: an
        over-matching rule writes the set, and ``winner_eval_score.csv`` is the one a
        reader takes as the claim that a quantum method won.
        """
        results, raw = cli_quantum_run
        relabelled = results.assign(model=spelling)

        found = qml_winner(relabelled, raw, str(tmp_path), "classical")

        assert found is None, (
            f"{spelling!r} is a classical model, but qml_winner reported it as a quantum "
            f"winner: {sorted(set(found[0]['model']))}"
        )
        for name in UNCONDITIONAL_OUTPUTS:
            assert (
                tmp_path / name.format(tag="classical")
            ).exists(), "qml_winner returned before reaching its quantum branch at all"
        for name in WINNER_OUTPUTS:
            written = tmp_path / name.format(tag="classical")
            assert (
                not written.exists()
            ), f"a table whose only model is {spelling!r} still produced a {name}"


# ======================================================================================
# QPL's projection cache
# ======================================================================================



    def test_both_parameter_columns_in_one_table_lose_no_rows(
        self, cli_quantum_run, tmp_path
    ):
        """A mixed tuned/untuned run carries both parameter columns, and neither is dropped.

        ``modeleval`` decides between ``Model_Parameters`` and ``BestParams_Tuned`` per row
        now, not per run, so a single ``grid_search: True`` run whose quantum models stay
        untuned (the documented default, absent ``tune_quantum``) writes a table holding
        BOTH -- each row populated in one and NaN in the other.

        ``qml_winner`` grouped on ``parameter_columns[0]``, whichever of the candidate
        names happened to come first. Every row belonging to the *other* column therefore
        had NaN in that grouping key, and pandas' ``groupby`` drops NaN keys by default --
        so those rows vanished from the winner table with no error, no warning and a
        plausible-looking CSV. That is the failure mode this test exists for: silence, not
        a crash. It coalesces now (``df[parameter_columns].bfill(axis=1).iloc[:, 0]``).

        Built by splitting the real table's single parameter column across the two names
        rather than by fabricating a frame, so the row count, the labels and the F1 values
        are the ones a genuine run produced.
        """
        results, raw = cli_quantum_run
        present = [
            name for name in ("Model_Parameters", "BestParams_Tuned", "BestParams_GridSearch")
            if name in results.columns
        ]
        assert len(present) == 1, (
            f"the CLI table already carries {present}; this test needs to create the mix "
            "itself so that it knows which rows went where"
        )
        source_column = present[0]

        # The real shape of a mixed run: a TUNED CLASSICAL row beside the UNTUNED QUANTUM
        # one. Built by copying the genuine quantum row and relabelling the copy, so the
        # dataset name, embedding and raw-evaluation join key stay real. The classical row
        # is given the lower F1 deliberately -- qml_winner only reports a dataset whose
        # best model is quantum, so a classical row that won would make the test pass for
        # the wrong reason.
        quantum_row = results.copy()
        classical_row = results.copy()
        classical_row["model"] = "dt_opt"
        classical_row["f1_score"] = quantum_row["f1_score"].astype(float) - 0.10

        quantum_row["Model_Parameters"] = quantum_row[source_column]
        quantum_row["BestParams_Tuned"] = None
        classical_row["BestParams_Tuned"] = classical_row[source_column]
        classical_row["Model_Parameters"] = None

        mixed = pd.concat([quantum_row, classical_row], ignore_index=True)
        if source_column not in ("Model_Parameters", "BestParams_Tuned"):
            mixed = mixed.drop(columns=[source_column])
        assert mixed["BestParams_Tuned"].notna().any(), "the tuned half is empty"
        assert mixed["Model_Parameters"].notna().any(), "the untuned half is empty"

        found = qml_winner(mixed, raw, str(tmp_path), "mixed")

        assert found is not None, (
            "qml_winner found no quantum winner once the parameters were split across "
            "both column names -- the rows whose parameters live in the column it did not "
            "group on have been dropped"
        )
        # Every row must survive the grouping, not just enough of them to leave the
        # dataset non-empty: the quantum row is the one whose parameters sit in the column
        # that does NOT sort first, so it is the one the old code dropped.
        # qml_winner returns (qml_winner, winner_eval_score, df_best); the first is the
        # frame of surviving rows, as the sibling tests above also read it.
        surviving = found[0]
        assert set(surviving["model"]) == {"qsvc", "dt_opt"}, (
            f"rows went missing in the grouping: {sorted(set(surviving['model']))}. Both "
            "the tuned-classical and the untuned-quantum row must survive."
        )
        assert set(surviving["Dataset"]) == set(results["Dataset"])

@pytest.fixture(scope="module")
def qpl_cache(tmp_path_factory):
    """A directory holding one completed baseline run, shared across the module.

    Module-scoped because the first ``compute_qpl`` call pays the qiskit warm-up; the
    projections themselves take milliseconds at this size.
    """
    directory = tmp_path_factory.mktemp("qpl_projections")
    _run_qpl(directory)
    assert len(_projections(directory)) == 2, "baseline run wrote no projection pair"
    return directory


class TestTheQplProjectionCache:
    """What is on disk after a run, and what a second run does with it."""

    @pytest.mark.parametrize(
        "overrides", [{"encoding": "ZZ"}, {"reps": 3}], ids=["encoding", "reps"]
    )
    def test_changing_the_feature_map_writes_a_separate_cache_entry(self, qpl_cache, overrides):
        """Different circuit, different projections, therefore different files.

        The QPL twin of tests/test_pqk_cache_key.py, which covers PQK only. Black-box on
        purpose: recomputing the fingerprint here would pass even if the fingerprint
        covered nothing.
        """
        before = _projections(qpl_cache)
        _run_qpl(qpl_cache, **overrides)
        after = _projections(qpl_cache)

        assert before < after, (
            f"changing {overrides} reused the existing cache: the projections on disk "
            "were computed with a different feature map"
        )
        assert len(after - before) == 2, f"expected a new pair, got {after - before}"

    def test_identical_settings_reuse_the_cache_without_opening_a_backend(
        self, qpl_cache, monkeypatch
    ):
        """A full cache hit must short-circuit before any session is opened.

        Also the premise of ``TestTheWarmCacheSeedLookup`` below: it is because a warm
        cache skips the whole quantum block that a missing ``seed`` surfaces instantly,
        with nothing in the traceback to connect it to a projection or a config.
        """

        def _fail(*args, **kwargs):
            raise AssertionError("a backend session was opened despite a full cache hit")

        monkeypatch.setattr(qutils, "get_backend_session", _fail)

        before = _projections(qpl_cache)
        _run_qpl(qpl_cache)  # exactly the baseline parameters
        assert _projections(qpl_cache) == before, "a cache hit wrote new files"

    @pytest.mark.xfail(
        strict=True,
        reason="qbiocode/learning/compute_qpl.py:221 requests the 'estimator' "
        "primitive unconditionally, while the `primitive` argument enters only the "
        "cache fingerprint (compute_qpl.py:157). So primitive='sampler' is accepted, "
        "recomputes the identical projection and stores it under a second key: the "
        "cache doubles and the reported measurement never happened. "
        "compute_pqk.py:137-148 rejects the same value for the same reason.",
    )
    def test_a_primitive_the_projection_never_uses_is_refused_rather_than_cached(self, tmp_path):
        """Either ``primitive`` changes the measurement, or it must be refused.

        Its own directory, not the module cache: this compares every projection in the
        directory byte for byte, so a file belonging to some other feature map would
        make the comparison meaningless.
        """
        _run_qpl(tmp_path, primitive="estimator")
        baseline = _digests(tmp_path)
        assert len(baseline) == 2, f"baseline wrote {sorted(baseline)}"

        # compute_pqk's contract: a primitive the computation cannot honour is a
        # ValueError naming the accepted value, raised before anything is written.
        try:
            _run_qpl(tmp_path, primitive="sampler")
        except ValueError as raised:
            assert "estimator" in str(
                raised
            ), f"the refusal does not name the primitive that is used: {raised}"
            assert _digests(tmp_path) == baseline, "a refused run still wrote a cache"
            return

        after = _digests(tmp_path)
        duplicates = {
            digest: sorted(name for name, value in after.items() if value == digest)
            for digest in set(after.values())
        }
        collisions = {d: names for d, names in duplicates.items() if len(names) > 1}
        assert not collisions, (
            "primitive='sampler' was accepted and produced projections identical to "
            f"the estimator ones under a second cache key: {list(collisions.values())}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="qbiocode/learning/compute_qpl.py:169 validates a cached projection on "
        "row count only. compute_pqk.py:256-263 also checks the width against "
        "3 * feat_dimension (one expectation value per Pauli X/Y/Z per qubit), so a "
        "QPL file of the wrong width -- what a change to the observable set would "
        "leave behind, since the fingerprint covers the inputs and not the output "
        "shape -- is loaded, fitted and reported as a complete row of metrics.",
    )
    def test_a_cached_projection_of_the_wrong_width_is_refused(self, tmp_path):
        """Silently wrong is the failure mode, which is why this is worth a guard.

        The corrupted pair is *consistent* -- both files the same wrong width -- so
        every downstream shape check agrees with itself and the heads fit happily on
        features that mean nothing.
        """
        directory = tmp_path / "width"
        directory.mkdir()
        _run_qpl(directory)

        expected_width = 3 * N_FEATURES
        for path, rows in (
            (next(directory.glob("*_train.npy")), N_TRAIN),
            (next(directory.glob("*_test.npy")), N_TEST),
        ):
            assert int(np.prod(np.load(path).shape[1:])) == expected_width
            np.save(path, np.zeros((rows, expected_width - 2)))

        with pytest.raises(ValueError) as raised:
            _run_qpl(directory)
        assert str(expected_width) in str(
            raised.value
        ), f"the message does not say what width was expected: {raised.value}"


# ======================================================================================
# QPL's boundary guards
# ======================================================================================

#: case -> the parameter a refusal has to name. ``compute_pqk`` validates every one of
#: these before it creates a directory or reads a cache (compute_pqk.py:107-179);
#: ``compute_qpl``, its twin, validates none of them.
GUARD_CASES = {
    "a_test_matrix_of_a_different_width": "X_test",
    "a_one_dimensional_training_matrix": "X_train",
    "args_that_is_not_a_mapping": "args",
    "a_data_key_that_is_not_a_string": "data_key",
}


def _guard_call(case, projection_dir):
    """The ``compute_qpl`` arguments for one boundary case."""
    X_train, X_test, y_train, y_test = _projection_dataset()
    args = _qpl_args(projection_dir)
    data_key = "guard"
    if case == "a_test_matrix_of_a_different_width":
        # One feature map is built for both matrices, from X_train's width.
        X_test = np.random.default_rng(1).random((N_TEST, N_FEATURES + 1))
    elif case == "a_one_dimensional_training_matrix":
        X_train = X_train[:, 0]
    elif case == "args_that_is_not_a_mapping":
        args = ["backend", "simulator"]
    elif case == "a_data_key_that_is_not_a_string":
        # Interpolated straight into the cache filename.
        data_key = 7
    else:  # pragma: no cover -- a parametrize id with no case is a test bug
        raise AssertionError(f"unhandled guard case {case!r}")
    return X_train, X_test, y_train, y_test, args, data_key


@pytest.mark.xfail(
    strict=True,
    reason="qbiocode/learning/compute_qpl.py:135 starts using its arguments with no "
    "validation of any of them, so each of these surfaces far from its cause: a "
    "mismatched test width as qiskit's \"Length of ('a[0]', 'a[1]') inconsistent with "
    'last dimension" from bindings_array.py, a 1-D matrix as IndexError: tuple index '
    "out of range, a non-mapping args as AttributeError: 'list' object has no attribute "
    "'get', and a non-string data_key as TypeError: can only concatenate str. "
    "compute_pqk.py:107-179 refuses all four by name.",
)
@pytest.mark.parametrize("case,parameter", sorted(GUARD_CASES.items()))
def test_compute_qpl_refuses_an_impossible_argument_by_name(case, parameter, tmp_path):
    """The message has to name the parameter, because the traceback does not.

    Only the parameter name is asserted, not the wording: any fix that says which
    argument was wrong satisfies this, while today's messages -- raised from qiskit,
    sklearn or the string concatenation in the cache filename -- name none of them.
    """
    X_train, X_test, y_train, y_test, args, data_key = _guard_call(case, tmp_path)

    with pytest.raises(ValueError) as raised:
        compute_qpl(
            X_train,
            X_test,
            y_train,
            y_test,
            args,
            data_key=data_key,
            classical_models=[HEAD],
        )
    assert parameter in str(raised.value), f"the refusal does not name {parameter}: {raised.value}"


# ======================================================================================
# The warm-cache missing-seed path
# ======================================================================================


@pytest.fixture(scope="module")
def warm_qpl_cache(tmp_path_factory):
    """A QPL projection pair on disk, so a second run reaches the head-fitting block."""
    directory = tmp_path_factory.mktemp("qpl_warm")
    _run_qpl(directory)
    return directory


@pytest.fixture(scope="module")
def warm_pqk_cache(tmp_path_factory):
    """The same for PQK, whose ``args['seed']`` lookup sits at compute_pqk.py:457."""
    directory = tmp_path_factory.mktemp("pqk_warm")
    X_train, X_test, y_train, y_test = _projection_dataset()
    args = {
        "backend": "simulator",
        "seed": 7,
        "grid_search": False,
        "pqk_projection_dir": str(directory),
    }
    compute_pqk(X_train, X_test, y_train, y_test, args, data_key="ds")
    return directory


class TestTheWarmCacheSeedLookup:
    """``args['seed']`` is read bare, after the projection, in both projection models.

    A config that omits ``seed`` is an ordinary mistake -- it is one key in a
    seventy-line YAML file, and nothing else in a run requires it -- and on a warm
    cache the whole quantum block is skipped, so the failure arrives in milliseconds
    with a traceback that points at ``create_svc_model`` and mentions neither the
    config, the key's purpose, nor the fact that a projection was already computed and
    is still valid. ``model_run`` sets the standard for this repo: its model-name
    validation names every accepted value and explains the ``_opt`` rule.

    The cold path has the same defect and costs more: there the projection is computed
    in full and then thrown away by the same lookup.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="qbiocode/learning/compute_qpl.py:380-392 read args['seed'] with no "
        "guard, so a config missing that key dies as a bare KeyError('seed') raised "
        "from a create_*_model call after the projection is loaded -- naming neither "
        "the key's purpose nor what the user must add.",
    )
    def test_qpl_names_the_missing_seed_and_what_it_is_for(self, warm_qpl_cache):
        X_train, X_test, y_train, y_test = _projection_dataset()
        args = _qpl_args(warm_qpl_cache)
        del args["seed"]

        with pytest.raises(ValueError) as raised:
            compute_qpl(
                X_train,
                X_test,
                y_train,
                y_test,
                args,
                data_key="ds",
                classical_models=[HEAD],
            )
        assert "seed" in str(
            raised.value
        ), f"the refusal does not name the key that is missing: {raised.value}"

    @pytest.mark.xfail(
        strict=True,
        reason="qbiocode/learning/compute_pqk.py:457 reads args['seed'] with no guard "
        "-- the same bare KeyError('seed') as compute_qpl, in the model this one "
        "otherwise validates thoroughly (compute_pqk.py:107-179).",
    )
    def test_pqk_names_the_missing_seed_and_what_it_is_for(self, warm_pqk_cache):
        X_train, X_test, y_train, y_test = _projection_dataset()
        args = {
            "backend": "simulator",
            "grid_search": False,
            "pqk_projection_dir": str(warm_pqk_cache),
        }

        with pytest.raises(ValueError) as raised:
            compute_pqk(X_train, X_test, y_train, y_test, args, data_key="ds")
        assert "seed" in str(
            raised.value
        ), f"the refusal does not name the key that is missing: {raised.value}"


# ======================================================================================
# vqc and qnn reproducibility
# ======================================================================================

#: The three quantum seeds the tests below run at. 7 twice (same answer required) and
#: two others (at least one different answer required). Fixed values, so the outcome of
#: every assertion here is fixed too: nothing in this section samples a seed.
SAME_SEED = 7
OTHER_SEEDS = (99, 1234)

VARIATIONAL_MODELS = ("qnn", "vqc")


@pytest.fixture(scope="module")
def variational_data():
    """24 rows, 2 features, split 18/6 and MinMax-scaled on the training rows only.

    Scaling is not cosmetic: the feature maps encode magnitudes as rotation angles, so
    unscaled features land anywhere on the circle and both models collapse to chance.
    Same shape as tests/test_quantum_models.py's ``data`` fixture, which is what the
    published runtimes for these two models were measured on.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test = scale_train_test(X[:18], X[18:], scaling="MinMaxScaler")
    return X_train, X_test, y[:18], y[18:]


@pytest.fixture(scope="module")
def variational_run(variational_data):
    """``model_run`` per (model, q_seed, repeat), computed once and remembered.

    A callable rather than a precomputed dict so that a model which fails to fit breaks
    only its own parametrizations. ``repeat`` distinguishes two genuinely separate runs
    at the same seed -- the whole point of the first test -- from a cache hit.
    """
    X_train, X_test, y_train, y_test = variational_data
    cache: dict[tuple, tuple] = {}

    def run(model, q_seed, repeat=0):
        key = (model, q_seed, repeat)
        if key not in cache:
            args = {
                "backend": "simulator",
                "shots": 64,
                "n_jobs": 1,
                "grid_search": False,
                "seed": 7,  # held fixed: only q_seed varies below
                "q_seed": q_seed,
                "model": [model],
            }
            out = model_run(X_train, X_test, y_train, y_test, "variational", args)
            row = out[f"results_{model}"][0]
            predictions = tuple(np.asarray(out[f"y_predicted_{model}"][0]).tolist())
            cache[key] = (row["accuracy"], row["f1_score"], predictions)
        return cache[key]

    return run


class TestTheVariationalModelsAreReproducible:
    """``q_seed`` reaching the variational stack, pinned from both sides at once.

    Both models draw their optimizer's initial point through ``TrainableModel``, which
    reads ``qiskit_machine_learning.utils.algorithm_globals`` -- a different singleton
    from the ``qiskit_algorithms.utils`` one that used to be the only one seeded. Until
    ``_call_with_global_seeds`` seeded both, VQC and QNN started from OS entropy and two
    runs at the same ``q_seed`` disagreed.

    ``tests/test_quantum_models.py`` guards the same-seed half of that itself:
    ``TestTheSeedReachesTheQuantumStack::test_two_runs_at_the_same_seed_agree`` is
    parametrized over ``pqk``, ``qsvc``, ``vqc`` and ``qnn``, on a fixture built the same
    way as ``variational_data`` below, and compares predictions as well as accuracy. The
    two variational models are named there deliberately, as the models that check exists
    for. So ``test_two_runs_at_the_same_quantum_seed_agree[vqc]`` and ``[qnn]`` overlap it,
    and the overlap is not the reason this class exists.

    The second test is. Nothing in that file varies ``q_seed`` and requires the *answer*
    to change: its other seed tests assert on ``_call_with_global_seeds`` -- which
    singletons it writes -- and never on a model's output. That matters because "two runs
    agree" is satisfied exactly as well by a ``q_seed`` that never reaches the optimizer
    at all, ``seed`` alone being enough to make a run repeatable through numpy. The pair
    is also why the same-seed case is kept here rather than deferred upward: both halves
    read one memoized ``variational_run``, so the agreeing runs and the differing ones are
    provably the same computation at three seeds, which a control drawn from another
    module's fixture would not be.
    """

    @pytest.mark.parametrize("model", VARIATIONAL_MODELS)
    def test_two_runs_at_the_same_quantum_seed_agree(self, model, variational_run):
        """Same input, same ``q_seed``, same answer -- predictions included.

        The predictions are compared rather than only the metrics because a variational
        fit that started from a different point often lands on a *nearby* accuracy: on
        6 test rows there are only seven possible values, so metric equality alone would
        pass by coincidence a good fraction of the time.
        """
        first = variational_run(model, SAME_SEED, repeat=0)
        second = variational_run(model, SAME_SEED, repeat=1)
        assert first == second, (
            f"two {model} runs at q_seed={SAME_SEED} disagreed; the quantum seed is not "
            "reaching every global RNG the variational stack reads "
            f"({first} vs {second})"
        )

    @pytest.mark.parametrize("model", VARIATIONAL_MODELS)
    def test_the_quantum_seed_is_what_that_answer_depends_on(self, model, variational_run):
        """The other half: a seed that is ignored would also reproduce perfectly.

        ``seed`` is held fixed across these runs, so only ``q_seed`` differs -- which
        makes this the one assertion in the file that fails if ``q_seed`` stops reaching
        the optimizer while ``np.random`` keeps being seeded. Two of three seeds are
        allowed to coincide: with a two-parameter ansatz on 6 test rows a collision is
        ordinary, and requiring all three to differ would pin a qiskit version rather
        than QBioCode's behaviour.
        """
        outcomes = {q_seed: variational_run(model, q_seed) for q_seed in (SAME_SEED, *OTHER_SEEDS)}
        assert len(set(outcomes.values())) > 1, (
            f"{model} produced the same answer at q_seed "
            f"{sorted(outcomes)} -- args['q_seed'] is not reaching the optimizer, so "
            "the reproducibility above is vacuous"
        )


def test_the_two_projection_models_keep_their_caches_apart(tmp_path):
    """A last cheap seam: PQK and QPL must not read each other's projections.

    Both write ``<name>_projection_<data_key>_<fingerprint>_{train,test}.npy``, and both
    reshape whatever they load to (n, -1) without asking which model wrote it. The
    default directories differ, so only the file-name prefix keeps them apart when the
    two ``*_projection_dir`` keys are pointed at one location -- which is the natural
    thing to configure, since the projections of one run belong together.
    """
    X_train, X_test, y_train, y_test = _projection_dataset()
    shared = {
        "backend": "simulator",
        "seed": 7,
        "grid_search": False,
        "pqk_projection_dir": str(tmp_path),
        "qpl_projection_dir": str(tmp_path),
    }
    compute_pqk(X_train, X_test, y_train, y_test, dict(shared), data_key="same")
    compute_qpl(
        X_train,
        X_test,
        y_train,
        y_test,
        dict(shared),
        data_key="same",
        classical_models=[HEAD],
    )

    written = sorted(path.name for path in tmp_path.glob("*_projection_*.npy"))
    assert len(written) == 4, f"expected two pairs, one per model, got {written}"
    assert (
        len({name.split("_projection_")[0] for name in written}) == 2
    ), f"the two models' cache files are not distinguishable by name: {written}"
