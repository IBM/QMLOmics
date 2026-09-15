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

"""``compute_qensemble``: the one learner in the package that nothing ever ran.

The quantum ensemble is exported from ``qbiocode.learning``, advertised in that
subpackage's docstring with two worked examples, and cited to two papers -- and until
this file it had no executed coverage of any kind. It is the only learner absent from
*both* ``model_run``'s ``compute_ml_dict`` and ``qbiocode.__all__``, so every test that
enumerates the models walked straight past it -- those enumerations are keyed on
``compute_ml_dict`` or on ``qbiocode.__all__``, and it is in neither.
test_model_contract_matrix.py is the strictest of them and reads both -- it walks the
dispatch table and checks every key against the two export lists
(``test_every_dispatchable_learner_is_exported_from_the_learning_subpackage`` and
``test_every_dispatchable_learner_is_reachable_from_the_package_root``) -- so a learner
missing from the table is missing from everything it compares. And ``grep -rl qensemble
tests/`` returned nothing at all; case-sensitively it now returns this file and no other.

That made it the one corner of the package where a rename could not fail. It is the sole
caller of ``normalize_data``, ``prepare_training_set``, ``retrieve_probabilities`` and
``execute_circuit`` -- ``execute_circuit`` has no other caller anywhere in the tree -- so
any of those four could have been renamed, re-signatured or deleted with the suite still
green. It also assembles its circuits by hand from ``QuantumCircuit`` and
``UnitaryGate`` instead of going through qiskit-machine-learning, which means a qiskit
API change reaches it and reaches nothing else. And it writes a ``results_QEnsemble`` row
through the same ``modeleval`` as every other learner, so it shares the ModelResults.csv
schema -- including the ``auc`` column that has just changed meaning from a balanced
accuracy to a real ranking AUC -- with nothing holding it to that schema.

**It works.** It fits, and on 24 points laid out by direction it separates them -- at
the default seed and this file's 64 shots, perfectly: accuracy 1.0 and AUC 1.0 -- so what
follows covers a working learner rather than pinning a stub. That pair belongs to one Aer
RNG stream rather than to the circuit (the closing paragraph), which is why nothing below
asserts it. Six defects surfaced while establishing that. Defect 1 has since been fixed
and is now guarded by a positive assertion; the other five are still pinned below with
``xfail(strict=True)`` rather than written down as correct. The numbering is kept as it
was, because several of the pins cross-reference each other by number:

1. **Fixed.** ``seed`` did not make a run reproducible. It fixed the training subset and
   every gate, but ``execute_circuit`` sampled its shots with no ``seed_simulator``, so
   accuracy, f1 and auc all moved between two identical calls. With that seeding patched
   back out, sixty identical calls on this file's own split return sixty *distinct* score
   arrays: that count is the half of the measurement that reproduces, and it is the half
   ``TestTheSeedFixesTheResult`` is built on. The metric spread is the other half and is
   not a bound -- those sixty accuracies ran from 0.5, chance on a split this learner
   separates perfectly once it is seeded, up to 1.0, and a range quoted "over twenty
   runs" is only whichever draw that batch made: the three consecutive twenty-call
   windows inside those sixty reported 0.5-1.0, 0.625-1.0 and 0.75-1.0. An earlier
   revision of this paragraph quoted one of them, 0.75 to 1.0, as if it were a floor.
   ``execute_circuit`` now takes a ``seed`` and forwards it as ``seed_simulator``, and
   ``compute_qensemble`` passes its own, so ``TestTheSeedFixesTheResult`` asserts what
   used to be pinned: three identical calls agree probability for probability, and one
   fixed circuit sampled at four separated seeds does not.
2. An unrecognised ``mode`` -- including the capitalisation slip ``"Balanced"`` -- is
   accepted and silently drops the ensemble entirely.
3. The ``n_train`` actually used is ``2 * (n_train // 2)``, but ``Model_Parameters``
   records the number that was *asked* for.
4. The width guard's limit of 36 qubits sits above what the Aer statevector simulator
   will accept, so for a band of widths the guard never fires.
5. The ``random_unitary`` branch allocates a ``pred_qubit`` that no instruction touches.
6. On the ``random_unitary`` branch only ``mode="balanced"`` is implemented. ``mode`` is
   documented independently of ``ensemble_method``, but that branch's ``if`` has no
   ``elif``, so ``"unbalanced"`` and ``"pair_sample"`` -- correctly spelled, documented
   values -- emit no controlled unitary and no controlled swap at all. It is defect 2
   again, reached without a typo: the row records the mode it was asked for and reads
   exactly like a real ensemble run.

The exactness of the wiring tests here is deliberate rather than lucky. Now that the
sampling is seeded a rerun reproduces its numbers, but those numbers are a property of
one Aer RNG stream rather than of the circuit -- an Aer version bump moves every count
that is not certain, and on this fixture the class-0 probabilities sit within shot noise
of the 0.5 threshold, so accuracy moves with the seed even though the ranking does not.
The assertions that have to be exact are therefore still built on degenerate inputs whose
measurement outcome has probability one (a test point identical to its training points),
which is the only way to check the gate wiring without a tolerance loose enough to also
pass for a circuit that is wrong; and the metric assertions are still bounds.
Reproducible is not the same as robust.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import pathlib

import numpy as np
import pytest
from qiskit.exceptions import QiskitError
from qiskit_aer import AerSimulator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Imported at module scope, deliberately: importing `qbiocode` orders the OpenMP
# runtimes (tests/test_openmp_import_order.py), and tests/test_suite_hygiene.py forbids
# reaching first-party or base-requirement modules through pytest.importorskip.
import qbiocode
from qbiocode import learning
from qbiocode.learning import compute_dt, compute_qensemble
from qbiocode.utils import (
    execute_circuit,
    label_to_array,
    normalize_data,
    prepare_training_set,
    retrieve_probabilities,
)

#: The *module*, not the function. ``qbiocode.learning`` binds the name
#: ``compute_qensemble`` to the function, which shadows the submodule of the same name,
#: so ``import qbiocode.learning.compute_qensemble as m`` hands back the function and
#: ``m.build_ensemble_circuit`` raises AttributeError. ``import_module`` reads
#: ``sys.modules`` and returns the module. The circuit builders and the ``modeleval``
#: reference this file patches are only reachable this way.
QENSEMBLE = importlib.import_module("qbiocode.learning.compute_qensemble")

# The ensemble prints nothing by default, but sklearn's f1_score warns on a single-class
# test set (UndefinedMetricWarning is a UserWarning); that warning is a subject of one
# test here, not a failure of any.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

#: The keys ``modeleval`` writes for a row whose parameters did not come from a search --
#: the ModelResults.csv schema, shared with all 28 dispatchable learners. It used to be
#: "the keys written when ``args['grid_search']`` is off", which was the defect: the flag
#: describes the run and the column describes the row. This learner has no tuned twin, so
#: these are its keys under either flag (``_was_tuned`` in model_evaluation.py).
DOCUMENTED_KEYS = frozenset({"model", "accuracy", "f1_score", "time", "auc", "Model_Parameters"})

#: The three sampling strategies the docstring documents.
MODES = ("balanced", "unbalanced", "pair_sample")

#: Small enough to keep the file in the default tier. It is *not* enough to spread the
#: eight test points' probabilities: seeded, they land on two values, 29/64 and 44/64,
#: because the shot grid is coarse and every point's circuit is sampled under the same
#: simulator seed. Unseeded, shot noise alone scattered them across most of the eight --
#: five to eight distinct values from run to run in one batch of twenty, a draw and not a
#: bound -- which is why the "auc is a ranking" test used to count distinct values: at
#: three, the threshold it asked for, it was counting noise.
#: ``TestTheAucIsTheEnsembleProbability`` now establishes the ranking on test points that
#: differ by geometry instead.
SHOTS = 64

#: The value ``compute_qensemble`` documents as its default.
SEED = 123

#: Aer derives its statevector width limit from available memory, so it is a property of
#: the machine, not of qiskit. Read once, and used only to skip the guard pin on a box
#: that really could simulate 36 qubits.
AER_MAX_QUBITS = AerSimulator(method="statevector").num_qubits

MODEL_RUN_SOURCE = pathlib.Path(qbiocode.__file__).resolve().parent / "evaluation" / "model_run.py"


def _dispatch_keys():
    """The keys of ``compute_ml_dict``, read out of ``model_run``'s body.

    The table is built from lazy imports *inside* the function, so there is no object to
    import; it has to be parsed. Only the keys are needed here --
    tests/test_model_contract_matrix.py owns resolving the values.
    """
    tree = ast.parse(MODEL_RUN_SOURCE.read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "model_run"]
    assert functions, f"model_run is no longer a module-level function in {MODEL_RUN_SOURCE}"
    for node in ast.walk(functions[0]):
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "compute_ml_dict" for target in node.targets
        ):
            keys = [k.value for k in node.value.keys]
            # Non-vacuity: an empty table would make the export invariants below pass by
            # comparing two empty sets.
            assert len(keys) > 1, "compute_ml_dict parsed as empty; this reader is stale"
            return frozenset(keys)
    raise AssertionError("could not find compute_ml_dict in model_run")


DISPATCH_KEYS = _dispatch_keys()


@contextlib.contextmanager
def captured_scores():
    """Record the ``y_score`` array each fit hands to ``modeleval``.

    Neither the score nor a fitted estimator survives into the results frame -- this
    learner has no estimator at all -- so the only way to check that ``auc`` is the ROC
    AUC of the ensemble's own probabilities is to read the array on its way past. The
    name is rebound inside the learner's module, which is where the call resolves; the
    real ``modeleval`` still runs and the real circuits still execute.
    """
    recorded: list[np.ndarray] = []
    original = QENSEMBLE.modeleval

    def spy(*args, **kwargs):
        recorded.append(np.asarray(kwargs["y_score"], dtype=float))
        return original(*args, **kwargs)

    QENSEMBLE.modeleval = spy
    try:
        yield recorded
    finally:
        QENSEMBLE.modeleval = original


@contextlib.contextmanager
def captured_circuits():
    """Record every circuit a fit hands to ``execute_circuit``, one per test point.

    A results row keeps two numbers about the circuit, ``n_qubits`` and
    ``circuit_depth``, and neither of them says whether an ensemble was built -- an
    empty ensemble is the same width and a plausible depth. This learner returns no
    estimator and no circuit either, so reading the circuits on their way to the
    simulator is the only way for a test that goes in through the public API to assert
    anything at all about the wiring that call produced. The name is rebound inside the
    learner's module, which is where the call resolves; the real simulation still runs.

    The spy mirrors ``execute_circuit``'s signature in full, ``seed`` included, and
    forwards it. A spy that silently drops a keyword its caller passes does not weaken a
    test, it breaks it: when ``seed`` was added to ``execute_circuit`` this spy raised
    ``TypeError: spy() got an unexpected keyword argument 'seed'`` from inside the
    learner, which reads as a defect in the learner and is not one.
    """
    recorded = []
    original = QENSEMBLE.execute_circuit

    def spy(circuit, n_shots=8192, device="CPU", seed=None):
        recorded.append(circuit)
        return original(circuit, n_shots=n_shots, device=device, seed=seed)

    QENSEMBLE.execute_circuit = spy
    try:
        yield recorded
    finally:
        QENSEMBLE.execute_circuit = original


def _control_register(circuit):
    """The ``control`` register, found by name and not by position.

    Every detector below needs to know which qubits are control qubits, and the obvious
    shortcut -- ``control`` is declared first, so its qubits are absolute indices
    ``0 .. d-1`` -- hardcodes the very layout these tests exist to verify. The two
    branches already disagree about the order of everything after it
    (``random_unitary`` inserts ``pred_qubit`` before ``test_label``), so an index-based
    reader would keep returning plausible numbers about the wrong qubits the day a
    register moves.
    """
    registers = [register for register in circuit.qregs if register.name == "control"]
    assert len(registers) == 1, (
        f"expected exactly one register named 'control', found {[r.name for r in circuit.qregs]}"
    )
    return registers[0]


def _control_conditioned_swaps(circuit):
    """How many controlled swaps in ``circuit`` are conditioned on a control qubit.

    A circuit with none of these has no ensemble in it: the Hadamards still put the
    control register in superposition, but nothing downstream depends on it, so all
    ``2**d`` branches are the same single classifier. The swap test's own ``cswap``
    (conditioned on ``test_label``, not on ``control``) is excluded, which is what makes
    zero the honest reading for an ensemble that was never built.
    """
    control = set(_control_register(circuit))
    return sum(
        1
        for instruction in circuit.data
        if instruction.operation.name == "cswap"
        and any(bit in control for bit in instruction.qubits)
        and any(bit not in control for bit in instruction.qubits)
    )


def _control_conditioned_operations(circuit):
    """The same count, for a branch whose ensemble is not made of swaps.

    ``random_unitary`` conditions Haar unitaries rather than swaps, so the swap detector
    would read low there for a reason that is not a defect. Any multi-qubit instruction
    reaching across the control register boundary counts: it is the ensemble operations
    and nothing else, since ``h`` and ``x`` on ``control`` act on a control qubit alone
    and every other gate leaves the register untouched.
    """
    control = set(_control_register(circuit))
    return sum(
        1
        for instruction in circuit.data
        if len(instruction.qubits) > 1
        and any(bit in control for bit in instruction.qubits)
        and any(bit not in control for bit in instruction.qubits)
    )


def _conditioned_control_indices(circuit):
    """Which control qubits, by index within their own register, condition something.

    Superposition on ``d`` qubits is half of "``2**d`` ensemble members"; the other half
    is that each of those qubits is *read* by an operation downstream. A control qubit
    that nothing is conditioned on splits the state and then recombines branches that
    were never made to differ, so this returning less than ``set(range(d))`` means the
    ensemble is narrower than the ``d`` it reports.
    """
    control = _control_register(circuit)
    members = set(control)
    return {
        control.index(bit)
        for instruction in circuit.data
        if len(instruction.qubits) > 1
        and any(qubit not in members for qubit in instruction.qubits)
        for bit in instruction.qubits
        if bit in members
    }


def _hadamards_on(circuit, register):
    """How many Hadamards act on a qubit of ``register``.

    Counted per register, never over the whole circuit: an ensemble circuit carries
    Hadamards in two unrelated places -- ``d`` on ``control``, which *is* the ensemble,
    and two on ``test_label``, which are the swap test in the final classification step.
    A whole-circuit count of them cannot tell the two apart, so it stays satisfied by the
    swap test alone after the ensemble's own superposition has been deleted.
    """
    members = set(register)
    return sum(
        1
        for instruction in circuit.data
        if instruction.operation.name == "h"
        and all(bit in members for bit in instruction.qubits)
    )


def _expected_control_conditioned_swaps(mode, n_obs, n_swap, d):
    """How many control-conditioned swaps each documented mode's loops emit.

    Derived from the loop structure rather than copied off a run, so these are three
    claims about how the three strategies differ and not three numbers that happen to
    match today. Every sample exchange emits *two* ``cswap`` instructions -- one moving
    the sample's data qubit, one moving its label qubit alongside it -- and each control
    qubit is used twice, once before and once after the ``x`` that selects its other
    branch. So:

    * ``balanced`` exchanges ``n_swap`` pairs drawn inside class 0 and ``n_swap`` inside
      class 1, on both halves of the first ``d - 1`` control qubits, then one last
      data/label pair on ``control[d-1]``: ``2 * 2 * 2 * n_swap * (d - 1) + 2``.
    * ``unbalanced`` exchanges ``n_swap`` pairs drawn from the whole training set with
      the classes ignored, on both halves of all ``d`` control qubits:
      ``2 * 2 * n_swap * d``.
    * ``pair_sample`` shuffles the samples into ``n_obs / 2`` disjoint pairs and
      exchanges every one of them, on both halves of all ``d`` control qubits:
      ``2 * 2 * (n_obs / 2) * n_swap * d``.

    ``qubits_per`` is 1 everywhere in this file (2 features). A wider encoding does not
    change any of these: the builder swaps the single randomly chosen index ``U_b`` of a
    sample's register, not the whole register.
    """
    if mode == "balanced":
        return 8 * n_swap * (d - 1) + 2
    if mode == "unbalanced":
        return 4 * n_swap * d
    if mode == "pair_sample":
        return 2 * n_swap * d * n_obs
    raise AssertionError(f"{mode!r} is not one of the documented modes {MODES}")


def _gate_structure(circuit):
    """A comparable fingerprint of a circuit: every gate name with its qubit indices.

    ``str(circuit.data)`` cannot be used for this -- the repr of an ``initialize``
    instruction embeds object identities, so two structurally identical circuits compare
    unequal and a reproducibility test built on it would pass for the wrong reason.
    """
    return tuple(
        (instruction.operation.name, tuple(circuit.find_bit(b).index for b in instruction.qubits))
        for instruction in circuit.data
    )


@pytest.fixture(scope="module")
def data():
    """24 unit vectors laid out by angle, split 16/8, exactly balanced in both halves.

    The classifier at the heart of this learner is a cosine similarity, so it can only
    separate classes that differ in *direction*. Points are therefore placed on the unit
    circle with the two classes in disjoint arcs, deterministically and without an RNG:
    a Gaussian blob would leave the ensemble at chance, and a finite-and-in-range
    assertion cannot tell a chance-level fit from a broken one.

    Both features are needed and both are used: ``build_ensemble_circuit`` reads
    ``log2(n_features)`` as its per-sample qubit count, so 2 features is the narrowest
    circuit the encoding admits.
    """
    class_zero = np.linspace(0.10, 0.60, 12)
    class_one = np.linspace(0.95, 1.45, 12)
    angles = np.empty(24)
    angles[0::2] = class_zero
    angles[1::2] = class_one
    X = np.column_stack([np.cos(angles), np.sin(angles)])
    y = np.zeros(24, dtype=int)
    y[1::2] = 1
    # 8 of each class in the training half, 4 of each in the test half. prepare_training_set
    # samples n/2 per class without replacement, so the training half has to hold both.
    return X[:16], X[16:], y[:16], y[16:]


@pytest.fixture(scope="module")
def base_run(data):
    """One real fit at the documented defaults, plus the score array it produced.

    Module-scoped and shared: a fit is ~0.9 s, and ten tests read from this one -- seven
    in ``TestTheEnsembleFitsAndWritesTheStandardRow`` and three in
    ``TestTheAucIsTheEnsembleProbability``. The two written out because the count went
    stale once already: it said nine while ten tests took the fixture.
    """
    X_train, X_test, y_train, y_test = data
    with captured_scores() as recorded:
        frame = compute_qensemble(
            X_train,
            X_test,
            y_train,
            y_test,
            {"grid_search": False},
            data_key="qens",
            n_train=4,
            d=2,
            n_shots=SHOTS,
            seed=SEED,
        )
    assert len(recorded) == 1, "modeleval was called more than once for a single fit"
    return frame.to_dict(), recorded[0]


class TestTheEnsembleFitsAndWritesTheStandardRow:
    """The contract every reader of ModelResults.csv depends on."""

    def test_a_fit_returns_the_three_columns_every_learner_writes(self, base_run):
        """``modeleval`` names its columns after the model; nothing here may differ."""
        columns, _ = base_run
        assert set(columns) == {
            "y_test_QEnsemble",
            "y_predicted_QEnsemble",
            "results_QEnsemble",
        }
        # Each column is a dict keyed by frame index, not a list: model_run ends in
        # DataFrame.to_dict(), and every downstream reader indexes with 0.
        for column in columns.values():
            assert set(column) == {0}

    def test_the_result_row_carries_exactly_the_keys_a_dispatched_learner_carries(self, base_run, data):
        """Parity with a real sibling, not with a hardcoded list.

        The ensemble is not in ``compute_ml_dict``, so no matrix test compares it with
        anything. Fitting the cheapest dispatchable learner on the same split and
        diffing the row keys is what makes "shaped like every other learner's" checkable
        rather than asserted.
        """
        columns, _ = base_run
        X_train, X_test, y_train, y_test = data
        sibling = compute_dt(
            X_train,
            X_test,
            y_train,
            y_test,
            {"grid_search": False, "seed": 7, "n_jobs": 1},
            model="dt",
        ).to_dict()

        assert set(columns["results_QEnsemble"][0]) == set(sibling["results_dt"][0])
        assert set(columns["results_QEnsemble"][0]) == set(DOCUMENTED_KEYS)
        # The column-naming convention travels with the row shape.
        assert {name.rsplit("_", 1)[0] for name in columns} == {
            name.rsplit("_", 1)[0] for name in sibling
        }

    def test_the_recorded_model_name_is_the_default_the_signature_documents(self, base_run):
        """'QEnsemble' is the results label, and the docstring's examples never set it."""
        columns, _ = base_run
        assert columns["results_QEnsemble"][0]["model"] == "QEnsemble"

    def test_the_recorded_metrics_are_the_metrics_of_the_recorded_predictions(self, base_run, data):
        """accuracy and f1 must describe the very predictions stored beside them.

        ``auc`` is deliberately not recomputed here -- it comes from a score array that
        the frame does not carry. See ``TestTheAucIsTheEnsembleProbability``.
        """
        columns, _ = base_run
        _, _, _, y_test = data
        y_true = columns["y_test_QEnsemble"][0]
        y_predicted = columns["y_predicted_QEnsemble"][0]
        row = columns["results_QEnsemble"][0]

        np.testing.assert_array_equal(y_true, y_test)
        assert y_predicted.shape == y_test.shape
        assert set(np.unique(y_predicted)) <= {0, 1}
        assert row["accuracy"] == pytest.approx(accuracy_score(y_true, y_predicted))
        assert row["f1_score"] == pytest.approx(
            f1_score(y_true, y_predicted, average="weighted")
        )
        assert row["time"] > 0

    def test_the_ensemble_beats_chance_on_directionally_separable_points(self, base_run):
        """Otherwise every other assertion here would also pass for a broken circuit.

        Bounds, not a point estimate. The shot sampling is seeded now (see
        ``TestTheSeedFixesTheResult``), so this fit reproduces exactly, at accuracy 1.0
        and auc 1.0 -- but the exact counts belong to one Aer RNG stream and not to the
        circuit. At seed 999 the same split returns accuracy 0.5 with auc still 1.0,
        because the class-0 probabilities sit within shot noise of the 0.5 threshold that
        turns them into labels: the ranking survives what the argmax does not. So a floor
        of "better than a coin" remains the honest assertion rather than the exact pair,
        and it still fails outright for an ensemble that has stopped classifying.

        What makes that floor safe is the seeding and not a margin. This docstring used to
        close with "over twenty unseeded runs accuracy stayed in [0.75, 1.0] and auc in
        [0.9375, 1.0]", which read like a bound and was one batch's draw: with
        ``seed_simulator`` patched back out, sixty identical calls span accuracy 0.5 to 1.0
        and auc 0.8125 to 1.0, and that 0.5 -- one call in sixty -- is this very assertion
        going red, since it asks for strictly better than a coin. Unseeded there was no
        floor to quote.
        """
        columns, _ = base_run
        row = columns["results_QEnsemble"][0]
        assert row["accuracy"] > 0.5
        assert row["auc"] > 0.5

    def test_the_recorded_parameters_echo_the_keywords_it_was_called_with(self, base_run):
        """``Model_Parameters`` is the only record of how a row was produced."""
        columns, _ = base_run
        params = columns["results_QEnsemble"][0]["Model_Parameters"]
        assert set(params) == {
            "n_train",
            "n_swap",
            "d",
            "mode",
            "ensemble_method",
            "n_shots",
            "seed",
            "n_qubits",
            "circuit_depth",
        }
        assert params["n_train"] == 4
        assert params["n_swap"] == 1
        assert params["d"] == 2
        assert params["mode"] == "balanced"
        assert params["ensemble_method"] == "swap"
        assert params["n_shots"] == SHOTS
        assert params["seed"] == SEED
        assert isinstance(params["circuit_depth"], int) and params["circuit_depth"] > 0

    def test_the_recorded_qubit_count_is_the_width_the_circuit_actually_has(self, base_run):
        """The width is what the 36-qubit guard is checked against, so it must be right.

        The true width is ``d + qubits_per*n_obs + n_obs + qubits_per + 1``: one control
        register, one data register per training sample, one label qubit per training
        sample, one test-data register, one test-label qubit. For n_train=4, 2 features
        and d=2 that is 12.

        ``build_ensemble_circuit``'s own Notes section states
        ``d + 2*n_samples*log2(n_features) + n_samples + 1``, which gives 15 -- it
        doubles the per-sample data register instead of adding a single test register.
        The formula asserted here is the one the code implements.
        """
        columns, _ = base_run
        params = columns["results_QEnsemble"][0]["Model_Parameters"]
        qubits_per, n_obs, d = 1, 4, 2
        assert params["n_qubits"] == d + qubits_per * n_obs + n_obs + qubits_per + 1 == 12

    def test_it_reads_nothing_from_args_and_needs_no_grid_search_key(self, data):
        """Its configuration surface is its keywords, and ``args`` is inert.

        This test used to assert that ``grid_search`` was the *one* key read, which was
        true while ``modeleval`` branched on ``args["grid_search"]`` to choose between a
        ``Model_Parameters`` and a ``BestParams_Tuned`` column. That branch was the defect:
        a run-wide flag deciding a per-row column. It is gone -- whether a row's parameters
        came from a search is now decided per row, from an explicit ``tuned=`` argument or
        from the ``_opt`` suffix on the model label (``model_evaluation._was_tuned``) -- so
        ``modeleval`` reads nothing out of ``args`` and neither does ``compute_qensemble``.
        Two consequences, both asserted here:

        * The recorder sees no reads at all, which is the stronger form of the old claim.
          It is also the concrete reason this learner cannot be driven by ``model_run``,
          which passes nothing but ``args``: every dispatchable quantum learner takes
          ``backend``, ``shots`` and ``seed`` out of it, while this one takes ``n_shots``,
          ``seed`` and ``device`` as its own parameters. Setting ``args['shots']`` here
          still gets you 8192.
        * An ``args`` dict with no ``'grid_search'`` key is accepted. It used to raise
          ``KeyError`` inside ``modeleval`` *after* the fit, so a direct call that omitted
          a key it never needed paid for every circuit and then lost the row -- the reason
          the empty-dict half is asserted on the returned row and not just on the absence
          of an exception.
        """

        class RecordingArgs(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.read: list[str] = []

            def __getitem__(self, key):
                self.read.append(key)
                return super().__getitem__(key)

            def get(self, key, *default):
                self.read.append(key)
                return super().get(key, *default)

        X_train, X_test, y_train, y_test = data
        args = RecordingArgs({"grid_search": False})
        compute_qensemble(
            X_train, X_test[:2], y_train, y_test[:2], args, n_train=4, n_shots=16, seed=SEED
        )
        assert args.read == []

        empty = RecordingArgs()
        row = compute_qensemble(
            X_train, X_test[:2], y_train, y_test[:2], empty, n_train=4, n_shots=16, seed=SEED
        ).to_dict()["results_QEnsemble"][0]
        assert empty.read == []
        assert set(row) == set(DOCUMENTED_KEYS)
        assert row["Model_Parameters"]["seed"] == SEED

    def test_grid_search_true_does_not_relabel_the_untuned_parameters_as_tuned(self, data):
        """``grid_search: True`` must not make this row claim a search that never ran.

        It used to. ``modeleval`` chose the parameter column from
        ``args["grid_search"]``, a flag that describes the *run*, so every learner in a
        tuned run filed its parameters under ``BestParams_Tuned`` whether or not that
        learner had been tuned -- and this one never is: it has no ``_opt`` twin and no
        search space, so the numbers under that heading were the plain defaults from the
        keyword arguments. This test asserted that relabelling, which is why it had to
        change: the old expectation was the defect written down as the contract.

        The column is now decided per row, from the model label or an explicit ``tuned=``
        (``model_evaluation._was_tuned``). ``model`` here is ``'QEnsemble'``, which carries
        no ``_opt`` marker, so the row keeps ``Model_Parameters`` under either flag -- and a
        run that mixes a tuned ``dt_opt`` with this learner now writes both columns, each
        one truthful about its own row. Asserted because ``BestParams_Tuned`` is what
        ``qc_winner_finder`` and QuantumSage read as evidence a search happened.
        """
        X_train, X_test, y_train, y_test = data
        row = compute_qensemble(
            X_train,
            X_test[:4],
            y_train,
            y_test[:4],
            {"grid_search": True},
            n_train=4,
            n_shots=16,
            seed=SEED,
        ).to_dict()["results_QEnsemble"][0]

        assert "BestParams_Tuned" not in row
        assert set(row) == set(DOCUMENTED_KEYS)
        assert row["model"] == "QEnsemble"
        assert row["Model_Parameters"]["mode"] == "balanced"
        # The parameters under that heading are the untuned keywords, which is the whole
        # reason the tuned heading would have been a lie: nothing here searched for them.
        assert row["Model_Parameters"]["n_train"] == 4
        assert row["Model_Parameters"]["n_shots"] == 16


class TestTheAucIsTheEnsembleProbability:
    """``auc`` changed meaning package-wide; this learner is the one with no estimator.

    Every other learner's score comes from ``extract_binary_scores`` calling
    ``predict_proba`` or ``decision_function`` on a fitted object. There is no fitted
    object here at all: the ensemble measures a probability per test point directly, and
    ``retrieve_probabilities`` returns it as ``[p0, p1]``. So this is the one call site
    where a regression could not be caught by anything written about
    ``extract_binary_scores``.
    """

    def test_the_auc_is_the_roc_auc_of_the_probability_the_circuit_measured(self, base_run, data):
        columns, y_score = base_run
        _, _, _, y_test = data
        assert columns["results_QEnsemble"][0]["auc"] == pytest.approx(
            roc_auc_score(y_test, y_score)
        )

    def test_the_score_is_a_ranking_and_not_a_relabelling_of_the_predictions(self, base_run, data):
        """A balanced accuracy in disguise would be the label set itself.

        That is exactly the shape the old ``auc`` had -- ``roc_auc_score`` applied to hard
        predicted labels, a score array whose only values are 0 and 1. So the first half
        of this test says the recorded score is not that: eight measured frequencies,
        every one of them a multiple of ``1/n_shots`` and strictly inside the open
        interval, none of them 0 or 1.

        That half is not enough on its own, and this test used to lean on an assertion
        that was: "at least 3 distinct values among the 8". It held only because the shot
        sampling was unseeded -- it was counting noise. Seeded, the eight scores land on
        exactly two values, 29/64 and 44/64, one per class, so on this fixture the score
        array *is* numerically a function of the label and no count of distinct values can
        separate a ranking from a relabelling here. Raising ``n_shots`` would spread them
        (five distinct at 128, eight at 1024), but by resolving shot noise rather than
        geometry, which would be the same artefact with a bigger margin.

        The second half establishes the claim by geometry instead: nine test points sweeping
        the angle from the class-0 arc across the gap into the class-1 arc, whose overlaps
        with the training set genuinely differ. Points that share a predicted label come
        back with *different* scores there, and no function of the predicted label -- no
        relabelling, monotone or otherwise -- can do that. The sweep also runs up with the
        angle, which is the ranking itself.
        """
        _, y_score = base_run
        assert y_score.shape == (8,)
        # Strictly inside (0, 1), so the score cannot be the label set {0, 1}.
        assert np.all((y_score > 0.0) & (y_score < 1.0))
        # Multiples of 1/n_shots: the score is a measured frequency, which is also why
        # the ranking is coarse.
        np.testing.assert_allclose(y_score * SHOTS, np.round(y_score * SHOTS), atol=1e-9)

        X_train, _, y_train, _ = data
        # The training half's two classes occupy the arcs [0.10, 0.60] and [0.95, 1.45]
        # (see the `data` fixture), so this sweep starts inside class 0, crosses the empty
        # gap and ends inside class 1. 256 shots because the discrimination being asserted
        # is between probabilities ~0.06 apart, which the 1/64 grid of SHOTS cannot hold.
        sweep_angles = np.linspace(0.10, 1.45, 9)
        X_sweep = np.column_stack([np.cos(sweep_angles), np.sin(sweep_angles)])
        y_sweep = (sweep_angles > 0.775).astype(int)
        with captured_scores() as recorded:
            predicted = compute_qensemble(
                X_train, X_sweep, y_train, y_sweep, {"grid_search": False},
                n_train=4, d=2, n_shots=256, seed=SEED,
            ).to_dict()["y_predicted_QEnsemble"][0]
        sweep_score = recorded[0]

        assert sweep_score.shape == (9,)
        assert np.all((sweep_score > 0.0) & (sweep_score < 1.0))
        assert set(np.unique(predicted)) == {0, 1}
        for label in (0, 1):
            group = sweep_score[predicted == label]
            assert group.size >= 2
            assert len(set(group.tolist())) >= 2, (
                f"every point predicted {label} scored the same, so this score array "
                f"carries no more information than the predicted label: {sweep_score}"
            )
        # And it runs the right way: the far end of class 1's arc outranks the far end of
        # class 0's. A margin of 0.27 in the measured probabilities, against shot noise of
        # about 0.03 at 256 shots.
        assert sweep_score[0] < sweep_score[-1]

    def test_the_score_and_the_predictions_are_the_same_measurement(self, base_run):
        """``p1 > p0`` with ``p0 + p1 == 1`` is exactly ``p1 > 0.5``.

        This is the claim that the argmax and the AUC read the same quantity -- that the
        score was not obtained from some second, unrelated pass over the test set.
        """
        columns, y_score = base_run
        np.testing.assert_array_equal(
            columns["y_predicted_QEnsemble"][0], (y_score > 0.5).astype(int)
        )

    def test_a_single_class_test_set_records_auc_as_nan_and_keeps_the_rest(self, data):
        """No ROC curve exists, so NaN -- never a label-based number in its place."""
        X_train, X_test, y_train, _ = data
        row = compute_qensemble(
            X_train,
            X_test[:4],
            y_train,
            np.zeros(4, dtype=int),
            {"grid_search": False},
            n_train=4,
            n_shots=16,
            seed=SEED,
        ).to_dict()["results_QEnsemble"][0]

        assert np.isnan(row["auc"])
        assert np.isfinite(row["accuracy"])
        assert np.isfinite(row["f1_score"])


class TestTheCircuitWiringIsRight:
    """Exact assertions, made exact by degenerate inputs rather than by a tolerance."""

    @pytest.mark.parametrize("label", [0, 1])
    def test_the_swap_test_reads_back_the_training_label_when_test_equals_train(self, label):
        """``build_cosine_classifier`` documents ``P(0) = 1/2 + 1/2|<train|test>|^2``.

        With ``train == test`` the overlap is 1, so ``P(0) = 1`` before the label
        integration and the ``cx`` from ``y_train`` then flips the measured bit to the
        training label. Every shot must read that label -- a probability-one outcome, so
        no tolerance and no seed are involved, and an inverted ``cx``, a swap on the
        wrong register or a Hadamard in the wrong place all break it.

        This helper is not on any path ``compute_qensemble`` takes; it is a public
        function of the module with, until now, no caller in the tree at all.
        """
        vector = np.array(normalize_data(np.array([3.0, 4.0])))
        one_hot = label_to_array(np.array([label]))[0].astype(float)
        counts = execute_circuit(
            QENSEMBLE.build_cosine_classifier(vector, vector, one_hot), n_shots=SHOTS
        )
        assert counts == {str(label): SHOTS}

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("label", [0, 1])
    def test_the_ensemble_reads_back_a_unanimous_label_in_every_sampling_mode(self, mode, label):
        """Four identical training points, one shared label, test point equal to them.

        Every swap the ensemble applies then exchanges two identical states, so it is the
        identity whatever the mode picked, and the cosine slot sees overlap 1. The
        measured bit is the shared label with probability one, in all ``2**d`` branches
        at once. That pins the whole assembly end to end -- the ``train_qubit_map``
        indexing, the register order the ``initialize`` calls assume, the choice of
        ``n_obs - 1`` as the classification slot, and the final label ``cx`` -- for each
        of the three documented modes, deterministically.

        It is also the reason defect 2 is easy to miss: the label read-back on its own
        passes for ``mode="Balanced"`` too, because a circuit with no ensemble in it still
        classifies. Which is why the mode's own swap count is asserted alongside it --
        this test's name promises the read-back was checked in the mode it names, and
        without that count nothing here reads ``mode`` at all.
        """
        vector = normalize_data(np.array([3.0, 4.0]))
        X_data = np.array([vector] * 4)
        Y_data = np.array([label_to_array(np.array([label]))[0]] * 4, dtype=float)
        np.random.seed(3)  # fixes which pairs the mode swaps; the outcome does not depend on it
        circuit = QENSEMBLE.build_ensemble_circuit(X_data, Y_data, vector, n_swap=1, d=2, mode=mode)
        counts = execute_circuit(circuit, n_shots=SHOTS)

        assert counts == {str(label): SHOTS}
        assert retrieve_probabilities(counts)[label] == 1.0
        # The mode reached the builder. Without this the check is blind to its own
        # parameter: the measured bit is the shared label whichever strategy ran -- that
        # is the point of the degenerate input -- so all six non-``balanced``
        # parametrizations passed just as happily for a builder that ignored ``mode``
        # and ran ``balanced`` three times.
        assert _control_conditioned_swaps(circuit) == _expected_control_conditioned_swaps(
            mode, n_obs=4, n_swap=1, d=2
        )

    @pytest.mark.parametrize("d", [2, 3])
    @pytest.mark.parametrize("mode", MODES)
    def test_each_documented_mode_conditions_swaps_on_the_control_register(self, mode, d, data):
        """The ensemble *is* the control-conditioned swaps; a mode that emits none is not
        an ensemble, however plausible its results row looks.

        "``d`` control qubits, each in superposition, is ``2**d`` ensemble members" is
        three separate claims, and the two structural ones are exactly what an ensemble
        deleted in the way defect 2 deletes it still satisfies:

        * ``d`` Hadamards *on the control register*. Counted there and nowhere else: the
          final classification step puts two more on ``test_label``
          (compute_qensemble.py:337 and :343), so a whole-circuit count of Hadamards
          survives deleting the ensemble's own ``for i in range(d): qc.h(control[i])``
          loop -- the two swap-test Hadamards go on satisfying it by themselves.
        * every one of those ``d`` qubits conditions at least one operation. Superposition
          that nothing downstream reads recombines ``2**d`` identical branches.
        * as many conditioned swaps as this mode's own loops emit, so that the count is a
          statement about *this* strategy. ``> 0`` is not: it holds for any mode, and for
          any two of the three collapsed into the third.

        Repeated at ``d = 3`` because all three counts scale with ``d`` differently than
        they scale with anything else -- ``balanced`` uses ``d - 1`` control qubits in its
        loop and the last one only in its final swap, the other two use all ``d`` -- so a
        single width cannot tell the loop bound from a constant.
        """
        X_train, X_test, y_train, _ = data
        X_data, Y_data = prepare_training_set(X_train, y_train, n=4, seed=SEED)
        np.random.seed(3)
        circuit = QENSEMBLE.build_ensemble_circuit(
            X_data, Y_data, normalize_data(X_test[0]), n_swap=1, d=d, mode=mode
        )
        control = _control_register(circuit)

        assert control.size == d
        assert _hadamards_on(circuit, control) == d
        assert _conditioned_control_indices(circuit) == set(range(d))
        assert _control_conditioned_swaps(circuit) == _expected_control_conditioned_swaps(
            mode, n_obs=4, n_swap=1, d=d
        )

    def test_the_three_sampling_modes_are_three_different_strategies(self, data):
        """``mode`` selects between them, so no two of them may build the same circuit.

        Every other assertion about ``mode`` in this file is made one mode at a time, and
        each of those assertions holds for all three values -- so the two documented
        alternatives to ``balanced`` (the ``elif`` chain at compute_qensemble.py:285 and
        :305) could be deleted, their callers falling through to ``balanced``, with
        nothing here red. The strategies are cheap to tell apart, so this compares them
        against each other directly instead of against a bound each of them clears.

        The last two assertions are the other direction, and the reason the first two are
        not enough on their own: three circuits can be pairwise distinct and still all be
        the *empty* ensemble that defect 2 produces, differing only in the RNG draws that
        went unused. None of the three may look like that circuit.
        """
        X_train, X_test, y_train, _ = data
        X_data, Y_data = prepare_training_set(X_train, y_train, n=4, seed=SEED)
        test_point = normalize_data(X_test[0])

        def build(mode):
            # One identical RNG stream per build, so every difference between the
            # circuits is the mode's own doing and not the draws it happened to make.
            np.random.seed(3)
            return QENSEMBLE.build_ensemble_circuit(
                X_data, Y_data, test_point, n_swap=1, d=2, mode=mode
            )

        circuits = [build(mode) for mode in MODES]
        assert len({_control_conditioned_swaps(c) for c in circuits}) == len(MODES)
        assert len({_gate_structure(c) for c in circuits}) == len(MODES)

        no_ensemble = build("Balanced")  # the misspelling TestAnUnrecognised... pins
        assert _control_conditioned_swaps(no_ensemble) == 0
        assert all(_gate_structure(c) != _gate_structure(no_ensemble) for c in circuits)


class TestTheRandomUnitaryMethodIsReachable:
    """The second, "advanced" ensemble construction the docstring advertises.

    Reachable, and on ``mode="balanced"`` a real ensemble -- but only there. ``mode`` is
    documented as a parameter of its own, independent of ``ensemble_method``, and two of
    its three documented values build nothing on this branch (defect 6, pinned at the
    end of this class). So "reachable" is asserted about the circuit that runs rather
    than about the row that comes back: the row is identical whichever of the three was
    asked for.
    """

    def test_a_random_unitary_ensemble_fits_at_the_one_size_that_is_tractable(self, data):
        """``n_train=2`` is not a stylistic choice, it is the only usable size.

        The Haar unitary is drawn on ``2**(qubits_per*n_obs + n_obs)`` dimensions and
        then given a control qubit, so ``n_train=4`` on 2 features asks the transpiler to
        synthesise a 9-qubit unitary: over 100 s per test point, versus 0.2 s at
        ``n_train=2``. Nothing in the signature says so, and ``n_train`` defaults to 4.

        ``n_train=2`` is reachable *only* on this branch -- the fixed-swap balanced mode
        needs two distinct samples inside one class and raises for it (see
        ``TestTheFailureModesAreReported``).
        """
        X_train, X_test, y_train, y_test = data
        with captured_circuits() as circuits:
            columns = compute_qensemble(
                X_train,
                X_test[:2],
                y_train,
                y_test[:2],
                {"grid_search": False},
                n_train=2,
                d=2,
                n_shots=16,
                seed=SEED,
                ensemble_method="random_unitary",
            ).to_dict()
        row = columns["results_QEnsemble"][0]

        assert set(row) == set(DOCUMENTED_KEYS)
        assert row["Model_Parameters"]["ensemble_method"] == "random_unitary"

        # The row is not evidence that this branch did anything: `ensemble_method` is
        # echoed straight back from the keyword, and a circuit with no ensemble in it
        # still returns an accuracy, a width and a depth. So assert about the circuits
        # that ran -- one per test point -- and about the branch's own gate, the
        # controlled Haar unitary, which nothing on the fixed-swap path emits.
        assert len(circuits) == 2
        for circuit in circuits:
            assert sum(1 for i in circuit.data if i.operation.name == "c-unitary") == 2
            # Two controlled unitaries on control[0] plus the final data/label exchange
            # on control[1]: 2**d branches over d control qubits, each of them read.
            assert _control_conditioned_operations(circuit) == 4
            assert _hadamards_on(circuit, _control_register(circuit)) == 2
            assert _conditioned_control_indices(circuit) == {0, 1}

        # `n_qubits` and `circuit_depth` describe the last circuit built, so they are
        # checkable against it rather than against a range. 2 control + 2 data + 2 labels
        # + 1 test data + 1 test label is 8; today the width is 9, because of the dead
        # pred_qubit that test_the_random_unitary_branch_is_no_wider_than_the_swap_branch
        # pins. Both are accepted so that removing it does not turn this test red -- the
        # exact number is that pin's job -- but the row must report whichever it is.
        assert row["Model_Parameters"]["n_qubits"] == circuits[-1].num_qubits
        assert row["Model_Parameters"]["circuit_depth"] == circuits[-1].depth()
        assert row["Model_Parameters"]["n_qubits"] in (8, 9)
        # Not `0.0 <= accuracy <= 1.0`, which accuracy_score guarantees whatever happened:
        # the row's metric has to be the metric of the predictions stored beside it. Two
        # test points cannot support a floor above chance, so this is the whole of what
        # the numbers on this branch can be held to.
        assert row["accuracy"] == pytest.approx(
            accuracy_score(y_test[:2], columns["y_predicted_QEnsemble"][0])
        )

    def test_the_default_ensemble_method_is_the_fixed_swap_branch(self, data):
        """Which of the two branches an unset ``ensemble_method`` takes, and its width.

        Worth pinning because the branch is chosen by ``if ensemble_method ==
        "random_unitary" ... else``, so *everything* that is not that one string lands on
        the swaps -- the documented default, but also any typo, since ``Literal`` is a
        hint and hints do not run. Unlike a typo in ``mode`` no behaviour is lost, and the
        width is the tell either way: the swap branch allocates no ``pred_qubit``, so a
        row recording ``ensemble_method='random_unitry'`` at 12 qubits was a fixed-swap
        run wearing the wrong label. (Verified separately; not asserted here, because
        rejecting the typo would be a fix and a fix must not turn this red.)
        """
        X_train, X_test, y_train, _ = data
        X_data, Y_data = prepare_training_set(X_train, y_train, n=4, seed=SEED)
        test_point = normalize_data(X_test[0])

        np.random.seed(3)
        default = QENSEMBLE.build_ensemble_circuit(X_data, Y_data, test_point, n_swap=1, d=2)
        np.random.seed(3)
        swap = QENSEMBLE.build_ensemble_circuit(
            X_data, Y_data, test_point, n_swap=1, d=2, ensemble_method="swap"
        )
        assert _gate_structure(default) == _gate_structure(swap)
        assert default.num_qubits == 12
        # Non-vacuity, exact rather than `> 0`: two structurally identical circuits also
        # compare equal when both are the empty ensemble, so this says which circuit the
        # equality above was established on -- the default `mode` is `balanced`.
        assert _control_conditioned_swaps(default) == _expected_control_conditioned_swaps(
            "balanced", n_obs=4, n_swap=1, d=2
        )


    @pytest.mark.xfail(
        strict=True,
        reason="qbiocode/learning/compute_qensemble.py:172-176 allocates a 'pred_qubit' "
        "register on the random_unitary branch that no instruction ever touches, so that "
        "branch is one qubit wider than the swap branch for nothing -- in a simulation "
        "whose binding constraint is width (see the guard at line 444).",
    )
    def test_the_random_unitary_branch_is_no_wider_than_the_swap_branch(self, data):
        """No qubit of this circuit should be one no instruction ever touches."""
        X_train, X_test, y_train, _ = data
        X_data, Y_data = prepare_training_set(X_train, y_train, n=2, seed=SEED)
        test_point = normalize_data(X_test[0])

        np.random.seed(3)
        random_unitary = QENSEMBLE.build_ensemble_circuit(
            X_data, Y_data, test_point, n_swap=1, d=2, ensemble_method="random_unitary"
        )
        idle = [
            index
            for index in range(random_unitary.num_qubits)
            if not any(
                random_unitary.find_bit(bit).index == index
                for instruction in random_unitary.data
                for bit in instruction.qubits
            )
        ]
        assert idle == []


    @pytest.mark.parametrize("mode", [m for m in MODES if m != "balanced"])
    @pytest.mark.xfail(
        strict=True,
        reason="qbiocode/learning/compute_qensemble.py:203 opens the random_unitary "
        "branch's mode handling with `if mode == 'balanced':` and never adds an elif or "
        "an else, so the other two documented values of `mode` append no controlled "
        "unitary and no controlled swap at all. Verified at n_train=2, d=2: 4 "
        "control-conditioned operations and depth 9 for 'balanced' against 0 and depth 5 "
        "for either of these -- gate for gate the same circuit as the empty ensemble "
        "TestAnUnrecognisedSamplingModeIsAcceptedSilently pins for the misspelling "
        "'Balanced'. Through the public API the row reports acc 0.5, n_qubits 9 and a "
        "Model_Parameters echoing the mode it was asked for.",
    )
    def test_the_random_unitary_branch_honours_every_documented_sampling_mode(self, data, mode):
        """``mode`` and ``ensemble_method`` are documented as independent parameters.

        Both docstrings list the three sampling strategies under ``mode`` with no
        qualification, and neither says the choice is confined to one construction
        method. On this branch two of the three are silently inert, which is defect 2
        reached without misspelling anything -- and unlike defect 2 there is no wrong
        input to blame, so no amount of care at the call site avoids it.

        Asserted on the circuit rather than through ``compute_qensemble``, because the
        row a broken mode returns is indistinguishable from a working one; that is the
        whole difficulty. ``n_train=2`` for the reason
        ``test_a_random_unitary_ensemble_fits_at_the_one_size_that_is_tractable``
        explains: at 4 the Haar unitary is a 9-qubit synthesis.
        """
        X_train, X_test, y_train, _ = data
        X_data, Y_data = prepare_training_set(X_train, y_train, n=2, seed=SEED)
        np.random.seed(3)
        circuit = QENSEMBLE.build_ensemble_circuit(
            X_data,
            Y_data,
            normalize_data(X_test[0]),
            n_swap=1,
            d=2,
            mode=mode,
            ensemble_method="random_unitary",
        )
        assert _control_conditioned_operations(circuit) > 0
        assert _conditioned_control_indices(circuit) == {0, 1}


class TestTheFailureModesAreReported:
    """Where a misconfiguration does raise, it should raise before spending anything."""

    def test_a_feature_count_that_is_not_a_power_of_two_is_rejected(self, data):
        """``qubits_per = int(log2(n_features))`` truncates, so 3 features asks qiskit to
        initialize 3 amplitudes on 1 qubit.

        Rejection is the right outcome and it is what happens, which is the point worth
        pinning: amplitude encoding needs a power-of-two feature count, that constraint
        lives only in a parameter description, and the truncating ``int()`` that reaches
        the failure is silent.

        The refusal and the phrase naming the cause are asserted; the exception class is
        left open. Today the refusal comes out of qiskit's ``initialize`` as a
        ``QiskitError`` that never mentions ``n_features``, and replacing that with an
        up-front ``ValueError`` naming the feature count would be a fix, not a
        regression.
        """
        _, _, y_train, y_test = data
        rng = np.random.default_rng(0)
        X = rng.uniform(0.5, 1.5, size=(24, 3))
        with pytest.raises((QiskitError, ValueError), match="power of 2"):
            compute_qensemble(
                X[:16], X[16:18], y_train, y_test[:2], {"grid_search": False},
                n_train=4, n_shots=8, seed=SEED,
            )

    @pytest.mark.parametrize(
        "n_train",
        [
            pytest.param(40, id="more_samples_than_a_class_holds"),
            pytest.param(2, id="fewer_than_the_two_per_class_a_balanced_swap_needs"),
        ],
    )
    def test_an_impossible_training_subset_size_is_rejected(self, data, n_train):
        """Both ends of ``n_train`` are unusable on the default fixed-swap balanced path,
        and neither bound is documented: 40 exceeds the 8 rows each class has, and 2
        leaves one sample per class where the balanced mode needs to draw two distinct
        ones from the same class. Only the exception type is asserted -- both messages
        come from numpy and improving either is a fix, not a regression."""
        X_train, X_test, y_train, y_test = data
        with pytest.raises(ValueError):
            compute_qensemble(
                X_train, X_test[:1], y_train, y_test[:1], {"grid_search": False},
                n_train=n_train, d=2, n_shots=8, seed=SEED,
            )

    def test_a_circuit_wider_than_the_hardcoded_limit_is_rejected_with_its_width_named(self):
        """The guard's own contract: refuse before simulating, and say how wide.

        ``n_train=18`` on 2 features gives 40 qubits, which no statevector simulator will
        take. The message has to carry the number, because the only lever the user has is
        ``n_train`` and nothing else in the traceback relates the two.

        The limit itself is deliberately left out of the match: the pin below asks for it
        to be read off the backend instead of hardcoded, and 40 qubits exceeds either
        number, so that fix must not turn this test red.
        """
        angles = np.linspace(0.10, 1.45, 40)
        X = np.column_stack([np.cos(angles), np.sin(angles)])
        y = np.zeros(40, dtype=int)
        y[1::2] = 1
        with pytest.raises(ValueError, match=r"Circuit has 40 qubits.*exceeds simulation limit"):
            compute_qensemble(
                X, X[:1], y, y[:1], {"grid_search": False},
                n_train=18, d=2, n_shots=8, seed=SEED,
            )

    @pytest.mark.skipif(
        AER_MAX_QUBITS >= 36,
        reason=f"this machine's Aer statevector limit is {AER_MAX_QUBITS} qubits, so the "
        f"guard's 36 is not above it and there is no gap to pin",
    )
    @pytest.mark.xfail(
        strict=True,
        reason="qbiocode/learning/compute_qensemble.py:444 hardcodes a 36-qubit limit, "
        "but AerSimulator derives its own (32 here) from available memory. For any width "
        "in between, the guard passes and the transpiler raises CircuitTooWideForTarget "
        "instead -- an error naming a coupling map the caller never configured, with no "
        "mention of n_train. The limit should be read off the backend.",
    )
    def test_the_width_guard_fires_before_the_simulator_refuses_the_circuit(self, data):
        X_train, X_test, y_train, y_test = data
        # n_train=16 on 2 features is exactly 36 qubits: allowed by `> 36`, refused by Aer.
        with pytest.raises(ValueError, match="exceeds simulation limit"):
            compute_qensemble(
                X_train, X_test[:1], y_train, y_test[:1], {"grid_search": False},
                n_train=16, d=2, n_shots=8, seed=SEED,
            )


class TestTheSeedFixesTheResult:
    """``seed`` reaches the circuit *and* the shot sampling. It used to stop at the circuit.

    The two halves are asserted separately and in this order, because they used to have
    different answers: the construction was reproducible under ``seed`` while the
    measurement was not, and separating them is what said which half was broken.
    """

    def test_the_seed_fixes_the_training_subset_and_every_gate(self, data):
        """The half of the reproducibility story that always worked.

        ``prepare_training_set`` seeds the global RNG, and every swap pair
        ``build_ensemble_circuit`` draws comes off that stream -- so two calls at the same
        seed build the identical sequence of circuits, one per test point. Asserted
        separately from the metrics so that the probabilities test below is unambiguously
        about the shot sampling and not about the construction; while the sampling was
        unseeded, this test passing next to that one failing was the whole diagnosis.
        """
        X_train, X_test, y_train, y_test = data

        first = prepare_training_set(X_train, y_train, n=4, seed=SEED)
        second = prepare_training_set(X_train, y_train, n=4, seed=SEED)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

        structures: list[list] = []
        original = QENSEMBLE.execute_circuit

        # `seed` included and forwarded, for the reason `captured_circuits` gives.
        def spy(circuit, n_shots=8192, device="CPU", seed=None):
            structures[-1].append(_gate_structure(circuit))
            return original(circuit, n_shots=n_shots, device=device, seed=seed)

        QENSEMBLE.execute_circuit = spy
        try:
            for _ in range(2):
                structures.append([])
                compute_qensemble(
                    X_train, X_test[:3], y_train, y_test[:3], {"grid_search": False},
                    n_train=4, d=2, n_shots=8, seed=SEED,
                )
        finally:
            QENSEMBLE.execute_circuit = original

        assert len(structures[0]) == 3
        assert structures[0] == structures[1]

    def test_three_identical_calls_at_the_same_seed_measure_the_same_probabilities(self, data):
        """The half that was broken, and the regression this file exists to prevent twice.

        The bug: ``qbiocode/utils/qutils.py``'s ``execute_circuit`` called
        ``backend.run(...)`` with no ``seed_simulator``, so the shot sampling came off OS
        entropy. ``seed`` fixed the training subset and every gate and reached no further,
        while ``compute_qensemble`` documented it as "Random seed for reproducibility" and
        recorded it in ``Model_Parameters``: sixty identical calls at the same seed on this
        one 16/8 split returned sixty different score arrays, with accuracy landing anywhere
        from 0.5 to 1.0, and nothing in the row said why. Every other learner in the package
        was reproducible under its seed.

        Fixed: ``execute_circuit`` takes ``seed`` and passes it as ``seed_simulator``, and
        ``compute_qensemble`` forwards its own. Guarded here on the score arrays rather
        than the predictions -- eight hard predictions can coincide between two runs by
        chance about once in 250, eight probabilities quantised to 1/64 cannot: no pair
        among the 1770 drawn from those sixty unseeded runs was equal. Three calls, all
        required to agree.

        The second half asserts that the ``seed`` is what fixes it, rather than a constant
        stream having been hardcoded in place of the entropy -- which satisfies the first
        half exactly. It is asserted on ``execute_circuit`` directly, on one fixed circuit,
        and that is deliberate: two ``compute_qensemble`` calls at different seeds *do*
        return different probabilities, but that says nothing about the sampling, because
        ``seed`` also chooses the training subset and the second call therefore measured
        different circuits. Verified -- an ``execute_circuit`` patched to pin
        ``seed_simulator`` to a constant and ignore its argument passes this class if the
        comparison is made through the learner. The circuit used here is deliberately *not*
        one of this file's degenerate probability-one circuits, whose counts are the same
        under every seed: ``|<v1|v2>|^2`` is 1/2, so ``P(1)`` is 3/4 and the shot noise is
        real.

        The seeds are far apart, which is load-bearing. Aer behaves as if each shot's RNG
        were derived from ``seed_simulator`` plus the shot index, so two seeds closer
        together than ``n_shots`` sample overlapping streams: seeds 1 and 2 returned 51 and
        52 of one outcome out of 64 shots, seeds 1 and 65 returned 51 and 45. A pair a
        thousand apart is independent; a pair one apart would have made this assertion a
        coin flip.
        """
        X_train, X_test, y_train, y_test = data
        with captured_scores() as recorded:
            for _ in range(3):
                compute_qensemble(
                    X_train, X_test, y_train, y_test, {"grid_search": False},
                    n_train=4, d=2, n_shots=SHOTS, seed=SEED,
                )
        np.testing.assert_array_equal(recorded[0], recorded[1])
        np.testing.assert_array_equal(recorded[0], recorded[2])

        vector = np.array(normalize_data(np.array([1.0, 0.0])))
        other = np.array(normalize_data(np.array([1.0, 1.0])))
        one_hot = label_to_array(np.array([1]))[0].astype(float)

        def sample(seed):
            return execute_circuit(
                QENSEMBLE.build_cosine_classifier(vector, other, one_hot),
                n_shots=SHOTS,
                seed=seed,
            )

        assert sample(SEED) == sample(SEED), "one circuit, one seed, two different results"
        distinct = {
            tuple(sorted(sample(s).items()))
            for s in (SEED, SEED + 1000, SEED + 2000, SEED + 3000)
        }
        assert len(distinct) >= 2, (
            f"four well-separated seeds sampled one circuit identically ({distinct}), so "
            f"the sampling is fixed by something other than the seed"
        )


class TestTheRecordedParametersDescribeTheRunThatHappened:
    """``Model_Parameters`` is the only surviving account of how a row was produced.

    A results file is read long after the run, by ``qc_winner_finder`` and QuantumSage
    and by whoever is trying to reproduce a number. A parameter that records the request
    rather than what the code did with it is worse than a missing one.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="qbiocode/utils/data_encoding.py:121 takes int(n/2) per class, so an odd "
        "n_train silently trains on n-1 samples, while "
        "qbiocode/learning/compute_qensemble.py:465 records the requested n_train. A row "
        "reading n_train=5 describes a run that used 4; the width tells the truth and the "
        "parameter column does not.",
    )
    def test_the_recorded_training_size_is_the_number_of_samples_actually_used(self, data):
        X_train, X_test, y_train, y_test = data
        params = compute_qensemble(
            X_train, X_test[:1], y_train, y_test[:1], {"grid_search": False},
            n_train=5, d=2, n_shots=8, seed=SEED,
        ).to_dict()["results_QEnsemble"][0]["Model_Parameters"]

        # 2 control + n_obs data + n_obs labels + 1 test data + 1 test label, at one
        # qubit per sample: the width is an honest read-out of n_obs.
        n_obs = (params["n_qubits"] - 2 - 1 - 1) // 2
        assert n_obs == 4  # the run really did use four samples
        assert params["n_train"] == n_obs


class TestAnUnrecognisedSamplingModeIsAcceptedSilently:
    """The defect with no visible symptom: a working-looking row with no ensemble in it.

    ``TestTheCircuitWiringIsRight`` shows why nothing catches this from the outside -- a
    circuit with the ensemble stripped out still classifies, still reports a plausible
    accuracy, and still records the mode it was asked for.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="qbiocode/learning/compute_qensemble.py:236 opens an if/elif chain over "
        "mode with no else, so any unrecognised value -- 'Balanced' included -- skips "
        "every swap. Verified: the circuit for mode='Balanced' has 0 swaps conditioned on "
        "a control qubit against 10 for 'balanced', depth 5 against 14, so all 2**d "
        "branches are one plain cosine classifier. Model_Parameters faithfully records "
        "the misspelling and the row is indistinguishable from a real ensemble run.",
    )
    def test_an_unrecognised_sampling_mode_is_rejected(self, data):
        X_train, X_test, y_train, y_test = data
        with pytest.raises(ValueError, match="mode"):
            compute_qensemble(
                X_train, X_test[:2], y_train, y_test[:2], {"grid_search": False},
                n_train=4, d=2, n_shots=8, seed=SEED, mode="Balanced",
            )


class TestTheExportSurfaceTreatsTheEnsembleConsistently:
    """It is in ``qbiocode.learning.__all__``, in neither ``qbiocode.__all__`` nor the
    dispatch table. That is consistent, not an oversight -- and this is the invariant
    that says so.

    The quantum ``_opt`` twins were a genuine inconsistency because the criterion the
    package root uses is *dispatchability*: ``qbiocode/__init__.py`` says so in as many
    words, and six functions reachable through ``compute_ml_dict`` were importable only
    by their private module path. ``compute_qensemble`` is not dispatchable and cannot
    be made so without a signature change -- ``model_run`` passes nothing but ``args``,
    and this learner reads no configuration from ``args`` at all -- nothing whatsoever
    (``test_it_reads_nothing_from_args_and_needs_no_grid_search_key``). Its absence from
    the root is therefore the same single decision as its absence from the table.

    tests/test_model_contract_matrix.py already asserts the forward direction, that every
    dispatchable learner is exported from both places. The converse -- that the root
    exports nothing *but* dispatchable learners -- was unasserted, and it is the half
    that decides this question.
    """

    def test_the_documented_import_path_reaches_the_function(self):
        """``from qbiocode.learning import compute_qensemble`` is what its own docstring,
        and the subpackage's Usage section, tell users to write."""
        assert "compute_qensemble" in learning.__all__
        assert learning.compute_qensemble is compute_qensemble
        assert callable(compute_qensemble)

    def test_the_learner_names_at_the_package_root_are_exactly_the_dispatchable_ones(self):
        """Both directions at once, so either kind of drift fails here.

        A learner added to ``compute_ml_dict`` and not to the root is unreachable for
        every example in the docs; a learner added to the root and not to the table
        advertises a model ``args['model']`` will reject by name.
        """
        root_learners = {
            name
            for name in qbiocode.__all__
            if name.startswith("compute_") and name in learning.__all__
        }
        assert root_learners == {f"compute_{key}" for key in DISPATCH_KEYS}

    def test_the_ensemble_is_at_the_root_exactly_when_it_is_in_the_dispatch_table(self):
        """Stated on the ensemble itself, so the reason survives a refactor of the sets.

        Today both sides are False. Wiring it into ``compute_ml_dict`` would make both
        True and keep this green; adding it to ``qbiocode.__all__`` alone -- the change
        that would look like tidying up an oversight -- turns it red.
        """
        assert ("compute_qensemble" in qbiocode.__all__) == ("qensemble" in DISPATCH_KEYS)
        assert hasattr(qbiocode, "compute_qensemble") == ("qensemble" in DISPATCH_KEYS)
