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

"""``model_run``'s remaining edges: the failures that arrive late, or blame the wrong thing.

Nothing here is about a metric. Every test is about *where* a mistake in the config
surfaces, because ``model_run`` fans its work out through joblib and folds it back up
with ``pd.melt(...).dropna().pivot(...)``, and both ends of that turn a one-word
config error into a message about something else entirely -- minutes later, on a real
dataset, after every model has been fitted and thrown away. The validation block at
the top of ``model_run`` exists to convert exactly that shape into an immediate
ValueError naming the key at fault. These are the cases it did not yet cover.

**A duplicated name.** ``model: ['dt', 'dt']`` used to fit both models and then die
with ``ValueError: Index contains duplicate entries, cannot reshape`` -- a pandas
message, raised by the ``pivot`` two hundred and fifty lines below the config error
that caused it, naming neither ``dt`` nor ``args['model']``. Each entry writes the same
three columns (``results_dt``, ``y_test_dt``, ``y_predicted_dt``), so the pivot has two
values for one index; there is no arrangement of the results in which a repeated name means
anything, so the config is simply wrong and can say so before the first fit. That
check is now part of the same validation block, and the first class below pins both
the message and the claim that nothing runs before it -- a guard that merely moved the
failure earlier in the *same* function would be worth much less than one that fires
before joblib is touched at all.

**A typo'd key in a ``<model>_args`` block.** Still ``TypeError: compute_dt() got an
unexpected keyword argument 'n_estimators'``: the name of an internal function the
user never wrote, instead of the ``dt_args`` block they did. This is easy to reach by
copying a block from one model to another, because the classical signatures expose
different subsets of hyperparameters -- ``n_estimators`` is real for ``rf``, ``xgb``
and ``tabpfn`` and meaningless for the other six, ``catboost`` spells the same idea
``iterations``, and ``compute_nb`` accepts ``var_smoothing`` and nothing else. It is
pinned here, not fixed -- tests/test_classical_models.py already holds the assertion
that the offending keyword at least appears in the message; what is added here is the
blame. The better error is a ValueError naming both the block and the key, e.g. ``"dt_args has no hyperparameter 'n_estimators' (did you copy it from
rf_args?); compute_dt accepts: ccp_alpha, class_weight, criterion, ..."``, and the
machinery for it is already present: ``_seeded_kwargs`` inspects
``inspect.signature(compute_fn).parameters`` to decide whether to fill in
``random_state``, so the same call could diff the configured keys against that
signature and raise beside the other config errors instead of at call time.

**A direct ``compute_<model>()`` call.** ``modeleval`` used to read
``args["grid_search"]`` as a bare subscript to choose between the ``Model_Parameters``
and ``BestParams_Tuned`` column names, so a direct call whose ``args`` dict lacked that
key raised ``KeyError('grid_search')`` *after* the fit -- at the reporting step, which
is the worst possible time to learn the dict was incomplete. That is fixed. The column
is now decided per row, by ``modeleval``'s own ``tuned`` parameter (stated by the
caller, or inferred from the ``_opt`` suffix on the row's label), and ``args`` is read
nowhere in the function at all; it survives as a parameter only because 24 call sites
pass it positionally. So the xfail became an assertion -- and one made over *every*
untuned entry point in the dispatch table rather than over ``dt`` alone, because the
key was read by ``modeleval``, which all fourteen of them end in. What each of the
fourteen genuinely needs in ``args`` was measured rather than assumed: the nine
classical functions read nothing out of it, while the five quantum ones read
``'backend'`` and ``'seed'`` -- and ``'shots'`` for the three that ask for a sampler
primitive -- before they fit anything. That difference is asserted separately, so those
keys cannot be mistaken for a surviving ``grid_search`` requirement. The direct route is
public (``qbiocode.learning.compute_dt`` is exported) and is what a notebook uses. It
was not wholly untested -- tests/test_catboost_tabpfn.py calls ``compute_catboost`` and
``compute_tabpfn`` by hand throughout -- but every such call passes the same
``{"seed": 42, "grid_search": False}`` literal, so the missing key had been exercised
nowhere, and none of the seven ``OneVsOneClassifier``-wrapped learners was called
directly at all. Also covered: what the call returns, that a hand-passed
``random_state`` lands where the dispatcher puts ``args['seed']``, and that ``args``
cannot influence the fit -- now in the stronger form that a ``grid_search: True``
sitting in it does not even move the parameter column. The seeding half is read off the
recorded
``estimator__random_state`` rather than off the metrics, because ``compute_dt`` on this
particular split scores the same under every seed -- including ``None`` -- so comparing
its numbers would have proved the fit identical without saying anything about the seed
that produced it.

**The shipped configs.** ``qbiocode/apps/*/configs/*.yaml`` are what a new user
copies, and a stale name in one is not a syntax error -- it is a ValueError from
``model_run`` on someone else's machine. Every name in ``model:`` and every
``<model>_args`` / ``gridsearch_<model>_args`` block is checked against the dispatch
table read out of ``model_run`` itself, so renaming a learner without updating the
configs fails here. ``_opt`` twins are checked for separately: they are dispatch keys,
so ``model: ['dt_opt']`` passes the unknown-name check and then fails either way --
``TypeError: compute_dt_opt() got an unexpected keyword argument 'data_key'`` with
``grid_search`` off, and "no '_opt' implementation" (it looks for ``dt_opt_opt``) with
it on. They are selected by ``grid_search``, never by name -- and the
``grid_search``-off case is pinned as an xfail, since the block already explains the
convention in its unknown-name message and simply never reaches a name it recognises.

**The returned keys.** The docstring used to promise "keys as model names". They are
``results_<label>``, ``y_test_<label>`` and ``y_predicted_<label>``, each mapping to a
one-entry dict keyed by the integer ``0``, and ``<label>`` gains an ``_opt`` suffix
when the model was tuned -- so the documented ``result['dt']`` is a KeyError and the
documented ``result['results_dt']['accuracy']`` is another one. The docstring is fixed
and the real shape asserted, including against the docstring text, so the two cannot
drift apart again.

**The joblib fan-out.** ``args['n_jobs'] > 1`` runs the models in loky *subprocesses*,
where neither ``np.random.seed`` nor ``algorithm_globals.random_seed`` from the parent
survives -- which is why ``_call_with_global_seeds`` exists. That a fanned-out run
agrees with a sequential one is already asserted, over all nine classical learners, by
tests/test_model_contract_matrix.py. What no test showed is that the fan-out *happened*:
a run that had quietly fallen back to sequential execution would satisfy every one of
those comparisons, because they would then be comparing a sequential run with itself.
So the last class here proves the boundary was crossed, by running one extra task
through the same ``Parallel`` object and reading the process id back out of it, and
pins the ``min(n_jobs, len(model))`` clamp that decides how many workers there are.
"""

from __future__ import annotations

import inspect
import os
import pathlib
import re
import sys

import joblib
import numpy as np
import pytest
import yaml
from joblib import delayed
from sklearn.model_selection import train_test_split

# Imported before anything torch-backed can be: qbiocode's __init__ installs
# xgboost's OpenMP runtime first, which keeps a later `import torch` from killing the
# interpreter. See tests/test_openmp_import_order.py.
import qbiocode
from qbiocode import learning, scale_train_test
from qbiocode.evaluation.model_run import model_run

# `qbiocode/evaluation/__init__.py` does `from .model_run import model_run`, so the
# attribute `qbiocode.evaluation.model_run` is the FUNCTION, not the module. Patching
# `Parallel` on the result of `from qbiocode.evaluation import model_run` therefore
# sets an attribute on a function object and changes nothing -- a monkeypatch that
# silently no-ops, which would leave the fan-out tests below asserting on a spy that
# was never installed. sys.modules is the only unambiguous handle on the module.
import qbiocode.evaluation.model_run  # noqa: F401  (registers the module)

MODEL_RUN_MODULE = sys.modules["qbiocode.evaluation.model_run"]

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_RUN_SOURCE = REPO_ROOT / "qbiocode" / "evaluation" / "model_run.py"

#: Inner keys of an untuned metrics row.
RESULT_KEYS = frozenset(
    {"model", "accuracy", "f1_score", "time", "auc", "Model_Parameters"}
)

#: One column per model label, three labels' worth of prefixes.
LABEL_PREFIXES = ("results", "y_test", "y_predicted")


def _dispatch_keys():
    """Every name ``args['model']`` may hold, read out of ``model_run``'s own table.

    ``compute_ml_dict`` is built from lazy imports inside the function body, so there
    is no importable object to inspect. Restating the list here would let this file
    keep passing after a learner was renamed, which is precisely the drift the config
    tests below exist to catch; the same reader is used in test_docs_structure.py.
    """
    source = MODEL_RUN_SOURCE.read_text(encoding="utf-8")
    block = source.split("compute_ml_dict = {")[1].split("}")[0]
    keys = re.findall(r'"([a-z_0-9]+)":\s*compute_', block)
    assert len(keys) > 1, f"could not read compute_ml_dict out of {MODEL_RUN_SOURCE}"
    return frozenset(keys)


DISPATCH_KEYS = _dispatch_keys()

#: Names a config's ``model:`` list may legitimately hold: the dispatch table minus
#: the tuned twins, which are selected by ``grid_search`` and never by name.
SELECTABLE = frozenset(k for k in DISPATCH_KEYS if not k.endswith("_opt"))


# ----------------------------------------------------------------------------------
# Data: small, seeded, and separable enough that every learner produces a real score
# ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_split():
    """A 60x5 binary problem, split identically for every test in this file."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = (X[:, 0] + 0.4 * X[:, 1] > 0).astype(int)
    return train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)


def _classical_args(models, **extra):
    """The minimal ``args`` dict for an untuned classical run."""
    return {"model": list(models), "seed": 7, "n_jobs": 1, "grid_search": False, **extra}


def _comparable_parameters(row):
    """A metrics row's ``Model_Parameters`` with the estimator object dropped.

    Everything left is a plain value that compares by value -- including
    ``'estimator__random_state'`` and ``'estimator__n_estimators'``, the two this file
    reads to tell what the estimator was actually built with. ``'estimator'`` itself is
    an unfitted scikit-learn instance, and those compare by identity, so leaving it in
    would make any two rows unequal.
    """
    return {
        key: value
        for key, value in row["Model_Parameters"].items()
        if key != "estimator"
    }


def _result_rows(frame):
    """Every metrics row in a compute function's frame, keyed by the label it is filed under.

    One entry for thirteen of the fourteen learners. ``compute_qpl`` reports six -- one
    classical head per quantum projection -- and reports them *stacked*: six rows, each
    holding its own head's triple of columns and NaN under the other five heads', every
    one of them on index 0. That is why ``model_run`` folds results up with
    ``melt(...).dropna().pivot(...)`` instead of reading row 0, and dropping the nulls
    column by column is the same operation on one frame. Reading the labels off the frame
    rather than expecting one per learner is what lets a single parametrized test cover
    both shapes without a table of expected names that would have to be right about
    ``qpl``'s six heads as well.
    """
    rows = {}
    for column in frame.columns:
        if not column.startswith("results_"):
            continue
        filled = frame[column].dropna()
        assert len(filled) == 1, (
            f"{column} holds {len(filled)} non-null metrics rows; a compute function "
            f"reports exactly one row per label"
        )
        rows[column[len("results_"):]] = filled.iloc[0]
    return rows


class _ExplodingParallel:
    """Stand-in for ``joblib.Parallel`` that refuses to be built.

    Used to prove a validation error fires *before* any dispatch happens. Not a mock
    of a model: nothing is fitted on this path at all, which is the whole claim.
    """

    def __init__(self, *args, **kwargs):  # pragma: no cover - the point is not reaching here
        raise AssertionError(
            "model_run constructed joblib.Parallel despite an invalid args['model']; "
            "the config error should have been raised before any dispatch"
        )


class _SpyParallel(joblib.Parallel):
    """The real ``Parallel``, plus a record of how it was configured and where it ran.

    Subclassing rather than mocking: the models are still fitted for real, by real
    joblib, in whatever processes joblib chose. One extra task -- ``os.getpid`` -- is
    appended to the batch and stripped from the results before ``model_run`` sees
    them, so the pid comes back from a worker that ran alongside the actual fits. That
    is the only way to tell a genuine loky fan-out from a silent fallback to
    sequential execution, and without it a test that passes ``n_jobs=3`` is asserting
    nothing about the parallel path.
    """

    calls: list[dict] = []

    def __call__(self, iterable):
        tasks = list(iterable)
        tasks.append(delayed(os.getpid)())
        results = super().__call__(tasks)
        type(self).calls.append({"n_jobs": self.n_jobs, "worker_pid": results[-1]})
        return results[:-1]


# ----------------------------------------------------------------------------------
# A duplicated model name
# ----------------------------------------------------------------------------------


class TestADuplicatedModelNameIsRejectedBeforeAnyFit:
    """``model: ['dt', 'dt']``: a config error that used to arrive as a pandas one.

    The old failure was ``ValueError: Index contains duplicate entries, cannot
    reshape``, raised by ``pivot`` after both fits had completed. On the tiny data here
    that is instant; on a real dataset with ``catboost`` and ``qsvc`` in the list it is
    the end of a long run.
    """

    def test_a_repeated_model_name_is_rejected_by_name(self, tiny_split):
        X_train, X_test, y_train, y_test = tiny_split
        with pytest.raises(ValueError) as excinfo:
            model_run(
                X_train, X_test, y_train, y_test, "dup", _classical_args(["dt", "dt"])
            )
        message = str(excinfo.value)
        # What the pandas message lacked: the offending model name, and the config key
        # it came from. 'duplicate' is checked too, but note it is not discriminating on
        # its own -- "Index contains duplicate entries" contains the word as well, which
        # is exactly why the other three assertions are here.
        assert "dt" in message
        assert "args['model']" in message
        assert "duplicate" in message.lower()
        assert "cannot reshape" not in message

    def test_only_the_repeated_name_is_reported(self, tiny_split):
        """A long list should name the repeat, not everything in it.

        ``rf`` and ``nb`` are named once each and are not at fault; a message listing
        the whole ``model:`` list would send the reader looking in the wrong place.
        """
        X_train, X_test, y_train, y_test = tiny_split
        with pytest.raises(ValueError) as excinfo:
            model_run(
                X_train,
                X_test,
                y_train,
                y_test,
                "dup",
                _classical_args(["rf", "nb", "rf"]),
            )
        message = str(excinfo.value)
        # The list of duplicates is exactly ['rf'] -- printed as a literal, so this
        # also pins that 'nb' is not accused alongside it.
        assert "['rf']" in message

    def test_nothing_is_dispatched_before_the_duplicate_is_caught(
        self, tiny_split, monkeypatch
    ):
        """The wasted work, not just the wrong message, is what made this expensive.

        Raising from the fold-up at the bottom of ``model_run`` would still produce a
        clear ValueError while having fitted everything first. Replacing
        ``joblib.Parallel`` with something that cannot be constructed is the direct
        test of "before any dispatch": if the check ever moves back below the
        dispatch, this fails with the AssertionError from ``_ExplodingParallel``
        rather than the expected ValueError.
        """
        X_train, X_test, y_train, y_test = tiny_split
        monkeypatch.setattr(MODEL_RUN_MODULE, "Parallel", _ExplodingParallel)
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            model_run(
                X_train, X_test, y_train, y_test, "dup", _classical_args(["dt", "dt"])
            )

    def test_a_duplicate_is_caught_when_grid_search_is_on_too(self, tiny_split, monkeypatch):
        """The tuned branch writes ``results_dt_opt`` twice and collides identically.

        The check sits above the ``grid_search`` branch on purpose, so this is a pin
        on placement: a duplicate guard added inside the untuned branch would leave a
        tuned config failing in the pivot after the whole search had run.
        """
        X_train, X_test, y_train, y_test = tiny_split
        monkeypatch.setattr(MODEL_RUN_MODULE, "Parallel", _ExplodingParallel)
        args = _classical_args(
            ["dt", "dt"],
            grid_search=True,
            n_trials=2,
            cross_validation=2,
            gridsearch_dt_args={"criterion": ["gini", "entropy"], "max_depth": [2, 3]},
        )
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            model_run(X_train, X_test, y_train, y_test, "dup", args)

    def test_an_unknown_name_is_still_blamed_on_being_unknown(self, tiny_split):
        """A name that is both unknown and repeated is an unknown name first.

        "Unknown model(s) ['svm']" is actionable; "Duplicate model(s) ['svm']" would
        send the reader to fix the wrong thing, and fixing the duplicate would not
        make the config run.
        """
        X_train, X_test, y_train, y_test = tiny_split
        with pytest.raises(ValueError, match="Unknown model"):
            model_run(
                X_train, X_test, y_train, y_test, "dup", _classical_args(["svm", "svm"])
            )

    def test_a_list_with_no_repeats_still_runs(self, tiny_split):
        """The guard must not reject the ordinary case it sits in front of.

        A duplicate check written against the wrong collection -- comparing a name to
        the whole table, or counting ``_opt`` twins -- would refuse every config. Two
        distinct models, two full sets of columns.
        """
        X_train, X_test, y_train, y_test = tiny_split
        result = model_run(
            X_train, X_test, y_train, y_test, "pair", _classical_args(["dt", "nb"])
        )
        assert set(result) == {
            f"{prefix}_{model}" for prefix in LABEL_PREFIXES for model in ("dt", "nb")
        }


# ----------------------------------------------------------------------------------
# A typo'd key in a <model>_args block
# ----------------------------------------------------------------------------------


class TestATypoInAModelArgsBlockIsBlamedOnAnInternalFunction:
    """``dt_args: {n_estimators: 100}``: valid YAML, valid for ``rf``, fatal for ``dt``.

    Pinned as it stands, per the module docstring. The better error is a ValueError
    naming ``dt_args`` and ``n_estimators`` and listing what ``compute_dt`` does
    accept, raised in the validation block beside the other config errors rather than
    at call time inside a joblib task.

    tests/test_classical_models.py's
    ``test_a_key_the_learner_does_not_take_names_the_offending_keyword`` already pins
    the one useful part of today's message -- that the offending keyword appears in it
    -- so that is not repeated here. What this class adds is the xfail on the blame
    itself, and the structural premise underneath it: the differing signatures are read
    off ``inspect.signature`` rather than asserted in a docstring, so a hyperparameter
    added to one of these functions updates the premise instead of quietly invalidating
    it.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "qbiocode/evaluation/model_run.py:335 -- _seeded_kwargs inspects "
            "compute_fn's signature for 'random_state' but never diffs the configured "
            "keys against it, so a key that block does not accept reaches the call and "
            "surfaces as TypeError(\"compute_dt() got an unexpected keyword argument "
            "'n_estimators'\"), naming an internal function instead of the 'dt_args' "
            "block the user wrote"
        ),
    )
    def test_the_error_should_name_the_config_block_that_holds_the_key(self, tiny_split):
        """What the user needs to read: the block to edit, not the callee's name.

        ``compute_dt`` appears in no config file and in no documentation a user of
        ``qbiocode-profiler`` reads; ``dt_args`` is a line they wrote themselves.
        """
        X_train, X_test, y_train, y_test = tiny_split
        args = _classical_args(["dt"], dt_args={"n_estimators": 100})
        with pytest.raises(ValueError, match=r"dt_args"):
            model_run(X_train, X_test, y_train, y_test, "typo", args)

    def test_the_same_key_is_accepted_by_the_model_it_was_copied_from(self, tiny_split):
        """``n_estimators`` is not a nonsense word -- it is ``rf``'s, in ``dt``'s block.

        Without this, the xfail above could be pinning nothing more interesting than a
        malformed key. It is not malformed: the identical block runs cleanly one model
        over, which is exactly why the mistake gets made, and why the message needs to
        say which block it read the key from.

        "Runs cleanly" has to be read off the forest that was built, not off the run
        having finished. ``_model_args`` looks the block up by name and falls back to the
        estimator defaults *with a log line* when it is absent, so a run that never saw
        ``rf_args`` at all finishes just as cleanly and reports the same ``model``
        label -- which makes the label worthless as evidence that the block was
        accepted. ``estimator__n_estimators`` is the evidence: 5 when the block is
        passed and scikit-learn's default 100 when it is not. Both runs are made, so an
        ``<model>_args`` channel that silently dropped its blocks fails the first
        assertion, and a fallback default that stopped being 100 fails the second rather
        than quietly making the first one vacuous.
        """
        X_train, X_test, y_train, y_test = tiny_split
        args = _classical_args(["rf"], rf_args={"n_estimators": 5})
        configured = model_run(X_train, X_test, y_train, y_test, "copied", args)[
            "results_rf"
        ][0]
        assert configured["model"] == "rf"
        assert configured["Model_Parameters"]["estimator__n_estimators"] == 5, (
            "rf_args={'n_estimators': 5} did not reach the forest; the block was "
            "accepted in name only"
        )
        default = model_run(
            X_train, X_test, y_train, y_test, "default", _classical_args(["rf"])
        )["results_rf"][0]
        assert default["Model_Parameters"]["estimator__n_estimators"] == 100, (
            "the no-block default is no longer 100, so the assertion above no longer "
            "separates a block that was read from one that was ignored"
        )

    @pytest.mark.parametrize(
        "name, takes_n_estimators",
        [
            ("rf", True),
            ("xgb", True),
            ("tabpfn", True),
            ("dt", False),
            ("nb", False),
            ("lr", False),
            ("svc", False),
            ("mlp", False),
            ("catboost", False),
        ],
    )
    def test_the_classical_signatures_expose_different_hyperparameters(
        self, name, takes_n_estimators
    ):
        """The structural reason a copied block is a live hazard, not a hypothetical.

        Three of the nine classical learners take ``n_estimators`` and six do not --
        ``catboost`` spells the same idea ``iterations`` -- so a block copied between
        any two of them has a three-in-nine chance of being accepted and otherwise
        raises. Read off the signatures rather than asserted from memory, so a
        hyperparameter added to one of these functions updates the premise instead of
        silently invalidating it.
        """
        parameters = inspect.signature(getattr(learning, f"compute_{name}")).parameters
        assert ("n_estimators" in parameters) is takes_n_estimators

    def test_naive_bayes_accepts_one_hyperparameter_and_nothing_else(self):
        """The narrowest signature, and so the easiest one to over-fill.

        ``compute_nb`` takes ``var_smoothing`` alone; every other key in any other
        block is fatal to it. Anything added here should be a deliberate widening of
        the naive-Bayes config surface, not an accident.
        """
        parameters = inspect.signature(learning.compute_nb).parameters
        plumbing = {"X_train", "X_test", "y_train", "y_test", "args", "verbose",
                    "model", "data_key"}
        assert set(parameters) - plumbing == {"var_smoothing"}


# ----------------------------------------------------------------------------------
# A direct compute_<model>() call
# ----------------------------------------------------------------------------------

#: What a direct ``compute_<name>()`` call genuinely needs, one entry per untuned entry
#: point in the dispatch table: the ``args`` keys that learner reads *for itself*, and
#: the keywords a caller must pass instead of letting the dispatcher fill them in.
#:
#: Measured learner by learner rather than assumed, because an ``args`` key added on a
#: hunch is the easiest way to make
#: ``test_a_direct_call_does_not_need_a_grid_search_key`` pass while asserting less than
#: it claims. The nine classical functions read *nothing* out of ``args`` -- they hand it
#: to ``modeleval`` untouched -- so their dicts are empty; ``nb`` is the one that takes no
#: ``random_state``, GaussianNB having none, which is also why ``_seeded_kwargs``
#: inspects the signature instead of filling that keyword in unconditionally. The five
#: quantum functions call ``qutils.get_backend_session(args, ...)`` before they build a
#: circuit, which needs ``'backend'`` and ``'seed'``, plus ``'shots'`` for the three that
#: ask for a sampler primitive; ``pqk`` and ``qpl`` also cache their projections under a
#: directory named in ``args``, pointed at ``tmp_path`` below so that no run writes into
#: the repository or is served a projection computed for other data. Not one of the
#: fourteen needs ``'grid_search'``, which is the point; the tests after the parametrized
#: one keep that claim honest from both ends -- the table's own contents, and where the
#: quantum keys are actually read.
DIRECT_CALL_REQUIREMENTS = {
    "dt": ((), {"random_state": 7}),
    "lr": ((), {"random_state": 7}),
    "mlp": ((), {"random_state": 7}),
    "nb": ((), {}),
    "rf": ((), {"random_state": 7}),
    "svc": ((), {"random_state": 7}),
    "xgb": ((), {"random_state": 7}),
    "catboost": ((), {"random_state": 7}),
    # One tree and the CPU, per conftest's own TabPFN probe: the default 'auto' fits an
    # ensemble, and this node is about the reporting step rather than the score.
    "tabpfn": ((), {"random_state": 7, "n_estimators": 1, "device": "cpu"}),
    "qsvc": (("backend", "shots", "seed"), {}),
    "vqc": (("backend", "shots", "seed"), {}),
    "qnn": (("backend", "shots", "seed"), {}),
    "pqk": (("backend", "seed", "pqk_projection_dir"), {}),
    "qpl": (("backend", "seed", "qpl_projection_dir"), {}),
}

#: The learners that read ``args`` for themselves -- and so the ones that need a real
#: backend dict and the two-qubit fixture rather than ``tiny_split``.
QUANTUM_DIRECT = tuple(
    name for name, (keys, _) in DIRECT_CALL_REQUIREMENTS.items() if "backend" in keys
)


@pytest.fixture(scope="module")
def quantum_split():
    """18/6 rows on two features, MinMax-scaled on the training rows only.

    ``tiny_split`` cannot serve the five quantum entry points: they encode feature
    magnitudes as rotation angles, so unscaled columns wrap past 2*pi, and they build an
    n-by-n fidelity kernel or a per-row projection by simulating circuits, which on 42
    training rows and five qubits is slow for no gain here. Two features means two
    qubits. Nothing in this file asserts on a quantum *score*, so this fixture only has
    to be a fit those functions accept -- it follows the protocol in
    tests/test_quantum_models.py, scaler fitted on the training rows alone, so the two
    files do not disagree about what a reasonable quantum fixture looks like.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test = scale_train_test(X[:18], X[18:], scaling="MinMaxScaler")
    return X_train, X_test, y[:18], y[18:]


@pytest.fixture
def minimal_args(tmp_path):
    """``args`` holding exactly what one learner reads for itself, and not one key more.

    Function-scoped, for ``tmp_path``: ``pqk``'s projection cache key covers the feature
    map but not the row count, so a directory shared between nodes could hand one node a
    projection computed for another's data.
    """
    values = {
        "backend": "simulator",
        "shots": 64,
        "seed": 7,
        "pqk_projection_dir": str(tmp_path / "pqk"),
        "qpl_projection_dir": str(tmp_path / "qpl"),
    }

    def build(name):
        keys, _ = DIRECT_CALL_REQUIREMENTS[name]
        return {key: values[key] for key in keys}

    return build


@pytest.fixture
def direct_call(tiny_split, quantum_split, minimal_args, request):
    """Call ``compute_<name>`` by hand the way a notebook does, and hand back its frame.

    ``args`` defaults to the minimal dict for that learner; pass one to vary it. The
    split follows the learner, and the keywords come from
    ``DIRECT_CALL_REQUIREMENTS``. ``model=name`` is passed explicitly because the
    untuned default is a display name -- ``compute_dt``'s is ``'Decision Tree'`` -- and
    the columns are named after whatever label arrives, so a caller that omits it gets
    ``results_Decision Tree``.

    ``tabpfn`` is the one learner whose fit can be genuinely unavailable in a supported
    install: it lives in the ``[tabpfn]`` extra and downloads a checkpoint on first use.
    Its node is gated on conftest's ``tabpfn_ready`` probe, requested here rather than
    declared as a parameter so the other thirteen nodes do not pay for the probe.
    """

    def call(name, args=None):
        if name == "tabpfn":
            request.getfixturevalue("tabpfn_ready")
        keys, kwargs = DIRECT_CALL_REQUIREMENTS[name]
        X_train, X_test, y_train, y_test = (
            quantum_split if name in QUANTUM_DIRECT else tiny_split
        )
        compute = getattr(learning, f"compute_{name}")
        return compute(
            X_train,
            X_test,
            y_train,
            y_test,
            minimal_args(name) if args is None else args,
            model=name,
            **kwargs,
        )

    return call


class TestADirectComputeCallIsNotCoveredByTheDispatcher:
    """``compute_dt(...)`` called by hand, as a notebook does.

    Two gaps, not one whole missing route. tests/test_catboost_tabpfn.py does call
    ``compute_catboost`` and ``compute_tabpfn`` directly, seventeen times -- but
    always with the module-level ``UNTUNED = {"seed": 42, "grid_search": False}``, so no
    test in the suite omitted the key ``modeleval`` used to subscript unguarded, and none
    calls one of the seven ``OneVsOneClassifier``-wrapped learners directly.
    ``compute_dt`` stands in for those seven in the shape and seeding tests: it is the
    cheapest of them and the one whose ``random_state`` handling the dispatcher is known
    to depend on. The ``grid_search`` test does not stand in for anything -- it is
    parametrized over all fourteen untuned entry points, because the key it is about was
    read by ``modeleval``, which every one of them ends in.

    The dispatcher's seeding is why the seed itself is followed through in this class
    rather than in one of its own: the claim that a direct call and a dispatched one are
    the same fit only means something if ``args['seed']`` and a hand-passed
    ``random_state`` reach the estimator as the same value, and ``dt`` --
    seed-insensitive on this split -- is the one learner here that cannot demonstrate it,
    so ``mlp`` does.
    """

    def test_a_direct_call_returns_the_frame_the_dispatcher_folds_up(self, tiny_split):
        """One row, three columns, and the metrics row the dispatcher hands on.

        This is the shape ``model_run`` concatenates and pivots, so it pins the
        contract at the seam between a compute function and the dispatcher, from the
        side no other test looks at.
        """
        X_train, X_test, y_train, y_test = tiny_split
        frame = learning.compute_dt(
            X_train, X_test, y_train, y_test, {"grid_search": False},
            model="dt", random_state=7,
        )
        assert list(frame.columns) == ["y_test_dt", "y_predicted_dt", "results_dt"]
        assert len(frame) == 1
        assert set(frame["results_dt"].iloc[0]) == RESULT_KEYS

    def test_a_direct_call_agrees_with_the_dispatched_one(self, tiny_split):
        """The dispatcher adds seeding and fan-out, not a different fit.

        ``model_run`` fills ``random_state`` in from ``args['seed']`` via
        ``_seeded_kwargs``; passing the same value by hand must land in the same place.
        If it did not, a notebook and a profiler run would disagree on the same data
        and seed with nothing to explain the gap.

        The three metrics cannot show that on their own, and used to be all this
        compared. ``compute_dt`` on this exact split is seed-*insensitive*: twenty-five
        unseeded fits produce one single triple, (0.722222, 0.714286, 0.722222) -- the
        same one ``random_state=7`` produces -- because the ties it breaks at random all
        break to the same test-set predictions here. Their equality therefore held for
        every value the dispatcher could have passed, ``None`` included, and said
        nothing about the seeding it names. ``estimator__random_state`` is what
        discriminates: it is the seed the estimator was really built with, read back out
        of the row. The rest of ``Model_Parameters`` is compared beside it, so a
        divergence in any other hyperparameter -- a default the dispatcher fills in and
        a direct caller does not -- fails here too.
        """
        X_train, X_test, y_train, y_test = tiny_split
        direct = learning.compute_dt(
            X_train, X_test, y_train, y_test, {"grid_search": False},
            model="dt", random_state=7,
        )["results_dt"].iloc[0]
        dispatched = model_run(
            X_train, X_test, y_train, y_test, "direct", _classical_args(["dt"])
        )["results_dt"][0]
        assert dispatched["Model_Parameters"]["estimator__random_state"] == 7, (
            "args['seed'] = 7 did not arrive at the estimator as random_state=7, so "
            "the dispatched fit is not the one the direct call reproduces"
        )
        assert _comparable_parameters(dispatched) == _comparable_parameters(direct)
        for metric in ("accuracy", "f1_score", "auc"):
            assert direct[metric] == pytest.approx(dispatched[metric], nan_ok=True)

    def test_the_seed_the_dispatcher_is_given_is_the_seed_the_estimator_is_built_with(
        self, tiny_split
    ):
        """The mechanism the agreement above rests on, on a learner it is visible on.

        ``_seeded_kwargs`` fills ``random_state`` from ``args['seed']`` for every
        compute function whose signature takes one, and no part of a result reports the
        value it used except ``Model_Parameters``. So the seed is followed through here
        for four different values, and the metrics are collected alongside to show the
        threading is load-bearing rather than decorative.

        ``mlp`` rather than ``dt``: the weight init is drawn from ``random_state``, so
        these four seeds do not all score the same on this split, whereas ``dt`` ties
        identically under every one of them and could not tell a working
        ``_seeded_kwargs`` from one that dropped the seed. The metric assertion is on
        the *number of distinct* triples, not on any particular value, so a scikit-learn
        release that shifts the scores keeps it meaningful.
        """
        X_train, X_test, y_train, y_test = tiny_split
        triples = set()
        for seed in (0, 7, 8, 42):
            row = model_run(
                X_train, X_test, y_train, y_test, "seeded",
                _classical_args(["mlp"], seed=seed),
            )["results_mlp"][0]
            assert row["Model_Parameters"]["estimator__random_state"] == seed, (
                f"args['seed'] = {seed} reached the estimator as "
                f"{row['Model_Parameters']['estimator__random_state']!r}"
            )
            triples.add(tuple(row[metric] for metric in ("accuracy", "f1_score", "auc")))
        assert len(triples) > 1, (
            f"mlp scored identically under all four seeds ({triples}), so this file no "
            f"longer holds a case where the seed changes the fit; move this test to a "
            f"learner or a split where it does"
        )

    def test_args_cannot_influence_the_fit_of_a_direct_call(self, tiny_split):
        """``args`` was a reporting channel here, and is no longer even that.

        Everything the estimator needs arrives as an explicit keyword. ``args`` was read
        for one thing only -- the name of the parameter column -- and only at the very
        end; that decision has since moved to ``modeleval``'s ``tuned`` parameter, so on
        this path the dict is now read for nothing at all. Filling it with values that
        would change a *dispatched* run (a different model list, a different seed, a
        different ``n_jobs``) changes nothing about the numbers, and that is the
        measurement behind the claim that the ``KeyError`` this dict used to raise fired
        after all the work and for no reason connected to the work.
        """
        X_train, X_test, y_train, y_test = tiny_split
        minimal = learning.compute_dt(
            X_train, X_test, y_train, y_test, {"grid_search": False},
            model="dt", random_state=7,
        )["results_dt"].iloc[0]
        noisy = learning.compute_dt(
            X_train, X_test, y_train, y_test,
            {"grid_search": False, "model": ["nonsense"], "seed": 999, "n_jobs": 8},
            model="dt", random_state=7,
        )["results_dt"].iloc[0]
        for metric in ("accuracy", "f1_score", "auc"):
            assert minimal[metric] == pytest.approx(noisy[metric], nan_ok=True)
        # The recorded hyperparameters too, so a difference in how the estimator was
        # *built* is caught and not just one in how it scored.
        assert _comparable_parameters(minimal) == _comparable_parameters(noisy)

    @pytest.mark.parametrize("name", sorted(DIRECT_CALL_REQUIREMENTS))
    def test_a_direct_call_does_not_need_a_grid_search_key(self, name, direct_call):
        """An untuned function must not need telling that tuning is off. It no longer does.

        **The bug.** ``modeleval`` chose between the ``Model_Parameters`` and
        ``BestParams_Tuned`` column names by subscripting ``args["grid_search"]``, so a
        direct ``compute_<model>()`` call whose ``args`` dict lacked the key raised
        ``KeyError('grid_search')`` -- at the reporting step, after the estimator had been
        fitted and scored, over a key that could not have been anything but False:
        ``compute_dt`` *is* the untuned entry point, ``compute_dt_opt`` is the tuned one.
        A notebook paid for the whole fit and got an exception about its config instead of
        a result.

        **The fix.** The column is decided per row rather than per run: ``modeleval``
        takes a ``tuned`` argument, infers it from the ``_opt`` suffix on the row's own
        label when none is given, and reads ``args`` nowhere at all.

        **What this guards now.** That an ``args`` dict carrying none of the reporting
        step's business -- here exactly what the learner itself reads and not one key more
        -- still returns a complete untuned row: the six ``RESULT_KEYS``,
        ``Model_Parameters`` among them and ``BestParams_Tuned`` not, filed under the
        label the caller asked for. A regression in either direction fails here: the old
        subscript raises, and a ``tuned`` default that guessed True would swap the
        parameter column.

        Parametrized over all fourteen untuned entry points rather than the ``dt`` the pin
        covered, because the key was read in ``modeleval`` -- shared by every learner in
        the package -- so one caller's worth of evidence understated both where the defect
        lived and where a regression would land. Each learner is called with the ``args``
        keys it genuinely reads and no others, measured rather than assumed; see
        ``DIRECT_CALL_REQUIREMENTS``, where the classical nine need an empty dict and the
        quantum five need a backend, a seed and (for the sampler models) a shot count of
        their own.
        """
        rows = _result_rows(direct_call(name))
        assert rows, f"compute_{name} returned no 'results_' column at all"
        for label, row in rows.items():
            assert label == name or label.startswith(f"{name}_"), (
                f"compute_{name}(model={name!r}) filed a row under {label!r}; the label "
                f"the caller passed is what names the columns"
            )
            assert set(row) == RESULT_KEYS, (
                f"compute_{name} reported {sorted(row)} under {label!r}; an untuned row "
                f"carries {sorted(RESULT_KEYS)}. A missing key means the reporting step "
                f"did not complete; 'BestParams_Tuned' means it took an untuned fit for "
                f"a tuned one"
            )
            assert row["model"] == label

    def test_every_untuned_entry_point_is_covered_and_none_is_handed_the_key(self):
        """Non-vacuity, and the only guard on the requirements table itself.

        Two ways the parametrization above could quietly stop meaning anything. A learner
        added to ``compute_ml_dict`` and not to the table would go on never being called
        directly by any test, so the set is compared against ``SELECTABLE``, which is read
        out of ``model_run``'s own source and cannot drift from it. And a ``'grid_search'``
        key added to any one case's ``args`` would turn that node green while asserting the
        opposite of what its name says -- the table is the one place that could happen
        without anybody noticing.
        """
        assert set(DIRECT_CALL_REQUIREMENTS) == SELECTABLE, (
            f"DIRECT_CALL_REQUIREMENTS and model_run's dispatch table disagree: "
            f"{sorted(SELECTABLE - set(DIRECT_CALL_REQUIREMENTS))} are dispatchable and "
            f"untested by hand, "
            f"{sorted(set(DIRECT_CALL_REQUIREMENTS) - SELECTABLE)} are tested and not "
            f"dispatchable"
        )
        smuggled = {
            name: keys
            for name, (keys, _) in DIRECT_CALL_REQUIREMENTS.items()
            if "grid_search" in keys
        }
        assert not smuggled, (
            f"{smuggled} would be handed the very key "
            f"test_a_direct_call_does_not_need_a_grid_search_key exists to do without"
        )

    @pytest.mark.parametrize("name", QUANTUM_DIRECT)
    def test_a_quantum_entry_point_blames_itself_for_the_key_it_needs(
        self, name, direct_call, minimal_args
    ):
        """The five quantum ``args`` keys are theirs, and are missed before any fit.

        Without this, the parametrization above would be open to the reading that ``qsvc``
        and its four siblings still need a fuller ``args`` dict for the *reporting* step
        and that ``'grid_search'`` merely happened to fall out of it. They do not:
        ``qutils.get_backend_session`` reads ``'backend'`` to choose a primitive and
        ``'seed'`` to seed it -- ``'shots'`` too, for a sampler -- and it names whichever
        is missing, before anything is fitted, let alone reported. Dropping
        ``'seed'`` is the cheapest demonstration of that: the ValueError names the key and
        there is no fit to pay for.
        """
        args = minimal_args(name)
        del args["seed"]
        with pytest.raises(ValueError, match=r"seed"):
            direct_call(name, args)

    @pytest.mark.parametrize("name", ["dt", "qsvc"])
    def test_a_grid_search_flag_in_args_does_not_move_the_parameter_column(
        self, name, direct_call, minimal_args
    ):
        """The other half of the same fix: the key is not merely optional, it is unread.

        ``args['grid_search']`` was a run-wide flag deciding a per-row column, which is
        how ``'BestParams_Tuned'`` came to head the parameters of rows that had never been
        searched. Handing ``grid_search: True`` to an untuned entry point is that mistake
        in miniature -- these parameters are the estimator's defaults plus whatever the
        caller passed, tuned by nothing -- so the column must stay ``Model_Parameters``
        however the dict is filled in. Read together with the parametrized test above, the
        two say the key is ignored whether it is absent or present and wrong.

        One classical learner and one quantum one: the decision is ``modeleval``'s and
        neither family reaches it differently. ``compute_qpl``, the only learner that
        states ``tuned`` rather than leaving it to be inferred, computes it from its own
        label and so cannot read the flag either.
        """
        args = {**minimal_args(name), "grid_search": True}
        rows = _result_rows(direct_call(name, args))
        assert rows, f"compute_{name} returned no 'results_' column at all"
        for label, row in rows.items():
            assert set(row) == RESULT_KEYS, (
                f"compute_{name} reported {sorted(row)} under {label!r} for an untuned "
                f"fit with grid_search=True in args; the flag decided the parameter "
                f"column again"
            )


# ----------------------------------------------------------------------------------
# A tuned twin named directly in args['model']
# ----------------------------------------------------------------------------------


class TestNamingATunedTwinDirectlyFailsLateToo:
    """``model: ['dt_opt']``: an unknown-name check that lets the name through.

    Found while checking the shipped configs, and the same shape as the two above:
    ``dt_opt`` *is* a key of ``compute_ml_dict``, so the unknown-name guard passes it,
    and the run then fails for a reason that mentions neither the name nor the
    convention. It is a plausible mistake precisely because the twins are real,
    documented functions -- the docstrings for ``compute_dt_opt`` describe them -- and
    nothing in ``model:`` says they are not selectable there.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "qbiocode/evaluation/model_run.py:201 -- the validation block rejects "
            "names that are absent from compute_ml_dict, but the '_opt' twins are IN "
            "it, so model: ['dt_opt'] with grid_search off reaches dispatch and dies "
            "with TypeError(\"compute_dt_opt() got an unexpected keyword argument "
            "'data_key'\"); the same block already explains the convention in its "
            "unknown-name message and could enforce it"
        ),
    )
    def test_an_opt_twin_in_the_model_list_is_rejected_by_the_validation_block(
        self, tiny_split
    ):
        """The message already exists -- it is just not reached.

        The unknown-name ValueError ends with "Note the '_opt' variants are selected
        with args['grid_search'], not by naming them here", which is exactly what this
        user needs to read. A name ending in '_opt' should get it.
        """
        X_train, X_test, y_train, y_test = tiny_split
        with pytest.raises(ValueError, match=r"_opt"):
            model_run(
                X_train, X_test, y_train, y_test, "twin", _classical_args(["dt_opt"])
            )

    def test_an_opt_twin_with_grid_search_on_is_at_least_rejected_before_any_fit(
        self, tiny_split, monkeypatch
    ):
        """The other half is already sound, and worth holding still.

        With ``grid_search`` on, ``model_run`` looks for ``dt_opt_opt``, does not find
        it, and raises before dispatch. The message is aimed at the wrong problem --
        it reads as though ``dt_opt`` merely lacks a tuned implementation -- but it
        does name ``_opt``, and it costs nothing. Matched loosely on that substring so
        this stays green under any fix to the case above.
        """
        X_train, X_test, y_train, y_test = tiny_split
        monkeypatch.setattr(MODEL_RUN_MODULE, "Parallel", _ExplodingParallel)
        args = _classical_args(
            ["dt_opt"], grid_search=True, n_trials=2, cross_validation=2
        )
        with pytest.raises(ValueError, match=r"_opt"):
            model_run(X_train, X_test, y_train, y_test, "twin", args)


# ----------------------------------------------------------------------------------
# The shipped config files
# ----------------------------------------------------------------------------------


def _shipped_configs():
    """``(relative path, parsed mapping)`` for every config that ships in the wheel."""
    found = []
    for path in sorted(REPO_ROOT.glob("qbiocode/apps/*/configs/*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            found.append((str(path.relative_to(REPO_ROOT)), loaded))
    return found


SHIPPED_CONFIGS = _shipped_configs()

#: Only the configs that drive model_run have a ``model:`` list; quvine's does not.
CONFIGS_WITH_MODELS = [
    (name, config) for name, config in SHIPPED_CONFIGS if config.get("model")
]


def _hyperparameter_blocks(config):
    """``<model>_args`` and ``gridsearch_<model>_args`` keys of one parsed config."""
    return [key for key in config if isinstance(key, str) and key.endswith("_args")]


#: Only the configs that actually carry hyperparameter blocks. Parametrising the block
#: check over every shipped config instead adds a node for quvine's config.yaml, which
#: holds no ``_args`` key at all: the loop body would never run, the ``bad`` dict would
#: stay empty and the node would pass whatever the dispatch table said -- a green tick
#: for a file the check cannot say anything about.
CONFIGS_WITH_BLOCKS = [
    (name, config) for name, config in SHIPPED_CONFIGS if _hyperparameter_blocks(config)
]


class TestTheShippedConfigsNameOnlyRealModels:
    """A stale name in a shipped config is a ValueError on a new user's first run.

    These files are the documented starting point -- ``qbiocode-profiler`` reads one
    by default and the tutorials copy it -- so they are the one place where a name
    drifting out of the dispatch table is guaranteed to reach somebody.
    """

    def test_the_glob_actually_found_the_profiler_config(self):
        """Non-vacuity: an empty parametrisation would make this class a no-op.

        Everything below is driven by a filesystem glob, so a moved directory would
        turn each test into a skip on an empty parameter list -- green, and checking
        nothing. This names what must be there.
        """
        names = [name for name, _ in SHIPPED_CONFIGS]
        assert "qbiocode/apps/qprofiler/configs/config.yaml" in names
        assert CONFIGS_WITH_MODELS, (
            f"no shipped config has a 'model:' list; found {names}"
        )
        assert CONFIGS_WITH_BLOCKS, (
            f"no shipped config has a '<model>_args' block, so "
            f"test_every_hyperparameter_block_names_a_real_model is parametrised over "
            f"nothing; found {names}"
        )

    @pytest.mark.parametrize(
        "name, config", CONFIGS_WITH_MODELS, ids=[n for n, _ in CONFIGS_WITH_MODELS]
    )
    def test_every_model_named_in_a_config_is_dispatchable(self, name, config):
        unknown = sorted(set(config["model"]) - DISPATCH_KEYS)
        assert not unknown, (
            f"{name} names {unknown} in its 'model:' list, which model_run's "
            f"compute_ml_dict does not contain; a run from this config dies in "
            f"validation. Valid names: {sorted(SELECTABLE)}"
        )

    @pytest.mark.parametrize(
        "name, config", CONFIGS_WITH_MODELS, ids=[n for n, _ in CONFIGS_WITH_MODELS]
    )
    def test_no_config_names_a_tuned_twin_in_its_model_list(self, name, config):
        """``_opt`` names are dispatch keys, so the unknown-name check waves them through.

        And then the run fails anyway: with ``grid_search`` off,
        ``compute_dt_opt`` is called with ``data_key`` and raises TypeError; with it
        on, ``model_run`` looks for ``dt_opt_opt`` and reports "no '_opt'
        implementation". The twins are selected by ``grid_search: True``.
        """
        named = sorted(m for m in config["model"] if m.endswith("_opt"))
        assert not named, (
            f"{name} names the tuned twin(s) {named} in its 'model:' list; drop the "
            f"'_opt' suffix and set 'grid_search: True' instead"
        )

    @pytest.mark.parametrize(
        "name, config", CONFIGS_WITH_BLOCKS, ids=[n for n, _ in CONFIGS_WITH_BLOCKS]
    )
    def test_every_hyperparameter_block_names_a_real_model(self, name, config):
        """``<model>_args`` and ``gridsearch_<model>_args`` keys, stem by stem.

        A block for a model that no longer exists is worse than useless: it is silent.
        ``_model_args`` looks the block up by name, so a renamed learner leaves the old
        block sitting in the file being ignored while the new name runs on estimator
        defaults -- a config that looks configured and is not.

        Parametrised over ``CONFIGS_WITH_BLOCKS`` rather than every shipped config, and
        the emptiness re-checked per node: this test's only failure mode is a stem the
        dispatch table does not contain, so a node whose config has no blocks cannot
        fail whatever happens to that table, and a green node like that reads as coverage
        of a file nothing was checked in.
        """
        blocks = _hyperparameter_blocks(config)
        assert blocks, (
            f"{name} was parametrised in as a config with hyperparameter blocks and has "
            f"none, so this node asserts nothing; CONFIGS_WITH_BLOCKS and "
            f"_hyperparameter_blocks have drifted apart"
        )
        bad = {}
        for key in blocks:
            stem = key[: -len("_args")]
            if stem.startswith("gridsearch_"):
                stem = stem[len("gridsearch_"):]
            if stem not in DISPATCH_KEYS:
                bad[key] = stem
        assert not bad, (
            f"{name} has hyperparameter block(s) whose model is not in model_run's "
            f"dispatch table: {bad}"
        )

    def test_the_profiler_config_carries_a_block_for_every_model_it_names(self):
        """Not required by ``model_run`` -- but it is what this file demonstrates.

        ``_model_args`` falls back to the estimator defaults and logs, so a missing
        block runs. The shipped config is the worked example, though, and a model
        listed there without a block silently documents nothing about how to configure
        it. ``qpl`` is the reason this is asserted on the profiler config alone rather
        than parametrised: it has a ``gridsearch_qpl_args`` block and no plain one, and
        is not in the ``model:`` list either.
        """
        config = dict(SHIPPED_CONFIGS)["qbiocode/apps/qprofiler/configs/config.yaml"]
        missing = sorted(m for m in config["model"] if f"{m}_args" not in config)
        assert not missing, (
            f"the shipped profiler config lists {missing} in 'model:' with no "
            f"'<model>_args' block to copy from"
        )

    def test_the_shipped_config_still_covers_a_meaningful_number_of_blocks(self):
        """Second non-vacuity guard, on the block test rather than the file list.

        ``test_every_hyperparameter_block_names_a_real_model`` iterates whatever keys
        end in ``_args``; a reorganisation that nested them under a parent key would
        leave it iterating nothing.
        """
        config = dict(SHIPPED_CONFIGS)["qbiocode/apps/qprofiler/configs/config.yaml"]
        blocks = _hyperparameter_blocks(config)
        assert len(blocks) >= 20, f"only {len(blocks)} '_args' blocks found: {blocks}"


# ----------------------------------------------------------------------------------
# The returned keys
# ----------------------------------------------------------------------------------


class TestTheReturnedKeysArePrefixedNotBareModelNames:
    """The docstring promised ``result['dt']``. There is no such key, and never was.

    Both halves of the promise were wrong: the prefix, and the depth. ``result['dt']``
    is a KeyError and so is ``result['results_dt']['accuracy']`` -- the value is
    ``{0: {...}}``, because the frame is pivoted onto a single row index before
    ``to_dict()``.
    """

    def test_the_bare_model_name_is_not_a_key(self, tiny_split):
        X_train, X_test, y_train, y_test = tiny_split
        result = model_run(
            X_train, X_test, y_train, y_test, "shape", _classical_args(["dt"])
        )
        assert "dt" not in result
        assert set(result) == {f"{prefix}_dt" for prefix in LABEL_PREFIXES}

    def test_every_value_is_a_one_entry_dict_keyed_by_the_integer_zero(self, tiny_split):
        """Keyed by ``0``, not ``'0'``: a CSV round trip would give the string.

        Callers index it as ``result['results_dt'][0]``, so the key's type is part of
        the contract rather than an accident of the pivot.
        """
        X_train, X_test, y_train, y_test = tiny_split
        result = model_run(
            X_train, X_test, y_train, y_test, "shape", _classical_args(["dt"])
        )
        for key, value in result.items():
            assert list(value) == [0], f"{key} is not keyed by a single 0: {list(value)}"
        assert set(result["results_dt"][0]) == RESULT_KEYS

    def test_a_tuned_run_moves_the_suffix_into_the_key(self, tiny_split):
        """``grid_search: True`` renames the columns as well as the parameter key.

        A reader looking for ``results_dt`` after turning tuning on finds nothing, so
        the suffix belongs in the docstring: it is the difference between a KeyError
        and a result.
        """
        X_train, X_test, y_train, y_test = tiny_split
        args = _classical_args(
            ["dt"],
            grid_search=True,
            n_trials=2,
            cross_validation=2,
            gridsearch_dt_args={"criterion": ["gini", "entropy"], "max_depth": [2, 3]},
        )
        result = model_run(X_train, X_test, y_train, y_test, "tuned", args)
        assert set(result) == {f"{prefix}_dt_opt" for prefix in LABEL_PREFIXES}
        assert "results_dt" not in result
        assert "BestParams_Tuned" in result["results_dt_opt"][0]

    def test_the_docstring_documents_the_prefixed_keys(self):
        """The docstring is the only place most callers learn the shape.

        It claimed "keys as model names" while returning ``results_<model>``, so it
        was worse than absent -- a reader following it wrote a KeyError. Pinned as
        text so the fix cannot quietly regress: the assertions above would stay green
        if the sentence came back.
        """
        doc = model_run.__doc__
        assert "keys as model names" not in doc
        for prefix in LABEL_PREFIXES:
            assert f"'{prefix}_<label>'" in doc, f"the docstring does not name {prefix}_"
        # And the depth, which is the second half of what a caller gets wrong.
        assert "result['results_dt'][0]" in doc


# ----------------------------------------------------------------------------------
# The joblib fan-out
# ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fan_out(tiny_split):
    """One ``n_jobs=1`` run and one ``n_jobs=10`` run over the same three models.

    Module-scoped because the parallel half spawns loky workers, each of which imports
    ``qbiocode`` afresh -- seconds of process start-up that three assertions should
    not pay for three times. ``_SpyParallel`` is installed by hand rather than through
    ``monkeypatch``, which is function-scoped.

    ``n_jobs=10`` exceeds the three models on purpose: ``model_run`` clamps it with
    ``min(args['n_jobs'], len(args['model']))``, and the clamp is asserted below.
    """
    X_train, X_test, y_train, y_test = tiny_split
    models = ["dt", "lr", "nb"]
    original = MODEL_RUN_MODULE.Parallel
    _SpyParallel.calls = []
    MODEL_RUN_MODULE.Parallel = _SpyParallel
    try:
        serial = model_run(
            X_train, X_test, y_train, y_test, "serial",
            _classical_args(models, n_jobs=1),
        )
        serial_call = _SpyParallel.calls[-1]
        parallel = model_run(
            X_train, X_test, y_train, y_test, "parallel",
            _classical_args(models, n_jobs=10),
        )
        parallel_call = _SpyParallel.calls[-1]
    finally:
        MODEL_RUN_MODULE.Parallel = original
        _SpyParallel.calls = []
    return {
        "models": models,
        "serial": serial,
        "parallel": parallel,
        "serial_call": serial_call,
        "parallel_call": parallel_call,
    }


class TestTheJoblibFanOutMatchesTheSerialPath:
    """``n_jobs > 1`` runs the models in other processes. The results must not care."""

    def test_the_fan_out_really_ran_in_a_worker_process(self, fan_out):
        """Otherwise every other test in this class is about the sequential path.

        joblib picks a backend from ``n_jobs``: 1 means ``SequentialBackend``, in this
        process, and anything more means loky subprocesses. The pid of a task run in
        the same batch as the fits is the difference, and it is worth checking
        directly -- a fan-out that had silently degraded to sequential execution would
        produce identical numbers, so the equality tests below cannot detect it.
        """
        assert fan_out["serial_call"]["worker_pid"] == os.getpid()
        assert fan_out["parallel_call"]["worker_pid"] != os.getpid()

    def test_n_jobs_is_clamped_to_the_number_of_models(self, fan_out):
        """``n_jobs: 10`` with three models is three workers, not ten.

        The shipped config sets ``n_jobs: 10`` and its own advice is to keep it at or
        below the model count; the clamp is what makes a copied config with a shorter
        ``model:`` list harmless instead of ten idle interpreters.
        """
        assert fan_out["serial_call"]["n_jobs"] == 1
        assert fan_out["parallel_call"]["n_jobs"] == len(fan_out["models"])

    def test_the_spy_did_not_disturb_the_results_it_observed(self, fan_out):
        """The anchor under the pid claim above -- not the fan-out equality contract.

        That contract already exists: tests/test_model_contract_matrix.py's
        ``test_the_answer_does_not_depend_on_how_many_workers_ran`` compares all nine
        classical learners at ``n_jobs=4`` against a sequential run, metrics and
        predicted labels alike, and ``test_the_columns_do_not_depend_on_how_many_workers_ran``
        counts the columns. Nothing there proves a worker process was ever involved,
        which is what this class adds -- and the observation costs an extra ``os.getpid``
        task appended to the batch and stripped from the results again.

        So this checks what that manipulation could have broken: a model lost (the
        fold-up's ``dropna()`` would drop it silently rather than raise) or a result
        misaligned by the strip (which would show as the wrong model's numbers under a
        label, or a mismatch against the serial run). Without it the pid assertion
        could be passing on a batch whose real work had been thrown away.
        """
        expected = {
            f"{prefix}_{model}"
            for prefix in LABEL_PREFIXES
            for model in fan_out["models"]
        }
        assert set(fan_out["parallel"]) == expected
        assert set(fan_out["serial"]) == expected
        for model in fan_out["models"]:
            serial_row = fan_out["serial"][f"results_{model}"][0]
            parallel_row = fan_out["parallel"][f"results_{model}"][0]
            assert serial_row["model"] == parallel_row["model"] == model, (
                f"results_{model} does not hold {model}'s row; the strip of the "
                f"injected task misaligned the results"
            )
            for metric in ("accuracy", "f1_score", "auc"):
                assert serial_row[metric] == pytest.approx(
                    parallel_row[metric], nan_ok=True
                ), f"{model} disagrees on {metric} across the process boundary"
            np.testing.assert_array_equal(
                np.asarray(fan_out["serial"][f"y_predicted_{model}"][0]),
                np.asarray(fan_out["parallel"][f"y_predicted_{model}"][0]),
            )
