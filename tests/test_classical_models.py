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

"""Two of the nine classical learners were never fitted by the test suite at all.

``compute_mlp`` and ``compute_xgb`` had no executed coverage on their base path.
Every base-path assertion in the suite arrived through one of two doors:
``test_split_reproducibility.py``, whose parametrize list is ``dt, lr, rf, svc``
(plus ``nb``), or the end-to-end profiler run, whose ``MODELS`` is ``['lr', 'dt']``.
Their ``_opt`` twins are exercised, but a twin is a different function: the base
functions hand-write the constructor call, ``compute_mlp`` forwarding 24 keywords to
``MLPClassifier`` and ``compute_xgb`` 8 to ``XGBClassifier``, each by name. A keyword
renamed upstream breaks the fit outright, and the whole suite would have stayed
green -- the same for ``criterion`` or ``max_features`` disappearing from a tree.

The second thing nothing held was the shape of what comes back. The result row is
the published contract that ``ModelResults.csv``, ``qc_winner_finder`` and
``QuantumSage`` all read, and it has *six* keys -- ``model``, ``accuracy``,
``f1_score``, ``time``, ``auc`` and one parameter column -- yet no test pinned that
set for any model, on either branch of ``modeleval``. Existing tests index the one
key they happen to want, so a key renamed on the ``grid_search`` branch and not the
other, or an extra key on one model, would only surface as a ragged
``ModelResults.csv`` much later: the profiler writes its header once, from the first
row's keys, and appends every subsequent row positionally.

The third is subtler and is why ``row['model']`` is asserted here rather than taken
on trust. Six of the nine compute functions default ``model`` to a *display* name --
``'Decision Tree'``, ``'Naive Bayes'``, ``'Multi-layer Perceptron'`` -- and only
``model_run``'s ``model=method`` keyword overrides it. The name is interpolated into
the column name, so if that override were ever dropped the frame would come back
keyed ``results_Decision Tree`` and ``model_run``'s own ``raw['results_dt']`` lookup
would raise ``KeyError``. Nothing at unit tier watched that seam.

Assertions about hyperparameters are made against the *recorded* parameters rather
than against metrics, for the reason ``test_split_reproducibility.py`` sets out at
length: whether a hyperparameter changes the score depends on there being something
for it to change, so a metric comparison passes or fails by luck. The recorded value
is exact.

The fourth is what the ``auc`` column actually contains, and it was pinned *wrong*.
``modeleval`` used to compute ``roc_auc_score(y_test, y_predicted)`` -- ``roc_auc_score``
applied to hard predicted labels, which on a binary target is identically
``balanced_accuracy_score`` and no kind of ranking AUC. Because the number is finite, in
[0, 1] and tracks accuracy, a range check passes and a reader of ``ModelResults.csv``
has no way to notice. The test in this file asserted that identity, which recorded the
defect accurately and also meant a fix would fail the suite. ``auc`` is now the AUC of
the scores the fitted estimator supplies (``predict_proba``, else
``decision_function``), and the tests below pin *that*: the column is the
``roc_auc_score`` of the score array the learner handed ``modeleval``, and ``lr``'s
value now differs from its balanced accuracy, which is what proves the change took
effect. ``dt`` is the one model whose two numbers still agree -- a fully grown tree has
only pure leaves, so its ``decision_function`` takes two values and its AUC from scores
genuinely equals its AUC from labels. That coincidence is asserted where it appears so
it is not mistaken for the old bug surviving.

The remaining limitation pinned below is a limitation rather than a contract and is
marked as such where it is asserted: the classical path is binary-only for scoring
purposes. Three or more classes fit perfectly well -- ``OneVsOneClassifier`` is wrapped
around these estimators precisely for that -- and ``accuracy`` and ``f1_score`` come
back real, but no single ranking exists for a multiclass target, so ``auc`` is NaN. It
used to raise ``ValueError`` from inside ``roc_auc_score`` naming a ``multi_class``
parameter the user never set; NaN is the honest answer and better recorded than
rediscovered.
"""

from __future__ import annotations

import inspect
import math
import warnings

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.tree import DecisionTreeClassifier

# Imported directly rather than through ``pytest.importorskip``: every one of these is
# declared in requirements-base.txt, and tests/test_suite_hygiene.py fails the suite
# for guarding a mandatory dependency. ``qbiocode`` is imported first on purpose --
# tests/test_openmp_import_order.py exists because the vendored OpenMP runtimes have
# to be ordered before xgboost, catboost or torch arrive.
import qbiocode  # noqa: F401
from qbiocode.evaluation.model_evaluation import modeleval
from qbiocode.evaluation.model_run import model_run

#: The nine classical dispatch keys, in the order ``compute_ml_dict`` lists them.
#: ``catboost`` and ``tabpfn`` are here only for the cross-model half of the contract
#: -- the same result keys, the same metric ranges as the seven older learners. Their
#: internals are covered in depth by tests/test_catboost_tabpfn.py and are not
#: re-tested here.
CLASSICAL_MODELS = ["svc", "dt", "lr", "nb", "rf", "xgb", "mlp", "catboost", "tabpfn"]

#: The seven learners wrapped in ``OneVsOneClassifier``, which is what puts the
#: ``estimator__`` prefix on every recorded hyperparameter.
WRAPPED_MODELS = ["svc", "dt", "lr", "nb", "rf", "xgb", "mlp"]

#: The two fitted directly, with no wrapper, so their parameters are recorded bare.
UNWRAPPED_MODELS = ["catboost", "tabpfn"]

#: The result row every model must return with ``grid_search`` off. Six keys, not
#: five: ``time`` is the wall clock ``modeleval`` records, and it is easy to leave out
#: of a written contract because tests/integration/conftest.py deliberately excludes
#: it from the *reproducibility* contract -- a different question from whether the key
#: is there.
DOCUMENTED_KEYS = {"model", "accuracy", "f1_score", "time", "auc", "Model_Parameters"}

#: The three scores every downstream reader selects on.
METRICS = ["accuracy", "f1_score", "auc"]

#: Minimal untuned config: no ``<model>_args`` block, so every estimator runs at its
#: own defaults and ``_model_args`` has to substitute ``{}`` rather than raise.
BASE_ARGS = {"seed": 7, "n_jobs": 1, "grid_search": False}


def _dataset():
    """A separable binary problem, deliberately the same shape as the seeding tests.

    60 samples and 5 features: every model in the table fits it in well under a
    tenth of a second (TabPFN, a pretrained transformer doing a forward pass rather
    than a fit, takes a couple of seconds). Separable enough that a model which
    collapsed to a single predicted class fails the floor below rather than scoring
    a plausible-looking 0.5.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = (X[:, 0] + 0.3 * rng.normal(size=60) > 0).astype(int)
    return train_test_split(X, y, stratify=y, test_size=0.3, random_state=9)


def _run(model, **overrides):
    """One untuned ``model_run`` dispatch, returning its whole result dict.

    Warnings are swallowed rather than asserted on: scikit-learn 1.9 deprecates the
    ``penalty`` keyword ``compute_lr`` forwards, and XGBoost prints a native notice
    for the ``criterion`` it is handed and ignores. Neither is this file's subject,
    and letting them through drowns the failure messages that are.
    """
    X_train, X_test, y_train, y_test = _dataset()
    args = {**BASE_ARGS, "model": [model], **overrides}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model_run(X_train, X_test, y_train, y_test, "tiny", args)


@pytest.fixture(scope="module")
def base_run():
    """Fit each model at most once for the whole module, on first request.

    A dozen tests ask the same nine questions of the same nine fits; refitting per
    test would multiply TabPFN's forward pass by six for no added coverage. Caching
    per model rather than eagerly fitting all nine keeps one model's failure from
    erroring out the tests of the other eight.
    """
    cache: dict[str, dict] = {}

    def get(model):
        if model not in cache:
            cache[model] = _run(model)
        return cache[model]

    return get


def _compute_fn(model):
    """The base ``compute_<model>`` function itself, for signature inspection."""
    module = __import__(f"qbiocode.learning.compute_{model}", fromlist=["_"])
    return getattr(module, f"compute_{model}")


class TestEveryClassicalModelFits:
    """The floor: all nine complete a base fit and return a well-formed row.

    ``mlp`` and ``xgb`` had no test in this position at all, so a break in either
    constructor call was invisible. The other seven were reached only through tests
    asking a different question, which dereference one key of the row and would
    report a constructor break as a confusing ``KeyError``.
    """

    @pytest.mark.parametrize("model", CLASSICAL_MODELS)
    def test_every_classical_model_completes_a_base_fit_under_its_dispatch_key(
        self, model, base_run
    ):
        """The column name is built from the label, so it is also a seam test.

        ``modeleval`` interpolates whatever ``model`` it was handed into
        ``'results_' + model``. Six of these nine functions default that parameter to
        a display name, so a dropped ``model=method`` at dispatch produces
        ``results_Decision Tree`` -- which ``model_run`` itself then fails to find.
        """
        raw = base_run(model)
        results_columns = sorted(k for k in raw if k.startswith("results_"))
        assert results_columns == [f"results_{model}"], (
            f"a single-model run of {model!r} must produce exactly one results "
            f"column, named for the dispatch key; got {results_columns}. A "
            f"display-named column here means the model= keyword stopped reaching "
            f"modeleval."
        )

    @pytest.mark.parametrize("model", CLASSICAL_MODELS)
    def test_the_result_row_carries_exactly_the_documented_keys(self, model, base_run):
        """The published key set, pinned for the first time, on both counts.

        Equality rather than containment: an *extra* key is the damaging direction.
        The profiler writes ``ModelResults.csv``'s header once, from the first row it
        sees, then appends later rows positionally -- so one model contributing a key
        the others do not shifts every column after it in a file that still parses.
        """
        row = base_run(model)[f"results_{model}"][0]
        assert set(row) == DOCUMENTED_KEYS, (
            f"{model}'s result row is {sorted(row)}, not the documented "
            f"{sorted(DOCUMENTED_KEYS)}. Every reader of ModelResults.csv selects on "
            f"these names."
        )
        # The two parameter columns are exclusive by construction -- modeleval picks
        # one on the args['grid_search'] branch -- and QuantumSage's own comment says
        # "never both". With tuning off, a row claiming BestParams_Tuned would mean
        # that branch chose wrongly.
        assert "BestParams_Tuned" not in row, (
            f"{model} ran with grid_search off, so its parameters belong under "
            f"'Model_Parameters'; a 'BestParams_Tuned' key claims a search that never "
            f"happened."
        )

    @pytest.mark.parametrize("model", CLASSICAL_MODELS)
    def test_the_recorded_model_field_is_the_dispatch_key(self, model, base_run):
        """``qc_winner_finder`` groups by this field and QuantumSage selects on it.

        It is the config's own spelling that has to come back, not the estimator's
        prose name: six of the nine default to the latter, so this is asserting that
        the override at dispatch actually happened.
        """
        row = base_run(model)[f"results_{model}"][0]
        assert row["model"] == model, (
            f"the row records model={row['model']!r} for a run dispatched as "
            f"{model!r}; a display name here cannot be joined back to the config."
        )

    @pytest.mark.parametrize("metric", METRICS)
    @pytest.mark.parametrize("model", CLASSICAL_MODELS)
    def test_the_metrics_are_finite_and_within_the_unit_interval(
        self, model, metric, base_run
    ):
        """The range contract alone: every consumer reads these three as proportions.

        This is deliberately weak and is worth having anyway, because the readers it
        speaks for do not check: the correlation analysis, QuantumSage's regression
        target and ``qc_winner_finder``'s comparisons all take a value outside
        [0, 1] or a NaN and carry it silently into a published number.

        What it cannot see is a wrong value that happens to be in range -- a
        constant, or the right statistic computed from the wrong labels -- so it is
        not what pins these columns. That is the identity test immediately below for
        ``accuracy`` and ``f1_score``, and ``TestTheAucColumnIsARankingAuc`` near the
        end of the file for ``auc``. The finiteness half is not idle there either:
        ``auc`` is now NaN whenever the estimator cannot rank, so this test is also
        the assertion that all nine of these learners *can*.
        """
        value = base_run(model)[f"results_{model}"][0][metric]
        assert math.isfinite(value), f"{model} reported {metric}={value!r}"
        assert 0.0 <= value <= 1.0, (
            f"{model} reported {metric}={value!r}, outside [0, 1] -- every consumer "
            f"of this column, from the correlation analysis to QuantumSage's "
            f"regression target, treats it as a proportion."
        )

    @pytest.mark.parametrize("model", CLASSICAL_MODELS)
    def test_the_accuracy_and_f1_columns_score_the_predictions_recorded_beside_them(
        self, model, base_run
    ):
        """Both columns were held only by a range and a floor, which any literal meets.

        ``modeleval`` publishes five numbers and computes three of them, and until
        this test two of those three were unpinned: replacing
        ``accuracy_score(y_test, y_predicted)`` with the literal ``0.77`` and
        ``f1_score(...)`` with ``0.0`` -- two published columns computed from nothing
        at all -- left this file and the whole suite green. The range check above is
        satisfied by any constant in [0, 1], the floor below by any constant over
        0.6, and the only other reference to ``f1_score`` in the suite compares two
        runs of the same code to each other, where a constant is perfectly
        reproducible.

        The comparison needs nothing that was not already to hand: ``modeleval``
        records the two label arrays it scored in the same frame as the row, which is
        how ``auc`` was pinned from the start (wrongly, but it was pinned). These two
        were simply never asked.

        ``average='weighted'`` is asserted rather than assumed. Nothing overrides
        ``modeleval``'s default, and moving it to ``'macro'`` or ``'binary'`` would
        shift every f1 QBioCode has published while staying comfortably inside the
        range above.
        """
        raw = base_run(model)
        row = raw[f"results_{model}"][0]
        y_test = raw[f"y_test_{model}"][0]
        y_predicted = raw[f"y_predicted_{model}"][0]

        assert row["accuracy"] == pytest.approx(accuracy_score(y_test, y_predicted)), (
            f"{model} recorded accuracy={row['accuracy']!r}, which is not the "
            f"accuracy of the predictions recorded beside it "
            f"({accuracy_score(y_test, y_predicted)!r}). The column is meant to be "
            f"that score and nothing else."
        )
        expected_f1 = f1_score(y_test, y_predicted, average="weighted")
        assert row["f1_score"] == pytest.approx(expected_f1), (
            f"{model} recorded f1_score={row['f1_score']!r} against a weighted F1 of "
            f"{expected_f1!r} on the predictions recorded beside it. If the averaging "
            f"scheme moved, every historical f1 in every ModelResults.csv was computed "
            f"the old way and the two are not comparable."
        )

    @pytest.mark.parametrize("model", CLASSICAL_MODELS)
    def test_every_model_learns_more_than_chance_on_a_separable_problem(
        self, model, base_run
    ):
        """A floor, so a collapse to one class fails instead of passing at 0.5.

        The label is a thresholded linear function of the first feature, so 0.6 is a
        long way below what any of these nine manages (the weakest is logistic
        regression at 0.78) and a long way above chance. This says something only
        because the test above establishes that ``accuracy`` is the score of the
        recorded predictions; on its own a floor is met by any large enough literal.
        """
        accuracy = base_run(model)[f"results_{model}"][0]["accuracy"]
        assert accuracy > 0.6, (
            f"{model} scored {accuracy} on a linearly separable problem, which means "
            f"it is not learning rather than that it is a weak learner"
        )


#: The ``model`` default each compute function declares, which is *not* the dispatch
#: key for six of the nine. Recorded here so the divergence between the two layers is
#: documented rather than accidental: the display name is what a direct
#: ``compute_dt(...)`` call writes into its column name, and the tests above assert
#: that ``model_run`` overrides it.
DECLARED_MODEL_DEFAULTS = [
    ("svc", "SVC"),
    ("dt", "Decision Tree"),
    ("lr", "Logistic Regression"),
    ("nb", "Naive Bayes"),
    ("rf", "Random Forest"),
    ("mlp", "Multi-layer Perceptron"),
    ("xgb", "xgb"),
    ("catboost", "catboost"),
    ("tabpfn", "tabpfn"),
]


@pytest.mark.parametrize("model,declared", DECLARED_MODEL_DEFAULTS)
def test_each_compute_function_labels_its_own_columns_with_the_name_it_declares(
    model, declared
):
    """Two layers disagree about what a model is called, and that is load-bearing.

    A direct ``compute_dt(X_train, X_test, y_train, y_test, args)`` call -- which is
    how the tuning tests drive the ``_opt`` twins, and how a notebook uses one learner
    without the dispatcher -- returns a frame keyed ``results_Decision Tree``. Only
    ``model_run``'s ``model=method`` makes the column name match the config.

    So the call is made here rather than the signature merely read. Reading the
    default proves the label a direct caller *would* be given; it says nothing about
    whether the function uses it, and the two break independently. Deleting
    ``model=method`` from the dispatcher is caught by the tests at the top of this
    file. The mirror-image break -- a compute function that still accepts ``model``
    and then hands ``modeleval`` a hardcoded string instead, so the declared default
    is dead and a direct call comes back keyed ``results_dt`` -- is invisible to a
    signature check, which is exactly the frame name this file's third paragraph
    depends on.

    Both are asserted, in that order, so the failure message distinguishes 'the
    default moved' from 'the default is ignored'. Called with no ``model=`` keyword on
    purpose: passing one would test the dispatcher's path again instead of this one.
    """
    fn = _compute_fn(model)
    default = inspect.signature(fn).parameters["model"].default
    assert default == declared, (
        f"compute_{model} now defaults model={default!r} rather than {declared!r}. "
        f"That is the label a direct call writes into 'results_<label>', so it "
        f"changes the column name for every caller that does not pass model="
    )

    X_train, X_test, y_train, y_test = _dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = fn(X_train, X_test, y_train, y_test, dict(BASE_ARGS))

    expected_columns = sorted(
        [f"results_{declared}", f"y_predicted_{declared}", f"y_test_{declared}"]
    )
    assert sorted(frame.columns) == expected_columns, (
        f"compute_{model} declares model={declared!r} but a call that accepts that "
        f"default came back as {sorted(frame.columns)} rather than "
        f"{expected_columns}. The declared name is not the one the function uses, so "
        f"a direct caller cannot find its own results column."
    )
    assert frame[f"results_{declared}"][0]["model"] == declared, (
        f"compute_{model}'s column is named for {declared!r} while the row inside it "
        f"records model="
        f"{frame[f'results_{declared}'][0]['model']!r}; the two come from the same "
        f"argument and disagreeing means one of them was hardcoded"
    )


class TestTheRecordedParameterSchema:
    """``Model_Parameters`` is a published column, and its shape is not uniform.

    All seven older learners wrap their estimator in ``OneVsOneClassifier``, so
    ``get_params()`` records every hyperparameter under an ``estimator__`` prefix;
    CatBoost and TabPFN are fitted bare and record theirs unprefixed. The prefix
    appears in the existing suite only incidentally, inside ``random_state``
    assertions for four models -- nothing states that the wrapping is what produces
    it. Unwrapping a model to fix something else (multiclass AUC, say) would reshape
    the column for every downstream parser, and those four assertions would fail with
    a ``KeyError`` that names nothing.
    """

    @pytest.mark.parametrize("model", WRAPPED_MODELS)
    def test_a_wrapped_model_records_every_hyperparameter_behind_the_prefix(
        self, model, base_run
    ):
        params = base_run(model)[f"results_{model}"][0]["Model_Parameters"]
        # ``estimator`` is the wrapped instance itself and ``n_jobs`` belongs to the
        # wrapper rather than to the estimator; everything else is the estimator's.
        bare = sorted(k for k in params if not k.startswith("estimator__"))
        assert bare == ["estimator", "n_jobs"], (
            f"{model}'s recorded parameters carry unprefixed keys {bare}. Only the "
            f"OneVsOneClassifier's own 'estimator' and 'n_jobs' belong there -- "
            f"anything else means the wrapping changed, and with it the schema of "
            f"the Model_Parameters column."
        )
        assert params, f"{model} recorded no parameters at all"

    @pytest.mark.parametrize("model", UNWRAPPED_MODELS)
    def test_an_unwrapped_model_records_its_parameters_bare(self, model, base_run):
        """The converse, so the asymmetry is stated rather than inferred.

        CatBoost selects a multiclass loss itself and TabPFN classifies in context,
        so neither is wrapped -- which means a reader cannot assume one prefix rule
        for the whole table.
        """
        params = base_run(model)[f"results_{model}"][0]["Model_Parameters"]
        prefixed = sorted(k for k in params if k.startswith("estimator__"))
        assert not prefixed, (
            f"{model} is fitted without OneVsOneClassifier, so its parameters are "
            f"recorded bare; {prefixed} means it acquired a wrapper"
        )
        assert params, f"{model} recorded no parameters at all"

    def test_logistic_regression_still_defaults_to_the_saga_solver(self, base_run):
        """A deliberate choice that reads like an accident, so it needs pinning.

        ``compute_lr`` defaults ``solver='saga'`` where scikit-learn's own default is
        ``'lbfgs'``. saga is markedly slower, needs scaled features and converges
        differently, so 'tidying' it to the library default would move every logistic
        regression number QBioCode has published -- and nothing recorded that the
        divergence was chosen.
        """
        params = base_run("lr")["results_lr"][0]["Model_Parameters"]
        assert params["estimator__solver"] == "saga", (
            f"compute_lr ran with solver={params['estimator__solver']!r}. saga is "
            f"QBioCode's own default, not scikit-learn's; changing it changes results"
        )


#: ``(dispatch key, hyperparameter, value)`` -- one live hyperparameter per learner,
#: each chosen to differ from that function's own declared default so the assertion
#: cannot pass vacuously (the test checks that too).
FORWARDED_HYPERPARAMETERS = [
    ("dt", "max_depth", 3),
    ("lr", "C", 0.25),
    ("nb", "var_smoothing", 1e-7),
    ("rf", "n_estimators", 7),
    ("svc", "kernel", "linear"),
    ("xgb", "n_estimators", 7),
    ("mlp", "activation", "tanh"),
]


class TestPerModelHyperparametersReachTheEstimator:
    """The point of the config's per-model blocks, tested for one parameter only.

    Before this file the only ``args['<model>_args']`` forwarding assertion anywhere
    covered ``random_state``, for ``dt``. Everything else was untested, and the value
    hops through ``model_run._model_args`` -> ``_seeded_kwargs`` -> ``**kwargs`` ->
    a hand-written explicit argument list inside each compute function. A keyword
    dropped from one of those lists leaves the estimator at its default while the
    user's config looks honoured, and the results file agrees with the config because
    it is written from the same absent value.

    ``nb`` is the sharpest case: ``compute_nb``'s whole signature exposes exactly one
    hyperparameter, so a break there is total.
    """

    @pytest.mark.parametrize(
        "model,param,value",
        FORWARDED_HYPERPARAMETERS,
        ids=[f"{m}-{p}" for m, p, _ in FORWARDED_HYPERPARAMETERS],
    )
    def test_a_configured_hyperparameter_reaches_the_estimator(self, model, param, value):
        declared_default = inspect.signature(_compute_fn(model)).parameters[param].default
        assert value != declared_default, (
            f"the test value for {model}.{param} equals compute_{model}'s own "
            f"default {declared_default!r}, so the assertion below would pass even "
            f"if nothing were forwarded. Pick a different value."
        )

        params = _run(model, **{f"{model}_args": {param: value}})[f"results_{model}"][0][
            "Model_Parameters"
        ]
        recorded = params.get(f"estimator__{param}", "<absent>")
        assert recorded == value, (
            f"{model}_args set {param}={value!r} but the fitted estimator recorded "
            f"{recorded!r}. The config block did not reach the estimator, and the "
            f"results file records the default as though it had been chosen."
        )

    def test_a_hyperparameter_scikit_learn_has_dropped_is_swallowed_not_forwarded(self):
        """``lr_args: {multi_class: ovr}`` must neither crash nor claim to have worked.

        scikit-learn removed ``multi_class`` from ``LogisticRegression`` -- it is not
        in the installed signature at all -- and ``compute_lr`` keeps the parameter in
        its own signature at ``'deprecated'`` without forwarding it. That is the right
        shape for a shim: an old config that names it still runs. Both halves need
        holding, because each has a plausible-looking regression. Forwarding it again
        turns every such config into a ``TypeError`` from inside sklearn; dropping it
        from the signature turns the same config into a ``TypeError`` naming
        ``compute_lr``. And the parameter must stay absent from the recorded
        parameters, or the results file claims a multiclass strategy that had no
        effect.
        """
        assert (
            "multi_class"
            in inspect.signature(_compute_fn("lr")).parameters
        ), "compute_lr no longer accepts multi_class, so old configs naming it break"

        params = _run("lr", lr_args={"multi_class": "ovr"})["results_lr"][0][
            "Model_Parameters"
        ]
        assert "estimator__multi_class" not in params, (
            "multi_class reached LogisticRegression, which no longer has that "
            "parameter -- the shim is meant to absorb it, not pass it on"
        )
        # The contrast matters: if forwarding were broken for *every* key, the
        # assertion above would be satisfied for the wrong reason.
        live = _run("lr", lr_args={"C": 0.25})["results_lr"][0]["Model_Parameters"]
        assert live["estimator__C"] == 0.25, (
            "a live hyperparameter in the same block did not reach the estimator "
            "either, so the multi_class assertion above proves nothing"
        )

    def test_a_key_the_learner_does_not_take_names_the_offending_keyword(self):
        """Copying a config block between models is an ordinary mistake.

        The nine signatures expose different subsets -- ``n_estimators`` is valid for
        ``rf`` and ``xgb`` but not for ``dt``, and ``compute_nb`` takes only
        ``var_smoothing`` -- so a pasted block produces this. The failure is a raw
        ``TypeError`` naming an internal function rather than the config block, which
        is the same shape ``model_run``'s validation was written to remove for model
        names; at least the offending keyword is in the message.
        """
        with pytest.raises(TypeError, match="n_estimators"):
            _run("dt", dt_args={"n_estimators": 5})


def test_xgboost_records_a_split_criterion_it_never_used():
    """``compute_xgb`` forwards ``criterion``, and ``XGBClassifier`` has no such thing.

    XGBoost prints its own notice and ignores the value, but it is recorded as
    ``estimator__criterion`` in ``Model_Parameters`` all the same -- so the results
    file states a split criterion that had no effect, and any value is accepted. This
    is exactly the defect class ``warn_ignored_hyperparameter`` and
    ``test_xgboost_says_so_when_given_a_parameter_it_ignores`` exist to catch, but
    that guard lives only in ``compute_xgb_opt``; the base path has neither the guard
    nor, until now, a test. It is worse here than in the tuned path, because a tuned
    run at least reports what it searched.

    Pinned rather than endorsed: routing the base path through the same warning would
    fail this test, which is the correct way to find out that the fix landed.
    """
    gini = _run("xgb", xgb_args={"criterion": "gini", "n_estimators": 10})
    entropy = _run("xgb", xgb_args={"criterion": "entropy", "n_estimators": 10})

    gini_row = gini["results_xgb"][0]
    entropy_row = entropy["results_xgb"][0]
    # The predicted labels, not their score. Inertness is a claim about the fitted
    # model, and a metric is a lossy view of it -- two runs can score identically
    # while disagreeing about which samples they got right. It also keeps this
    # assertion honest if ``accuracy`` itself ever stops being a real score: a
    # constant metric column would make an accuracy comparison here tautological,
    # which is how this test quietly stopped testing anything once before.
    assert np.array_equal(gini["y_predicted_xgb"][0], entropy["y_predicted_xgb"][0]), (
        "two xgb runs differing only in criterion predicted different labels, so "
        "criterion now does something -- if XGBoost has grown the parameter this test "
        "should become a real forwarding assertion, and if compute_xgb stopped "
        "forwarding it the recorded-parameter assertion below is what to update"
    )
    assert gini_row["Model_Parameters"]["estimator__criterion"] == "gini", (
        "criterion is inert but still recorded, which is the misleading part: a "
        "reader of ModelResults.csv concludes the split criterion was configured"
    )
    assert entropy_row["Model_Parameters"]["estimator__criterion"] == "entropy", (
        "the other value is recorded just as faithfully, which is what makes the "
        "column readable as a configured choice: whatever a config names comes back "
        "in the results file having done nothing"
    )


#: The eight learners whose fitted estimator can rank test samples finely. ``dt`` is
#: excluded and is the interesting exclusion: a ``DecisionTreeClassifier`` grown to pure
#: leaves separates the test set into exactly the two leaf-purity levels, so its
#: ``decision_function`` takes two values and ranks no better than its own labels do.
#: That is a real property of the model, not a residue of the old bug -- see
#: ``test_a_fully_grown_tree_is_the_one_model_whose_two_numbers_still_agree``.
RANKING_MODELS = ["svc", "lr", "nb", "rf", "xgb", "mlp", "catboost", "tabpfn"]


def _run_capturing_the_score(model):
    """One base fit, plus the ``y_score`` its module handed ``modeleval``.

    ``auc`` cannot be recomputed from anything in the results frame: the frame records
    ``y_test`` and ``y_predicted``, and the whole point of the fix is that the column is
    no longer a function of those. Nor is the fitted estimator returned -- the compute
    functions hand back a DataFrame -- so there is nothing to re-score afterwards.

    Rather than rebuild nine estimators here and hope each reconstruction matches the
    keywords its compute function actually passes (which is how a test starts asserting
    against its own copy of the code), this intercepts the real hand-off. The wrapper is
    installed on the ``modeleval`` name inside the learner's own module, which is where
    the call resolves, and delegates to the real one, so the frame that comes back is
    the genuine one.
    """
    module = __import__(f"qbiocode.learning.compute_{model}", fromlist=["_"])
    captured: dict = {}

    def spy(*args, **kwargs):
        captured["y_score"] = kwargs.get("y_score", "NOT PASSED AT ALL")
        return modeleval(*args, **kwargs)

    original = module.modeleval
    module.modeleval = spy
    try:
        raw = _run(model)
    finally:
        module.modeleval = original
    return raw, captured["y_score"]


class TestTheAucColumnIsARankingAuc:
    """What replaced the old balanced-accuracy pin, and why it is shaped this way.

    ``modeleval`` used to compute ``roc_auc_score(y_test, y_predicted)``. On hard
    predicted labels that is identically ``balanced_accuracy_score``, so the column
    named ``auc`` held a different statistic -- one that is finite, in [0, 1] and
    correlated with accuracy, which is precisely why nothing caught it. It now scores
    the ``y_score`` its caller supplies and reports NaN when there is none.

    The three tests below split the claim into the three things that can independently
    go wrong: the arithmetic (is the column the AUC of that array), the input (is that
    array a ranking rather than labels), and the outcome (did the number actually
    move).
    """

    @pytest.mark.parametrize("model", CLASSICAL_MODELS)
    def test_the_auc_column_is_the_roc_auc_of_the_scores_the_estimator_supplied(
        self, model
    ):
        """The arithmetic, against the exact array the learner handed over.

        Two failure modes this catches that a range check cannot: ``auc`` computed from
        ``y_predicted`` again (the original bug), and ``y_score`` computed correctly and
        then dropped on the floor by a call site that forgot the keyword -- which would
        silently produce NaN on that one model while the other eight stayed right.
        """
        raw, y_score = _run_capturing_the_score(model)
        row = raw[f"results_{model}"][0]
        y_test = raw[f"y_test_{model}"][0]

        assert y_score is not None and not isinstance(y_score, str), (
            f"compute_{model} passed {y_score!r} for y_score. None means its estimator "
            f"offers neither predict_proba nor decision_function, and the string means "
            f"the keyword was never passed at all -- either way this model's auc is "
            f"now NaN and nothing else in the suite would say so."
        )
        expected = roc_auc_score(y_test, np.asarray(y_score, dtype=float))
        assert row["auc"] == pytest.approx(expected), (
            f"{model} recorded auc={row['auc']!r} against a roc_auc_score of "
            f"{expected!r} on the scores it supplied. The column is meant to be that "
            f"and nothing else -- in particular not roc_auc_score of the predicted "
            f"labels, which is balanced accuracy and was what this column held for "
            f"every result QBioCode has ever written."
        )

    @pytest.mark.parametrize("model", RANKING_MODELS)
    def test_the_scores_are_a_ranking_and_not_a_relabelling_of_the_predictions(
        self, model
    ):
        """The input, checked for the property that makes an AUC mean anything.

        An AUC of a two-valued score is a balanced accuracy wearing a different name,
        so 'it is an AUC of *something*' is not enough: passing ``y_predicted`` through
        as ``y_score`` would satisfy the arithmetic test above exactly. What separates
        the two is resolution -- a genuine ``predict_proba`` or ``decision_function``
        output distinguishes samples the argmax lumps together.

        18 test rows, so the bar is set well below them: more than two distinct values
        is the qualitative line between a ranking and a thresholded label.
        """
        _, y_score = _run_capturing_the_score(model)
        distinct = np.unique(np.asarray(y_score, dtype=float))
        assert len(distinct) > 2, (
            f"{model} supplied {len(distinct)} distinct scores ({distinct!r}), so its "
            f"auc is a balanced accuracy under another name. Either the call site is "
            f"passing labels, or the estimator lost its predict_proba/"
            f"decision_function and extract_binary_scores fell through."
        )

    def test_logistic_regressions_auc_now_differs_from_its_balanced_accuracy(
        self, base_run
    ):
        """The outcome: the one assertion that fails if the fix is reverted.

        Everything above would still pass on the old code for a model whose scores
        happened to be two-valued, and ``lr`` is the clean witness that they do not
        have to be: its ``decision_function`` is a continuous margin, so its AUC and its
        balanced accuracy are genuinely different numbers rather than accidentally
        equal ones. On the fixture in this file they are 0.9125 and 0.7875 -- a gap far
        too wide to be rounding.

        Stated as an inequality rather than as those two literals so it survives a
        change to the fixture, which would move both numbers and invalidate a pinned
        pair while leaving the property intact.
        """
        raw = base_run("lr")
        row = raw["results_lr"][0]
        y_test = raw["y_test_lr"][0]
        y_predicted = raw["y_predicted_lr"][0]
        label_based = balanced_accuracy_score(y_test, y_predicted)

        assert row["auc"] != pytest.approx(label_based), (
            f"lr's auc is {row['auc']!r} and balanced_accuracy_score of its predictions "
            f"is {label_based!r}. Equal means modeleval is scoring hard labels again -- "
            f"the exact bug this file used to pin. Every 'auc' written before the fix "
            f"is the label-based number, including in "
            f"tutorial/QSage/data/qprofiler_benchmarks.csv, so old and new results "
            f"tables are not comparable."
        )

    def test_a_fully_grown_tree_is_the_one_model_whose_two_numbers_still_agree(
        self, base_run
    ):
        """``dt``'s coincidence, recorded so it is not read as the bug surviving.

        ``compute_dt`` grows the tree to purity, so every test sample lands in a leaf
        of one class and ``OneVsOneClassifier.decision_function`` returns one of two
        values. The AUC of a two-valued score *is* balanced accuracy, so ``dt``'s two
        numbers coincide -- honestly, because a tree of pure leaves has no ranking to
        offer, not because it is still being scored on labels.

        Which is why the witness above is ``lr`` and not ``dt``. Constraining depth or
        setting ``ccp_alpha`` would give this model a real ranking and turn the equality
        below into an inequality; that is an improvement and this test is the place to
        find out it happened.
        """
        raw = base_run("dt")
        row = raw["results_dt"][0]
        y_test = raw["y_test_dt"][0]
        y_predicted = raw["y_predicted_dt"][0]

        assert row["auc"] == pytest.approx(balanced_accuracy_score(y_test, y_predicted)), (
            "dt's auc no longer equals its balanced accuracy, which means the tree now "
            "produces more than two distinct decision_function values -- add dt to "
            "RANKING_MODELS and retire this test"
        )

    def test_the_seven_wrapped_learners_still_reach_their_scores_through_decision_function(
        self,
    ):
        """Why the fix needed a helper rather than a one-line ``predict_proba`` call.

        ``OneVsOneClassifier`` has no ``predict_proba``, which is what made the
        label-based AUC look unavoidable, and ``compute_svc``'s ``probability=True`` is
        accepted, recorded in ``Model_Parameters`` and then discarded by the wrapper.
        What the wrapper does have is ``decision_function``: on a binary target it holds
        one pairwise estimator and returns that estimator's own margin, a
        ``(n_samples,)`` vector oriented towards the positive class. If sklearn ever
        adds ``predict_proba`` to the wrapper, ``extract_binary_scores`` will start
        preferring it and the recorded AUCs may move (monotonically-related scores give
        the same AUC, and Platt scaling is not guaranteed to be monotone in the margin),
        so this is worth knowing about.
        """
        assert not hasattr(OneVsOneClassifier, "predict_proba"), (
            "OneVsOneClassifier has grown predict_proba, so extract_binary_scores now "
            "prefers it over decision_function for the seven wrapped learners -- check "
            "whether the recorded AUCs moved"
        )
        assert hasattr(OneVsOneClassifier, "decision_function"), (
            "OneVsOneClassifier lost decision_function, which is the only ranking the "
            "seven wrapped learners have -- their auc is now NaN"
        )


class TestAMissingScoreIsRecordedAsNaN:
    """The other half of the fix, asserted on ``modeleval`` directly.

    Threading a score through fifteen learner modules is the visible part; the part
    that keeps the column honest is what happens when a learner has no score to give.
    Falling back to ``roc_auc_score(y_test, y_predicted)`` is what the old code did,
    and it is the tempting fix for a NaN in a results table -- it puts a plausible
    number back in the column and reintroduces the bug wholesale. These two call
    ``modeleval`` with no estimator and no data pipeline, so they pin the policy rather
    than any learner's implementation of it.
    """

    ARGS = {"grid_search": False}

    def test_no_score_gives_nan_rather_than_the_label_based_number(self):
        y_test = [0, 0, 1, 1, 0, 1]
        y_predicted = [0, 1, 1, 1, 0, 0]
        frame = modeleval(
            y_test, y_predicted, 0.0, {}, self.ARGS, model="m", verbose=False
        )
        auc = frame["results_m"][0]["auc"]
        assert math.isnan(auc), (
            f"modeleval with no y_score recorded auc={auc!r}. The only honest answer is "
            f"NaN: a number here is a different statistic under the same column name, "
            f"which is what every consumer of this column then publishes as an AUC."
        )
        assert auc is not None, "NaN, not None -- the column must stay numeric"

    def test_a_supplied_score_is_the_only_thing_the_column_is_computed_from(self):
        """``y_predicted`` is deliberately wrong here, and must not affect ``auc``.

        The predictions passed in disagree with the scores, so a ``modeleval`` that
        still consulted them could not produce this value. It also fixes the
        orientation: larger ``y_score`` means the positive class, and an inverted
        convention would give ``1 - 0.75`` rather than ``0.75``.
        """
        y_test = [0, 0, 1, 1]
        y_predicted = [1, 1, 0, 0]
        y_score = [0.1, 0.4, 0.35, 0.8]
        frame = modeleval(
            y_test,
            y_predicted,
            0.0,
            {},
            self.ARGS,
            model="m",
            verbose=False,
            y_score=y_score,
        )
        row = frame["results_m"][0]
        assert row["auc"] == pytest.approx(0.75), (
            f"auc={row['auc']!r} for a score whose roc_auc_score is 0.75; 0.25 means "
            f"the orientation is inverted and 0.0 means y_predicted is still being "
            f"scored"
        )
        assert row["accuracy"] == pytest.approx(0.0), (
            "accuracy must still come from y_predicted -- the two inputs are not "
            "interchangeable and this run separates them"
        )

    def test_a_single_class_test_set_gives_nan_rather_than_raising(self):
        """No ROC curve exists, and the run should still produce a row.

        A stratified split can hand a small fold one class, and aborting the whole
        model run over an undefined metric loses the accuracy and F1 that are perfectly
        well defined. ``evaluation_metrics`` in the same module already answers an
        impossible AUC request with NaN, so this is the module's own convention.

        scikit-learn 1.9 reaches the same answer by warning and returning NaN itself
        rather than raising, so ``modeleval``'s own ``except ValueError`` is not what
        produces this -- it is the belt to that braces, and this test pins the outcome
        rather than the route to it. The warning is swallowed because it is the
        library announcing the condition the test set up on purpose.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = modeleval(
                [1, 1, 1, 1],
                [1, 1, 0, 1],
                0.0,
                {},
                self.ARGS,
                model="m",
                verbose=False,
                y_score=[0.2, 0.9, 0.4, 0.6],
            )
        row = frame["results_m"][0]
        assert math.isnan(row["auc"])
        assert row["accuracy"] == pytest.approx(0.75)


class TestTheClassicalPathIsBinaryOnlyForScoring:
    """A third class fits, scores accuracy and F1, and reports ``auc`` as NaN.

    No single ranking exists for a multiclass target -- a multiclass AUC needs the whole
    probability matrix plus an explicit one-vs-rest or one-vs-one averaging choice,
    which is a different metric from the binary one this column reports (and is what
    ``evaluation_metrics`` in the same module is for). So ``extract_binary_scores``
    returns ``None`` for three or more classes and ``modeleval`` records NaN.

    This used to be a ``ValueError`` raised from inside ``roc_auc_score``, naming a
    ``multi_class`` parameter the user never set, which aborted the whole run over one
    column. It was doubly confusing because ``OneVsOneClassifier`` is wrapped around
    these estimators precisely to handle multiclass: the models could do it, the
    evaluation could not. They still cannot rank it -- but the fit, the accuracy and
    the F1 now survive, and only the undefined column is missing.
    """

    @staticmethod
    def _three_class_dataset():
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 5))
        y = np.digitize(X[:, 0], [-0.5, 0.5])
        return train_test_split(X, y, stratify=y, test_size=0.3, random_state=9)

    @pytest.mark.parametrize("model", ["dt", "rf"])
    def test_a_third_class_leaves_auc_undefined_without_losing_the_other_columns(
        self, model
    ):
        X_train, X_test, y_train, y_test = self._three_class_dataset()
        args = {**BASE_ARGS, "model": [model]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = model_run(X_train, X_test, y_train, y_test, "tiny", args)

        row = raw[f"results_{model}"][0]
        y_predicted = raw[f"y_predicted_{model}"][0]
        assert len(np.unique(y_train)) == 3
        assert math.isnan(row["auc"]), (
            f"{model} reported auc={row['auc']!r} on a three-class target. A single "
            f"score column cannot rank three classes, so anything finite here is a "
            f"binary AUC computed from two of them -- worse than the missing value."
        )
        assert row["accuracy"] == pytest.approx(accuracy_score(y_test, y_predicted)), (
            "accuracy is well defined for any class count and must survive -- the "
            "multiclass limit is in the ranking metric alone, and the run used to "
            "abort before reaching this"
        )
        assert math.isfinite(row["f1_score"]), "weighted F1 is defined for three classes"

    def test_the_wrapped_estimator_itself_handles_three_classes_perfectly_well(self):
        """Locates the limit in the evaluation rather than in the learner.

        Without this, the NaN above reads as 'these models are binary classifiers',
        which is wrong and would send a reader to the wrong file.
        """
        X_train, X_test, y_train, y_test = self._three_class_dataset()
        wrapped = OneVsOneClassifier(DecisionTreeClassifier(random_state=7))
        predictions = wrapped.fit(X_train, y_train).predict(X_test)
        assert len(np.unique(y_train)) == 3
        assert set(predictions) <= set(np.unique(y_train)), (
            "the OneVsOneClassifier wrapper predicts three classes without "
            "complaint, so the NaN above comes from extract_binary_scores in "
            "qbiocode/evaluation/model_evaluation.py, not from the learner"
        )
        assert wrapped.decision_function(X_test).shape[1] == 3, (
            "three columns, one per class, and no way to reduce them to one ranking "
            "without choosing an averaging scheme -- which is exactly why "
            "extract_binary_scores declines and modeleval records NaN"
        )
