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

"""TabPFN, a pretrained tabular transformer, as a QProfiler classical learner.

TabPFN is not a learner in the sense the rest of this package's classical models
are. There is no training: the weights are fixed and pretrained on synthetic tabular
problems, and ``fit`` merely records the training rows so that ``predict`` can attend
over them. Everything it exposes as a hyperparameter is therefore an *inference*
setting, which has three consequences that shape this module.

**The import must stay lazy.** ``import tabpfn`` imports ``torch``, and this package
goes to some trouble not to. ``qbiocode/__init__.py`` calls
:func:`qbiocode.utils._openmp.preload_openmp_libraries` so that xgboost's copy of
``libomp`` initialises before torch's; ``tests/test_openmp_import_order.py`` then
asserts that ``import qbiocode`` leaves ``torch`` out of ``sys.modules`` entirely.
A module-level ``from tabpfn import TabPFNClassifier`` here would break that
assertion and map torch's OpenMP runtime into every QBioCode process, including the
ones that only ever wanted a random forest. So availability is probed with
``importlib.util.find_spec``, which does not execute the module, and the class is
imported inside :func:`_load_tabpfn_classifier` at first use.

**The weights are versioned, and the versions are licensed differently.** QBioCode
pins ``v2``, whose weights are under the Prior Labs License (Apache 2.0 plus an
attribution clause) and download anonymously -- no API token, no license acceptance.
Upstream's constructor defaults to ``v3`` instead, which is non-commercial *and*
non-production and reachable only after an interactive license acceptance; see
:data:`TABPFN_DEFAULT_VERSION` for the table and the reasoning. Selecting a restricted
version still works and still warns, and if its weights turn out to be unreachable
:func:`_explain_weight_access_failure` translates the failure at the boundary --
otherwise it surfaces as ``TabPFNLicenseError`` from several frames inside
``model_loading``, below anything the user wrote.

**There is a hard class limit.** The pretrained head supports at most ten classes,
and unlike the row and column limits this one cannot be waived by
``ignore_pretraining_limits``. It is checked up front so an eleven-class dataset
fails with its own class count in the message instead of a tensor shape error.

No ``OneVsOneClassifier`` wrapper, unlike :func:`compute_rf` and
:func:`compute_xgb`: TabPFN is natively multiclass up to that limit, and wrapping it
would multiply an already expensive forward pass by n(n-1)/2 to reach the same
answer.
"""

# ====== Base class imports ======

import importlib
import importlib.util
import os
import sys
import time
import warnings

import numpy as np

# ====== Scikit-learn imports ======

from sklearn.model_selection import GridSearchCV

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import extract_binary_scores, modeleval
from qbiocode.learning._grid import build_param_grid
from qbiocode.learning._tuning import build_search_space, run_study

# ====== Module constants ======

#: The pretrained classification head's hard ceiling. The row and feature limits
#: (50_000 and 2_000 as of the v2.5 checkpoints) merely degrade accuracy and can be
#: waived with ``ignore_pretraining_limits``; this one raises regardless.
TABPFN_MAX_CLASSES = 10

#: Cache for the lazily imported estimator class, so a tuned run importing it once per
#: trial does not re-enter the import machinery on every one.
_TABPFN_CLASSIFIER = None

#: The model version QBioCode pins, and the reason it is not the newest one.
#:
#: TabPFN's *code* is under the Prior Labs License (Apache 2.0 plus an attribution
#: clause). Its *weights* are licensed per model version, and the two regimes differ:
#:
#: ===========  ==========================================  ==================
#: version      weights licence                             commercial use
#: ===========  ==========================================  ==================
#: ``v2``       Prior Labs License v1.1 (Apache 2.0 + attr) permitted
#: ``v2.5``     TABPFN-2.5 Non-Commercial License           **no**
#: ``v2.6``     TABPFN-2.6 Non-Commercial License           **no**
#: ``v3``       TABPFN-3 Non-Commercial License             **no**
#: ===========  ==========================================  ==================
#:
#: The three newest are non-commercial *and non-production*, and reaching them requires
#: accepting that licence against your account -- which upstream does interactively, so
#: they also cannot be downloaded unattended. QBioCode is Apache-2.0 software whose users
#: include companies, so defaulting to a version that quietly imposes a non-commercial
#: licence on them would be the wrong default regardless of convenience. ``v2`` needs no
#: licence acceptance, no API token, and no account: it downloads anonymously and fits.
#:
#: Set ``model_version`` explicitly to opt into a newer one, having satisfied yourself
#: that its licence fits your use. Doing so warns, once, naming the licence.
TABPFN_DEFAULT_VERSION = "v2"

#: Version name -> the ``ModelSource`` accessor giving its default checkpoint filename.
#: Version selection reduces entirely to ``model_path``: upstream's
#: ``create_default_for_version`` sets that plus two values that are already the
#: constructor defaults. Resolving to a path rather than calling that classmethod is what
#: keeps :func:`qbiocode.learning._tuning.run_study` working -- it needs an estimator
#: *class* to instantiate per trial, not a pre-built instance -- and keeps the choice
#: inside ``get_params()``, so ``sklearn.base.clone`` cannot silently drop it mid-search.
_VERSION_SOURCES = {
    "v2": "get_classifier_v2",
    "v2.5": "get_classifier_v2_5",
    "v2.6": "get_classifier_v2_6",
    "v3": "get_classifier_v3",
}

#: Versions whose weights are non-commercial and non-production.
_RESTRICTED_VERSIONS = frozenset({"v2.5", "v2.6", "v3"})

#: The licence each restricted version is published under, for the warning text.
_VERSION_LICENCES = {
    "v2.5": "TABPFN-2.5 Non-Commercial License",
    "v2.6": "TABPFN-2.6 Non-Commercial License",
    "v3": "TABPFN-3 Non-Commercial License",
}

# ====== Begin functions ======


def tabpfn_is_available():
    """Whether the ``tabpfn`` distribution is importable, without importing it.

    ``find_spec`` locates the module and stops, so this stays free of the torch
    import that a real ``import tabpfn`` performs. It says nothing about whether the
    model *weights* are reachable -- that is a separate gate and cannot be known
    without attempting a fit.
    """
    try:
        return importlib.util.find_spec("tabpfn") is not None
    except (ImportError, ValueError):
        # A broken or partially uninstalled distribution can leave find_spec raising
        # rather than returning None. Either way it is not usable.
        return False


def _cap_openmp_threads():
    """Keep a TabPFN fit from killing the process on macOS, and say when it does.

    ``qbiocode/__init__.py`` deliberately initialises xgboost's ``libomp`` first (see
    :mod:`qbiocode.utils._openmp`), and importing ``tabpfn`` brings torch's copy in as a
    second LLVM OpenMP runtime under the same install name. The second one to open a
    parallel region dies in ``__kmp_fork_barrier`` -- below Python, so there is no
    traceback, no exception to catch, and no output at all::

        $ python fit_tabpfn.py ; echo $?
        139                     # SIGSEGV

    Measured on this tree: a ``compute_tabpfn`` call in a process that has imported
    ``qbiocode`` exits 139 without ``OMP_NUM_THREADS`` set, and returns a score with it.
    Reordering the imports cannot fix it -- xgboost-first breaks torch's parallel regions
    and torch-first breaks XGBoost's fits, so the two orderings are mutually exclusive and
    only disabling OpenMP satisfies both. Setting the variable here, immediately before the
    ``tabpfn`` import, is early enough: torch reads it when its runtime initialises.

    Scoped to macOS because that is where the duplicate-runtime crash occurs; capping
    threads elsewhere would be a needless performance ceiling. Uses the caller's value if
    they set one, so this cannot override a deliberate choice -- but it does then warn,
    because on this platform any value other than 1 risks the crash above.

    Returns:
        bool: Whether this call set the variable.
    """
    if sys.platform != "darwin":
        return False
    existing = os.environ.get("OMP_NUM_THREADS", "").strip()
    if existing:
        if existing != "1":
            warnings.warn(
                f"OMP_NUM_THREADS is set to {existing!r} and is being left alone, but on "
                f"macOS a TabPFN fit maps torch's OpenMP runtime in alongside xgboost's "
                f"and the second one to start can kill the process with SIGSEGV -- no "
                f"traceback, exit 139, a notebook front end reporting only 'kernel died'. "
                f"Set OMP_NUM_THREADS=1 before importing qbiocode if that happens.",
                UserWarning,
                stacklevel=3,
            )
        return False
    os.environ["OMP_NUM_THREADS"] = "1"
    warnings.warn(
        "Set OMP_NUM_THREADS=1 for this process before importing TabPFN. On macOS, torch's "
        "OpenMP runtime and xgboost's cannot both open a parallel region in one process, "
        "and the failure is a bare SIGSEGV rather than an exception. This caps OpenMP "
        "parallelism, which is the only setting that satisfies both libraries; set it "
        "yourself before importing qbiocode to choose differently.",
        UserWarning,
        stacklevel=3,
    )
    return True


def _load_tabpfn_classifier():
    """Import and return ``TabPFNClassifier``, with an actionable error if it is absent.

    TabPFN is an optional extra rather than a base dependency because of its weight: it
    pulls torch's ecosystem plus ``mlx``, ``lightgbm``, ``huggingface-hub`` and
    ``safetensors``, and ``import qbiocode`` must not pull torch in. That is the only
    reason -- the pinned ``v2`` weights need no token and no license acceptance.
    """
    global _TABPFN_CLASSIFIER
    if _TABPFN_CLASSIFIER is not None:
        return _TABPFN_CLASSIFIER

    if not tabpfn_is_available():
        raise ImportError(
            "TabPFN is not installed. It is an optional extra rather than a core "
            "dependency because of its weight: it brings torch's ecosystem along with "
            "mlx, lightgbm, huggingface-hub and safetensors.\n\n"
            "Install it with:\n"
            '  pip install "qbiocode[tabpfn]"\n'
            "or:\n"
            "  pip install -r requirements/requirements-tabpfn.txt\n\n"
            "No API key or license acceptance is needed: QBioCode pins model_version 'v2', "
            "whose weights are under the Prior Labs License (Apache 2.0 plus attribution) "
            "and download anonymously on first fit.\n\n"
            "To run without it, drop 'tabpfn' from the 'model' list in your config."
        )

    # Must happen before the import below, which pulls torch in. See _cap_openmp_threads.
    _cap_openmp_threads()

    try:
        module = importlib.import_module("tabpfn")
        _TABPFN_CLASSIFIER = module.TabPFNClassifier
    except Exception as exc:  # noqa: BLE001 -- a broken native install, not a typo
        raise ImportError(
            f"TabPFN is installed but could not be imported.\n"
            f"Error: {type(exc).__name__}: {exc}\n\n"
            f"This usually means its torch build does not match this platform. "
            f"Reinstall with:\n"
            f'  pip install --force-reinstall "qbiocode[tabpfn]"'
        ) from exc
    return _TABPFN_CLASSIFIER


def normalise_model_version(model_version):
    """Canonicalise a version name, accepting the spellings people actually write.

    ``'v2.5'``, ``'V2_5'``, ``'2.5'`` and ``tabpfn.constants.ModelVersion.V2_5`` all mean
    the same thing; a config is written by hand, so refusing one spelling of it would be
    an obstacle rather than a safeguard.

    Returns:
        str: One of the keys of :data:`_VERSION_SOURCES`.

    Raises:
        ValueError: If the name matches no known version.
    """
    name = getattr(model_version, "name", model_version)  # ModelVersion enum -> 'V2_5'
    text = str(name).strip().lower().lstrip("v").replace("_", ".")
    canonical = f"v{text}"
    if canonical not in _VERSION_SOURCES:
        raise ValueError(
            f"Unknown TabPFN model_version {model_version!r}. Choose one of "
            f"{sorted(_VERSION_SOURCES)}; {TABPFN_DEFAULT_VERSION!r} is the default and "
            f"the only one whose weights permit commercial use."
        )
    return canonical


def _warn_if_licence_restricted(version):
    """Say so, once per call, when a non-commercial checkpoint has been selected.

    Not refused: whether the licence fits is the user's call to make, and a library that
    silently overrode it would be worse than one that says what it is doing. But it must
    not be possible to end up on a non-commercial model without being told.
    """
    if version not in _RESTRICTED_VERSIONS:
        return
    warnings.warn(
        f"TabPFN model_version {version!r} selected. Its weights are published under the "
        f"{_VERSION_LICENCES[version]}, which permits testing, evaluation and internal "
        f"benchmarking but NOT revenue-generating activity, production systems, or "
        f"training other models for commercial use -- it is non-production as well as "
        f"non-commercial. It also requires accepting that licence against your Prior Labs "
        f"account before the weights will download. QBioCode's default, "
        f"{TABPFN_DEFAULT_VERSION!r}, is under the Prior Labs License (Apache 2.0 plus "
        f"attribution) and carries none of those restrictions. Satisfy yourself that this "
        f"licence fits your use before relying on the result.",
        UserWarning,
        stacklevel=3,
    )


def resolve_model_path(model_path="auto", model_version=TABPFN_DEFAULT_VERSION):
    """The checkpoint to load: an explicit path if given, else the pinned version's.

    Args:
        model_path (str): ``'auto'`` (the default) resolves from ``model_version``.
            Anything else is taken as a real path and wins -- someone pointing at a local
            checkpoint has been more specific than a version name, and that is also the
            way to run fully offline.
        model_version (str): See :data:`TABPFN_DEFAULT_VERSION`.

    Returns:
        str: A path suitable for ``TabPFNClassifier(model_path=...)``. The file need not
        exist yet; TabPFN downloads it there on first use.
    """
    if model_path not in (None, "auto"):
        return model_path

    _load_tabpfn_classifier()  # ensures tabpfn is importable, with the actionable error
    from tabpfn.model_loading import ModelSource, prepend_cache_path

    version = normalise_model_version(model_version)
    _warn_if_licence_restricted(version)
    source = getattr(ModelSource, _VERSION_SOURCES[version])()
    return prepend_cache_path(source.default_filename)


def _explain_weight_access_failure(exc, model):
    """Translate a weights-unavailable failure, or return None if that is not what it is.

    TabPFN raises ``TabPFNLicenseError`` when the license has not been accepted and
    ``TabPFNHuggingFaceGatedRepoError`` when the checkpoint repository itself is
    gated. Both come from inside ``model_loading`` during ``fit``, so without this the
    user sees a stack ending in a download helper and no indication that QProfiler
    asked for the model or how to satisfy it. Upstream's own text is good -- it names
    the URL and the environment variable -- so it is quoted rather than replaced.

    Returns:
        ImportError or None: The error to raise instead, or None when ``exc`` is an
        ordinary failure that should propagate untouched.
    """
    # Matched by name rather than by isinstance so that this function does not itself
    # need tabpfn imported, and so a renamed or removed upstream class degrades to
    # "propagate unchanged" instead of raising AttributeError inside error handling.
    gated = {"TabPFNLicenseError", "TabPFNHuggingFaceGatedRepoError"}
    if type(exc).__name__ not in gated:
        return None
    return ImportError(
        f"TabPFN is installed but its pretrained weights are not available, so "
        f"{model!r} cannot run.\n\n"
        f"{type(exc).__name__}: {exc}\n\n"
        f"QBioCode cannot accept the license on your behalf. Once the token is available "
        f"(or the checkpoint is already cached, or 'model_path' points at a local one), "
        f"this model runs unattended like any other.\n\n"
        f"To store the key so it survives a new shell, put it in a file outside the "
        f"repository -- which is also the only way it cannot be committed:\n"
        f"  python -c \"from qbiocode.utils import write_token_template; "
        f"print(write_token_template())\"\n"
        f"then paste the key into ~/.config/qbiocode/tabpfn.json. QProfiler reads it "
        f"automatically when 'tabpfn' is in the model list; elsewhere call "
        f"qbiocode.utils.load_tabpfn_token() first. See "
        f"qbiocode.utils.tabpfn_account.\n\n"
        f"Exporting TABPFN_TOKEN also works and takes precedence over the file, but it "
        f"does not survive a new shell or a notebook kernel started from a launcher.\n\n"
        f"If a token IS already configured, note that upstream's message above is "
        f"misleading: TabPFN falls back to a browser login whenever the license has not "
        f"been accepted, and then reports the missing terminal rather than the missing "
        f"acceptance -- so it tells you to set a variable that is already set. An API key "
        f"authenticates you; accepting the license is a separate action on your account. "
        f"To see which of the two is actually missing:\n"
        f"  python -c \"from qbiocode.utils import check_tabpfn_access; "
        f"print(check_tabpfn_access()['advice'])\"\n\n"
        f"To proceed without it, drop 'tabpfn' from the 'model' list in your config."
    )


def _check_class_count(y, model):
    """Reject a target with more classes than the pretrained head can represent.

    Checked here rather than left to TabPFN so the message carries the actual class
    count and names the model, and so it fails before the weights are downloaded.
    """
    n_classes = len(np.unique(np.asarray(y)))
    if n_classes > TABPFN_MAX_CLASSES:
        raise ValueError(
            f"{model!r} cannot be used on this dataset: TabPFN's pretrained "
            f"classification head supports at most {TABPFN_MAX_CLASSES} classes and the "
            f"target has {n_classes}. This limit is fixed by the checkpoint and is not "
            f"waived by 'ignore_pretraining_limits'. Use a different model for this "
            f"dataset, or group the rarer labels."
        )


def compute_tabpfn(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="tabpfn",
    data_key="",
    *,
    n_estimators="auto",
    softmax_temperature=0.9,
    balance_probabilities=False,
    average_before_softmax=False,
    fit_mode="fit_preprocessors",
    inference_precision="auto",
    memory_saving_mode="auto",
    ignore_pretraining_limits=False,
    model_path="auto",
    model_version=TABPFN_DEFAULT_VERSION,
    device="auto",
    n_preprocessing_jobs=1,
    random_state=None,
):
    """
    This function generates a model using `TabPFN <https://github.com/PriorLabs/TabPFN>`__, a transformer
    pretrained on synthetic tabular tasks that classifies by in-context learning rather than by fitting
    parameters to your data. It takes in parameter arguments specified in the config.yaml file, but will use
    the default parameters specified above if none are passed.
    The model is trained on the training dataset and validated on the test dataset. The function returns the evaluation of the model
    on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Because the weights are fixed, ``fit`` only memorises the training rows and every
    argument below is an inference setting. There is correspondingly no notion of
    underfitting the training data by giving it a smaller budget -- ``n_estimators``
    buys ensemble members over differently preprocessed views of the same rows, not
    capacity.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels. At most
            ``TABPFN_MAX_CLASSES`` (10) distinct values; see :func:`_check_class_count`.
        y_test (array-like): Test data labels.
        args (dict): Additional arguments, typically from a configuration file.
        verbose (bool): If True, prints additional information during execution.
        model (str): Name of the model being used, default is 'tabpfn'.
        data_key (str): Key for identifying the dataset, default is an empty string.
        n_estimators (int or str): Number of forward passes whose predictions are averaged,
            each over a differently preprocessed view of the data. ``'auto'`` (the default) lets
            TabPFN choose, raising the count on wide datasets so every feature is seen.
        softmax_temperature (float): Temperature applied to the logits when predicting. Lower is
            more confident; 1.0 is a no-op. Default is 0.9. Affects ``predict_proba`` far more
            than ``predict``, which does not sample.
        balance_probabilities (bool): Reweight predicted probabilities by the training class
            distribution. Helps when classes are imbalanced and the metric ignores that.
        average_before_softmax (bool): Average ensemble members' logits before the softmax rather
            than averaging their probabilities after it. Only meaningful when more than one
            estimator is used.
        fit_mode (str): How much preprocessing work to cache between fit and predict --
            ``'low_memory'``, ``'fit_preprocessors'`` (default), ``'fit_with_cache'`` or
            ``'batched'``. A speed/memory trade-off; it does not change the prediction.
        inference_precision (str or torch.dtype): Numeric precision for the forward pass.
        memory_saving_mode (bool or str or float): Cap on activation memory, trading speed for
            peak usage.
        ignore_pretraining_limits (bool): Proceed on data outside the range the checkpoint was
            pretrained on (50_000 rows, 2_000 features), and on a large dataset on CPU. Does
            **not** waive the ten-class limit, which is unwaivable.
        model_path (str): Where to load the checkpoint from. ``'auto'`` (the default) resolves
            from ``model_version`` and downloads to the system cache directory on first use,
            overridable with ``TABPFN_MODEL_CACHE_DIR``. An explicit path wins over
            ``model_version`` and is the way to run fully offline.
        model_version (str): Which pretrained checkpoint to use -- ``'v2'`` (the default),
            ``'v2.5'``, ``'v2.6'`` or ``'v3'``. **This is a licensing choice as much as a
            modelling one:** only ``'v2'`` permits commercial use, and the others are
            non-production as well as non-commercial and require accepting that licence
            against a Prior Labs account before they will download. Selecting one warns.
            See :data:`TABPFN_DEFAULT_VERSION`.
        device (str): Torch device for inference. ``'auto'`` picks an accelerator when present;
            note that MPS and CUDA are not numerically identical to CPU, so pin this to ``'cpu'``
            when comparing runs across machines.
        n_preprocessing_jobs (int): Worker processes for preprocessing. Left at 1 because
            QProfiler already runs one model per joblib worker.
        random_state (int or None): Seed for the estimator's own randomness. QProfiler fills this
            in from the run's ``seed`` so two runs at one seed agree; None leaves the estimator
            drawing from the global RNG.

    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model, including accuracy, AUC, F1 score, and the time taken for training and validation.

    Raises:
        ImportError: If the ``tabpfn`` extra is not installed, or is installed but its
            pretrained weights cannot be reached because the license has not been accepted.
        ValueError: If the target has more than ``TABPFN_MAX_CLASSES`` classes.
    """

    classifier_cls = _load_tabpfn_classifier()
    _check_class_count(y_train, model)
    model_path = resolve_model_path(model_path, model_version)

    beg_time = time.time()
    tabpfn = classifier_cls(
        n_estimators=n_estimators,
        softmax_temperature=softmax_temperature,
        balance_probabilities=balance_probabilities,
        average_before_softmax=average_before_softmax,
        fit_mode=fit_mode,
        inference_precision=inference_precision,
        memory_saving_mode=memory_saving_mode,
        ignore_pretraining_limits=ignore_pretraining_limits,
        model_path=model_path,
        device=device,
        n_preprocessing_jobs=n_preprocessing_jobs,
        random_state=random_state,
    )
    # Fit the training datset
    try:
        model_fit = tabpfn.fit(X_train, y_train)
    except Exception as exc:  # noqa: BLE001 -- narrowed immediately by the translator
        explained = _explain_weight_access_failure(exc, model)
        if explained is None:
            raise
        raise explained from exc
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = tabpfn.predict(X_test)
    # `auc` is computed from these probabilities alone. TabPFN is fitted unwrapped and a
    # probabilistic model by construction -- its forward pass returns a posterior -- so
    # predict_proba costs nothing extra here beyond a second pass over the test rows.
    y_score = extract_binary_scores(tabpfn, X_test)
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


def compute_tabpfn_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    cv=5,
    model="tabpfn",
    n_estimators=None,
    softmax_temperature=None,
    balance_probabilities=None,
    average_before_softmax=None,
    fit_mode=None,
    inference_precision=None,
    memory_saving_mode=None,
    ignore_pretraining_limits=False,
    model_path="auto",
    model_version=TABPFN_DEFAULT_VERSION,
    device="auto",
    n_preprocessing_jobs=1,
    random_state=None,
    *,
    tuner="optuna",
    n_trials=50,
):
    """
    This function also generates a model using `TabPFN <https://github.com/PriorLabs/TabPFN>`__.
    The difference here is that this function tunes the model's hyperparameters.
    The values or ranges searched for each parameter are specified in the config.yaml file,
    and ``tuner`` selects the search engine (Optuna by default). The
    combination of parameters that led to the best performance is saved and returned as best_params, which can then be used on similar
    datasets, without having to repeat the search.
    The model is trained on the training dataset and validated on the test dataset. The function returns the evaluation of the model
    on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model across the search.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Scored by k-fold ``cross_val_score`` through :func:`qbiocode.learning._tuning.run_study`,
    exactly like every other classical model, so a tuned TabPFN score is comparable to a tuned
    random forest one on the same split. Be aware of what that costs here: a trial is
    ``cv`` transformer forward passes over the training rows, not ``cv`` cheap tree fits, and
    what is being searched are inference settings rather than model capacity. The shipped
    ``gridsearch_tabpfn_args`` block is deliberately small for that reason -- widen it only when
    the wall clock allows.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Additional arguments, typically from a configuration file.
        verbose (bool): If True, prints additional information during execution.
        cv (int): Number of cross-validation folds, default is 5. Each fold is a full
            forward pass, so this multiplies the cost of every trial.
        model (str): Name of the model being used, default is 'tabpfn'.
        n_estimators (list or dict): Values or ``{low, high}`` range of ensemble sizes to search.
        softmax_temperature (list or dict): Values or range of softmax temperatures to search.
        balance_probabilities (list): Whether to reweight by class frequency, e.g. ``[True, False]``.
        average_before_softmax (list): Whether to average logits rather than probabilities.
        fit_mode (list): Preprocessing-cache strategies to search. A speed/memory trade-off that
            does not change predictions, so searching it only spends budget -- listed for
            completeness rather than recommended.
        inference_precision (list): Numeric precisions to search.
        memory_saving_mode (list or dict): Activation-memory caps to search.
        ignore_pretraining_limits (bool): Passed to every trial rather than searched.
        model_path (str): Passed to every trial rather than searched -- resolved once, so no
            trial can differ from another in which checkpoint it loaded.
        model_version (str): Which checkpoint to pin, resolved into ``model_path`` before the
            search starts. Not searchable on purpose: trials would then be comparing
            different models under different licences and reporting the winner as though it
            were a hyperparameter. See :data:`TABPFN_DEFAULT_VERSION`.
        device (str): Passed to every trial rather than searched.
        n_preprocessing_jobs (int): Passed to every trial rather than searched.
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
        ImportError: If the ``tabpfn`` extra is not installed, or its weights cannot be reached.
        ValueError: If the target has more than ``TABPFN_MAX_CLASSES`` classes.
    """

    classifier_cls = _load_tabpfn_classifier()
    _check_class_count(y_train, model)
    # Resolved once, before the search: every trial and the final refit must load the same
    # checkpoint, and the licence warning should fire once rather than n_trials times.
    model_path = resolve_model_path(model_path, model_version)

    beg_time = time.time()
    # Only the hyperparameters actually supplied. Passing all of them meant a
    # config that named a subset died in sklearn on the first one it left at its
    # `[]` default; see qbiocode.learning._grid.
    candidates = {
        "n_estimators": n_estimators,
        "softmax_temperature": softmax_temperature,
        "balance_probabilities": balance_probabilities,
        "average_before_softmax": average_before_softmax,
        "fit_mode": fit_mode,
        "inference_precision": inference_precision,
        "memory_saving_mode": memory_saving_mode,
    }

    # Settings every candidate shares. These are environment and cost choices rather
    # than things worth optimising, so they are fixed instead of searched -- and
    # `model_path` in particular must stay fixed, or trials would compare different
    # checkpoints and report the winner as though it were a hyperparameter.
    fixed = {
        "ignore_pretraining_limits": ignore_pretraining_limits,
        "model_path": model_path,
        "device": device,
        "n_preprocessing_jobs": n_preprocessing_jobs,
        "random_state": random_state,
    }

    # Optuna by default; the exhaustive grid stays reachable so a number published
    # against it can still be reproduced. Both engines are handed the same
    # `candidates`, so switching `tuner` never changes *which* hyperparameters are
    # searched -- only how the search spends its fits.
    try:
        if tuner == "grid":
            search = GridSearchCV(
                classifier_cls(**fixed),
                param_grid=build_param_grid("tabpfn", candidates),
                cv=cv,
            )
            search.fit(X_train, y_train)
            best_params = search.best_params_
        else:
            best_params = run_study(
                classifier_cls,
                build_search_space("tabpfn", candidates),
                X_train,
                y_train,
                cv=cv,
                n_trials=n_trials,
                seed=random_state,
                fixed=fixed,
            )
        best_tabpfn = classifier_cls(**best_params, **fixed)
        best_tabpfn.fit(X_train, y_train)
    except Exception as exc:  # noqa: BLE001 -- narrowed immediately by the translator
        explained = _explain_weight_access_failure(exc, model)
        if explained is None:
            raise
        raise explained from exc

    # Make predictions and calculate accuracy
    y_predicted = best_tabpfn.predict(X_test)
    # See compute_tabpfn: fitted unwrapped, so `auc` comes from predict_proba.
    y_score = extract_binary_scores(best_tabpfn, X_test)
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
