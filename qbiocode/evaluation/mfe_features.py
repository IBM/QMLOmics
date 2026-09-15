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

"""The pyMFE meta-feature block used by :mod:`qbiocode.evaluation.dataset_evaluation`.

`pyMFE <https://github.com/ealcobaca/pymfe>`_ implements roughly 105 meta-features
from the meta-learning literature, including the full Lorena et al. (2019) data
complexity suite and the landmarking family. QBioCode uses a **curated subset**,
not all of them: a large fraction of pyMFE's catalogue is either broken,
degenerate, or computationally intractable on the data QBioCode actually profiles,
which is

* all-numeric (omics matrices and their embeddings -- no categorical attributes),
* binary-labelled (``qprofiler`` refuses anything else, see ``qprofiler.py``), and
* very often ``p >> n`` (thousands of features, tens to hundreds of samples).

``MFE_FEATURES`` below is that subset. Every exclusion is annotated with the
measured reason it was excluded, because the failures are silent: pyMFE returns
``NaN`` for a feature it could not compute and emits a warning that
``suppress_warnings=True`` -- which we need, or a single run prints thousands of
lines -- discards. A feature that quietly returns ``NaN`` or a constant on every
dataset is worse than an absent one, since it still occupies a column that QSage
then trains on.

Two numbers set the cost budget. At ``n=100, p=20000`` the curated set extracts in
about 50 s; the same set with ``eigenvalues`` added takes about 1010 s, and with
the ``itemset`` group added it does not finish in any reasonable time. Both are
excluded for that reason among others.

See ``MFE_FEATURES`` and ``BINARY_SCALAR_SD_SUPPRESSED`` for the specifics.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Optional

import numpy as np

# ==========================================================================
# The curated feature list
# ==========================================================================
# Grouped by pyMFE group. The comment above each group records what was dropped
# from it and why -- all reasons are measurements on QBioCode-shaped data, not
# guesses. tests/test_dataset_evaluation.py asserts that the excluded names stay
# excluded, so re-adding one fails loudly rather than silently shipping a column
# of NaN.
MFE_FEATURES = (
    # ---- general -------------------------------------------------------
    # Dropped: nr_class (always 2 -- qprofiler is binary-only by contract),
    # nr_cat / nr_bin / cat_to_num (always 0 -- the data is all-numeric),
    # num_to_cat (always NaN -- it divides by the categorical count),
    # freq_class (0.5 by construction on balanced binary; c2 below carries
    # imbalance without the redundancy).
    "nr_inst",
    "nr_attr",
    "attr_to_inst",
    "inst_to_attr",
    # ---- statistical ---------------------------------------------------
    # Dropped: eigenvalues -- O(p^3); it accounted for 957 s of a 1007 s run at
    #   p=20000, and its ".mean" is bit-identical to var.mean (the mean
    #   eigenvalue of the covariance matrix is trace/p, i.e. the mean variance),
    #   while its ".sd" duplicates the `Condition number` that
    #   dataset_evaluation still computes natively.
    # Dropped: g_mean / h_mean -- geometric and harmonic means are undefined for
    #   negative values, so they are NaN for anything that has been through
    #   `scale_train_test`, which is every embedded dataset.
    # Dropped: sd_ratio -- NaN in every shape probed (it needs an invertible
    #   pooled covariance matrix).
    # Dropped: lh_trace / roy_root -- these go to +inf once p >= n. That is
    #   strictly worse than NaN: an inf propagates silently into QSage's
    #   regressors instead of being caught as a missing value.
    # Dropped: nr_disc -- always 1 for two classes.
    # Kept despite degeneracy: can_cor, w_lambda, p_trace. All three collapse to
    #   a constant at p >= n (canonical correlation is exactly 1 when the classes
    #   are trivially separable) but move cleanly at p < n, which is the regime
    #   of the *embedded* evaluate() call, and all are bounded so they degrade to
    #   a constant rather than to inf.
    "cor",
    "nr_cor_attr",
    "cov",
    "var",
    "sd",
    "mad",
    "iq_range",
    "skewness",
    "kurtosis",
    "range",
    "min",
    "max",
    "median",
    "mean",
    "t_mean",
    "nr_outliers",
    "nr_norm",
    "gravity",
    "can_cor",
    "w_lambda",
    "p_trace",
    # ---- info-theory ---------------------------------------------------
    # Dropped: attr_conc and class_conc. Both take a `max_attr_num` argument that
    #   defaults to 12, so on a matrix with thousands of features they silently
    #   describe 12 randomly chosen columns -- effectively noise, and different
    #   noise on every run. class_conc also cost 2.6 s at p=2000.
    # Dropped: attr_ent -- constant under pyMFE's uniform-width discretization.
    "class_ent",
    "mut_inf",
    "ns_ratio",
    "eq_num_attr",
    "joint_ent",
    # ---- model-based (decision-tree structure) -------------------------
    # Dropped: var_importance -- exactly 1/p in every probe, so it only restates
    #   nr_attr. leaves_per_class is 0.5 by construction on balanced binary.
    "leaves",
    "nodes",
    "tree_depth",
    "tree_shape",
    "tree_imbalance",
    "leaves_branch",
    "leaves_corrob",
    "leaves_homo",
    "nodes_per_attr",
    "nodes_per_inst",
    "nodes_per_level",
    "nodes_repeated",
    # ---- landmarking ---------------------------------------------------
    # The whole point of the exercise: QBioCode had no landmarking features at
    # all, and cheap-classifier performance is the strongest known predictor for
    # model selection, which is exactly what QSage does.
    # Dropped: random_node -- constant across the entire separation sweep.
    "best_node",
    "worst_node",
    "linear_discr",
    "naive_bayes",
    "one_nn",
    "elite_nn",
    # ---- clustering ----------------------------------------------------
    # Dropped: nre (ln 2 for balanced binary) and sc (0 in every probe).
    "sil",
    "ch",
    "vdb",
    "vdu",
    "int",
    "pb",
    # ---- concept -------------------------------------------------------
    "conceptvar",
    "wg_dist",
    "impconceptvar",
    "cohesiveness",
    # ---- complexity (Lorena et al. 2019) -------------------------------
    # Dropped: f1v -- raises ValueError on *every* dataset, iris included, under
    #   NumPy >= 2. pymfe 0.4.4 complexity.py assigns a (1, 1) array into a
    #   scalar slot (`df[ind] = _numen / _denom`), which NumPy 2 refuses. This is
    #   an upstream bug, and QBioCode cannot avoid NumPy 2 because
    #   qiskit-machine-learning 0.9.0 requires it. f1v is the multivariate
    #   Fisher measure, so its loss is why dataset_evaluation keeps its own
    #   `get_fdr` -- see that module's docstring.
    # Dropped: t1 -- 1.0 in every shape and at every separation probed.
    # Dropped: t2 -- exactly p/n, i.e. a duplicate of attr_to_inst above.
    #   pyMFE's own source comment notes the link.
    # Dropped: c1 -- for two classes, normalized class entropy equals class_ent
    #   above (verified equal to 12 significant figures).
    # Dropped: sparsity -- 0 for continuous data.
    # Kept despite degeneracy at p >= n: f2 (a product over features, so it
    #   underflows towards 0 in high dimension), f4, l1, l2, l3 (all 0 once the
    #   data is trivially linearly separable, which is free when p > n). All are
    #   bounded and all move cleanly at p < n.
    "f1",
    "f2",
    "f3",
    "f4",
    "l1",
    "l2",
    "l3",
    "n1",
    "n2",
    "n3",
    "n4",
    "t3",
    "t4",
    "lsc",
    "density",
    "cls_coef",
    "hubs",
    "c2",
)

#: Measures whose ``.sd`` column is *structurally* empty on binary data.
#:
#: These are one-vs-one measures: pyMFE computes one value per pair of classes.
#: With two classes there is exactly one pair, so the result is a length-1 array
#: and its standard deviation is NaN by definition -- not because anything went
#: wrong. ``can_cor`` is here for the same reason: ``min(n_classes - 1, p)`` is 1,
#: so there is only ever one canonical correlation to summarize.
#:
#: Their ``.mean`` columns are kept; only the empty ``.sd`` ones are dropped, so
#: the output carries no column that is guaranteed to be NaN.
BINARY_SCALAR_SD_SUPPRESSED = (
    "f2",
    "f3",
    "f4",
    "l1",
    "l2",
    "l3",
    "can_cor",
)

#: Prefix on every column this module produces.
#:
#: Load-bearing rather than decorative: it marks provenance in a wide results
#: table, guarantees no collision with the natively-computed column names, and
#: lets a consumer select the entire pyMFE block with one ``startswith`` test.
#: :class:`qbiocode.apps.sage.sage.QuantumSage` uses it for exactly that.
MFE_COLUMN_PREFIX = "mfe."

#: Summary functions applied to the vector-valued measures.
MFE_SUMMARY = ("mean", "sd")


def get_mfe_features(
    X: Any,
    y: Any,
    random_state: int = 0,
    features: Optional[Iterable[str]] = None,
    summary: Iterable[str] = MFE_SUMMARY,
) -> Dict[str, float]:
    """Extract the curated pyMFE meta-feature block for one dataset.

    Args:
        X (array-like): Feature matrix, observations in rows. Converted to a
            float :class:`numpy.ndarray`; a :class:`pandas.DataFrame` is fine.
        y (array-like): Class labels, one per row of ``X``.
        random_state (int): Seed forwarded to pyMFE. Not optional in practice --
            the landmarking measures cross-validate and the clustering measures
            run k-means, so without a fixed seed this whole block changes between
            two runs on identical input. Default 0.
        features (Iterable[str], optional): Override the measure list. Defaults
            to :data:`MFE_FEATURES`. Intended for tests and for callers that
            deliberately want a wider or narrower set; the module docstring
            explains why the default is curated.
        summary (Iterable[str]): pyMFE summary functions applied to the
            vector-valued measures. Defaults to :data:`MFE_SUMMARY`
            (``("mean", "sd")``).

    Returns:
        dict: Column name -> value, every key prefixed with
        :data:`MFE_COLUMN_PREFIX`. Columns listed in
        :data:`BINARY_SCALAR_SD_SUPPRESSED` are omitted when the labels are
        binary, because they would be NaN by construction.

    Raises:
        ImportError: If ``pymfe`` is not installed. It is a base dependency of
            QBioCode, so this only happens in a partial environment.
    """
    try:
        from pymfe.mfe import MFE
    except ImportError as exc:  # pragma: no cover - base dependency
        raise ImportError(
            "pymfe is required for dataset complexity evaluation but is not "
            "installed. It is a base dependency of QBioCode; install it with "
            "'pip install pymfe' or reinstall the package."
        ) from exc

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    selected = tuple(features) if features is not None else MFE_FEATURES

    # suppress_warnings is not cosmetic: pyMFE warns once per measure per
    # summary for anything it cannot compute, which on a wide matrix is
    # thousands of lines per dataset. The cost of hiding them is that a broken
    # measure returns NaN silently -- which is precisely why MFE_FEATURES is
    # curated against measured behaviour rather than trusted wholesale.
    extractor = MFE(
        features=list(selected),
        groups="all",
        summary=list(summary),
        random_state=random_state,
        suppress_warnings=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        extractor.fit(X, y, suppress_warnings=True)
        names, values = extractor.extract(suppress_warnings=True)

    # Only drop the structurally-empty .sd columns when the labels really are
    # binary. With more than two classes the one-vs-one measures return one
    # value per class pair and their .sd is meaningful.
    dropped: frozenset = frozenset()
    if np.unique(y).size == 2:
        dropped = frozenset(f"{name}.sd" for name in BINARY_SCALAR_SD_SUPPRESSED)

    return {
        f"{MFE_COLUMN_PREFIX}{name}": float(value)
        for name, value in zip(names, values)
        if name not in dropped
    }


def mfe_column_names(
    binary: bool = True,
    features: Optional[Iterable[str]] = None,
    summary: Iterable[str] = MFE_SUMMARY,
) -> tuple:
    """Column names :func:`get_mfe_features` produces, without computing them.

    Useful for building an empty results frame with the right schema, and for
    asserting the schema in tests without paying for an extraction.

    Args:
        binary (bool): Whether the labels are binary, which determines if the
            :data:`BINARY_SCALAR_SD_SUPPRESSED` ``.sd`` columns are omitted.
            Default True.
        features (Iterable[str], optional): Override the measure list. Defaults
            to :data:`MFE_FEATURES`.
        summary (Iterable[str]): Summary functions. Defaults to
            :data:`MFE_SUMMARY`.

    Returns:
        tuple: Column names in pyMFE's own (alphabetical) order.

    Note:
        pyMFE emits a bare name for a measure that is scalar for the given
        dataset and ``<name>.<summary>`` for one that is vector-valued, and which
        of those applies depends on the data (a per-feature measure is scalar
        only if there is one feature). This helper therefore returns the
        *superset*: both the bare and the summarized name for every measure. Use
        it for membership tests, not for an exact-equality assertion against a
        real extraction.
    """
    dropped = (
        frozenset(f"{name}.sd" for name in BINARY_SCALAR_SD_SUPPRESSED) if binary else frozenset()
    )
    selected = tuple(features) if features is not None else MFE_FEATURES
    names = []
    for measure in selected:
        names.append(measure)
        for suffix in summary:
            candidate = f"{measure}.{suffix}"
            if candidate not in dropped:
                names.append(candidate)
    return tuple(f"{MFE_COLUMN_PREFIX}{name}" for name in sorted(names))
