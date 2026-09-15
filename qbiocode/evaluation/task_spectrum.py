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

r"""The spectral distribution of ``y`` relative to the geometry of ``X``.

Every other complexity block QBioCode computes describes ``X`` alone (the
:mod:`~qbiocode.evaluation.mfe_features` statistical, correlation and
information-theoretic families, the native intrinsic-dimension and
conditioning measures) or describes ``y`` only through a *classifier's* view of
it (the pyMFE landmarking family, the Lorena et al. complexity family). None of
them answers the question this module is built around:

    given the geometry that ``X`` induces, is ``y`` a smooth function on that
    geometry, a structured oscillatory one, or broadband noise?

Those three cases are not distinguished by "hard" versus "easy". A target can be
perfectly deterministic, low-noise, and still sit almost entirely in the
*high-frequency* part of the geometry -- parity, checkerboard, and
alternating-sign targets all do. Classifiers with low-pass inductive bias
(linear models, small-bandwidth RBF, shallow trees) fail on those for reasons
that have nothing to do with label noise, and nothing in the existing block
separates that failure mode from ordinary difficulty.

Construction
------------
Build a weighted mutual k-NN graph on the rows of ``X`` with Zelnik-Manor and
Perona local scaling,

.. math::

    W_{ij} = \exp\!\left(-\frac{\lVert x_i - x_j \rVert^2}
                              {\sigma_i \sigma_j}\right)
    \quad\text{for } j \in N_k(i) \wedge i \in N_k(j),

where :math:`\sigma_i` is the distance from :math:`x_i` to its k-th nearest
neighbour, then take the symmetric normalized Laplacian
:math:`L_{\mathrm{sym}} = I - D^{-1/2} W D^{-1/2}` and diagonalize it. Its
eigenvectors :math:`u_j`, ordered by :math:`0 = \lambda_1 \le \lambda_2 \le
\dots \le \lambda_n \le 2`, are a Fourier basis intrinsic to the data geometry:
small :math:`\lambda_j` are smooth modes, large :math:`\lambda_j` oscillate
between neighbours. Expanding the centred label vector in that basis,

.. math::

    \alpha_j = u_j^\top \tilde y,
    \qquad p_j = \frac{\alpha_j^2}{\sum_l \alpha_l^2},

gives a distribution of *target power over geometric frequency*. Every feature
here is a functional of :math:`\{(\lambda_j, p_j)\}` or of a closely related
spectrum, and each is scale-free and bounded so it degrades to a constant rather
than to ``inf`` -- see :mod:`qbiocode.evaluation.mfe_features` for why an
unbounded column is worse than a missing one.

What the features separate
--------------------------
=========================== ==================== ================ ==============================
Target                      ``graph_hf_mass``    ``spec_entropy`` Reading
=========================== ==================== ================ ==============================
smooth / simple             low                  low              easy geometric target
complex smooth              low / moderate       high             many low-frequency modes
structured oscillatory      high                 low / moderate   parity / checkerboard-like
noisy                       high                 high             broadband target or noise
=========================== ==================== ================ ==============================

``graph_hf_mass`` is the share of target power in modes whose neighbour
autocorrelation is negative, :math:`\lambda_j \ge 1` -- see
:data:`HF_EIGENVALUE_THRESHOLD` for why that threshold is derived from
:math:`L_{\mathrm{sym}}` rather than read off a quantile of its spectrum, and for
the measurement showing the quantile version reports the alternating target as
*low*-frequency. It alone cannot tell the third row from the fourth, and that
distinction is the scientifically load-bearing one: a structured
high-frequency target is a candidate for a representation-driven advantage,
while a broadband one is just noise. The entropy is what splits them, which is
why both ship.

Three design decisions worth knowing about
------------------------------------------
**The graph is MST-augmented to exactly one connected component.** Mutual k-NN
routinely disconnects -- at ``k=5, n=50`` it leaves outright singletons -- and a
disconnected graph breaks this construction in three separate ways: the null
space of :math:`L_{\mathrm{sym}}` becomes multi-dimensional (so "remove the
trivial constant" is ambiguous), an isolated vertex has degree 0 so
:math:`D^{-1/2}` is singular, and :math:`\log(n-1)` stops being the right
entropy normalizer. Rather than accept any of that, the mutual graph is repaired
by adding the shortest edges of the full Euclidean minimum spanning tree that
merge distinct components -- Kruskal restricted to MST edges. That is the
sparsest possible repair, it adds at most (components - 1) edges, and it leaves
exactly one zero eigenvalue.

Adding the *edge* is not sufficient, which is the part worth knowing: with local
scaling, the affinity of a bridge between well-separated components underflows to
exactly 0.0 in double precision (three blobs at mutual distance 20 with
:math:`\sigma \approx 0.5` give an exponent of -5903, against an underflow limit
near -745), so the graph stays disconnected and the trivial-mode removal silently
takes one of three. Repair edges are therefore floored at the smallest affinity the
graph already contains -- see :func:`_mutual_knn_affinity`.

**The trivial mode is projected out explicitly, not by centring.** For
:math:`L_{\mathrm{sym}}` the null eigenvector is :math:`u_1 \propto D^{1/2}
\mathbf{1}`, *not* :math:`\mathbf{1}`. Centring makes :math:`\tilde y \perp
\mathbf{1}`, which does **not** give :math:`\alpha_1 = 0` on a graph with
non-uniform degrees. So :math:`u_1` is dropped from the basis and :math:`p_j` is
normalized over the remaining :math:`n-1` modes.

**Half-life is reported relative to the graph's own slowest timescale.** The raw
diffusion half-life :math:`\tau_{1/2} = \inf\{t : R_y(t) \le 1/2\}`, with
:math:`R_y(t) = \sum_j p_j e^{-t\lambda_j}`, has units of :math:`1/\lambda`, so
across datasets it is dominated by ``n`` and ``k`` rather than by anything about
``y`` -- which would defeat the whole point of the multi-``k`` aggregation below.
Because :math:`R_y(t) \le e^{-t\lambda_2}` for every ``t``, the bound
:math:`\tau_{1/2} \le \ln 2 / \lambda_2` is exact, so
:math:`\lambda_2 \tau_{1/2} / \ln 2 \in (0, 1]` is dimensionless: the fraction
of the graph's slowest relaxation time that the target survives. 1 means the
target lies entirely in the smoothest available mode; near 0 means an
infinitesimal amount of smoothing destroys it. That normalized quantity is what
``task.diffusion_half_life`` holds, and the raw value is recoverable as
``task.diffusion_half_life * ln2 / lambda_2``.

Two guards against measuring an artefact
----------------------------------------
**Multi-``k`` aggregation.** Fixing ``k=10`` would make "target frequency" partly
a property of an arbitrary graph construction. Every graph-spectral feature is
therefore computed at each ``k`` in :data:`GRAPH_NEIGHBOURS` (clipped for small
``n``) and reported as the median. :func:`get_task_spectrum_features` will also
emit the coefficient of variation across ``k`` when ``stability=True``: a
descriptor that reads 0.1, 1.5, 0.2 across ``k in (5, 10, 20)`` is not measuring
a geometric property of the dataset and should not be trusted for that row.

**A permutation null.** In sparse high-dimensional data -- which is most of what
QBioCode profiles -- random labels look high-frequency *by default*, because
almost nothing is smooth on a near-uniform point cloud in 20000 dimensions. A
raw ``graph_hf_mass`` of 0.4 is therefore uninterpretable on its own. Each
feature ships alongside

.. math::

    Z_T = \frac{T(X, y) - \mathbb{E}_\pi[T(X, \pi(y))]}
               {\mathrm{SD}_\pi[T(X, \pi(y))]},

the standardized deviation from the same statistic under random relabelling with
the class proportions held fixed. This is cheap rather than expensive:
:math:`\pi` permutes ``y`` and leaves the graph alone, so the eigendecomposition
is computed once per ``k`` and every permutation costs one
:math:`U^\top \tilde y_\pi` product.

Cost
----
Two numbers set the budget, measured the same way
:mod:`qbiocode.evaluation.mfe_features` measures its own. At ``n=100, p=20000`` --
the shape at which the curated pyMFE set takes about 50 s -- this whole block
including 100 permutations takes **0.18 s**, so the negative control is affordable
in the sense that matters: it is not the thing you would switch off.

The scaling is different from pyMFE's, though, and in the one direction QBioCode's
data does not usually go. The dominant term is the ``O(n^3)`` eigendecomposition,
once per ``k``, plus an ``O(n^2 p)`` distance matrix: 0.07 s at ``n=50``, 0.61 s at
``n=200``, 2.7 s at ``n=400``, 12 s at ``n=800``. Permutations add almost nothing to
any of those, because :math:`\pi` leaves the graph alone. Pass
``task_spectrum=False`` to :func:`qbiocode.evaluation.dataset_evaluation.evaluate`
for an unusually tall dataset.

Interpreting the block
----------------------
The result these features exist to make reachable is *not* "quantum wins on hard
datasets", nor "quantum wins on nonlinear datasets" -- both are nearly vacuous.
It is the conjunction: a structured high-frequency target (high
``graph_hf_mass``, high ``graph_hf_mass_z``) of low spectral rank (low
``graph_spec_entropy``, low ``graph_bandwidth90``) on a low-complexity manifold,
with discriminative energy outside the dominant variance subspace (high
``pca_tail_signal``) and classical landmarking baselines unable to express it
(the ``mfe.*_nn``, ``mfe.linear_discr`` and ``mfe.naive_bayes`` columns). Any one
of those alone is a weak finding; the conjunction is a testable claim about which
geometric modes a representation captures.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

# ==========================================================================
# Construction constants
# ==========================================================================

#: Neighbourhood sizes the graph-spectral features are aggregated over.
#:
#: A single ``k`` would make target frequency partly a property of an arbitrary
#: graph, so the reported value is the median across these and the optional
#: ``.cv_k`` column is the coefficient of variation. Three values is the smallest
#: set that gives a median rather than a mean of two and still spans a factor of
#: four in scale; each is clipped to ``n - 2`` on small datasets, and duplicates
#: after clipping are collapsed.
GRAPH_NEIGHBOURS: Tuple[int, ...] = (5, 10, 20)

#: Eigenvalue above which target power counts as high-frequency, for
#: ``graph_hf_mass``.
#:
#: 1.0, and it is not an arbitrary constant. For :math:`L_{\mathrm{sym}}`, an
#: eigenpair :math:`(\lambda, u)` with :math:`v = D^{-1/2}u` satisfies
#: :math:`\lambda = 1 - v^\top W v / v^\top D v`, so
#:
#:     :math:`\lambda > 1 \iff v^\top W v < 0`,
#:
#: i.e. the mode's neighbour-weighted autocorrelation is *negative*: adjacent
#: observations systematically disagree. That is what "oscillatory" has to mean
#: here, defined by the Laplacian itself rather than by a cut through the
#: eigenvalue distribution.
#:
#: **A quantile of the spectrum was tried first and is wrong.** The natural reading
#: of "the highest-frequency quarter" is :math:`\lambda_j \ge Q_{0.75}(\lambda)`, but
#: the *shape* of the k-NN Laplacian spectrum changes drastically with ``k``, so that
#: threshold moves relative to the geometry. Measured on 160 points on a circle with
#: an alternating (+-+-) target -- the case the measure exists to detect:
#:
#: =========================== ============================ ============================
#: quantity                    ``k = 5``                    ``k = 10``
#: =========================== ============================ ============================
#: spectrum                    ``[0.002, 1.513]``, q75 1.38 ``[0.006, 1.231]``, q75 1.18
#: alternating mode            :math:`\lambda = 1.198`      :math:`\lambda = 1.190`
#: mass above :math:`Q_{0.75}` **0.009**                    1.000
#: mass above 1                0.995                        1.000
#: =========================== ============================ ============================
#:
#: The alternating target is a *single* mode both times, at essentially the same
#: eigenvalue both times, and the quantile rule calls it low-frequency at ``k = 5``
#: because the k=5 spectrum happens to have a long tail above it. Aggregating over
#: ``k`` then reported 0.009 -- below the 0.231 that random labels score. The
#: absolute threshold reads 0.995 and 1.000, and separates smooth (0.014, 0.024)
#: from broadband (0.634, 0.784) at both ``k``.
#:
#: Raise it towards :math:`\lambda_{\max}` for a stricter test, but note that
#: :math:`\lambda_{\max}` itself shrinks with ``k`` (1.51 at ``k = 5``, 1.23 at
#: ``k = 10`` above), so a threshold much past 1.2 can select no modes at all.
HF_EIGENVALUE_THRESHOLD: float = 1.0

#: Cumulative target power defining ``graph_bandwidth90``, and cumulative
#: ``X`` variance defining the ``pca_tail_signal`` split.
POWER_THRESHOLD: float = 0.90

#: Permutations drawn for the negative control. 100 puts the relative error of the
#: null SD near 7 %, which is the dominant term in the z-score's own error; the
#: cost is one matrix-vector product per permutation because the graph does not
#: change (see the module docstring).
N_PERMUTATIONS: int = 100

#: Magnitude the permutation z-scores are clipped to.
#:
#: Not cosmetic, and not a fudge. Every base feature here is bounded, and on a
#: strongly-structured target the permutation null can be tight enough to put the
#: raw z-score in the tens of thousands -- measured: 95764 for
#: ``diffusion_half_life`` on two well-separated blobs, whose null SD is ~1e-5 on a
#: statistic bounded by 1. Those digits assert far more resolution than the null
#: supports: with :data:`N_PERMUTATIONS` draws the SD estimate carries ~7 % relative
#: error and no tail probability finer than ~1/100 is resolvable at all, so every
#: :math:`|Z|` past ~10 already means "definitively outside the null" and the
#: magnitude beyond that is noise. Clipping keeps the column on a scale a regressor
#: can use, and is the reason no ``_z`` column can reach ``inf`` -- which
#: :mod:`qbiocode.evaluation.mfe_features` documents as strictly worse than ``NaN``.
Z_CLIP: float = 50.0

#: Neighbourhood scales ``purity_auc`` integrates over, as a count.
PURITY_SCALES: int = 12

#: Prefix on every column this module produces.
#:
#: Load-bearing rather than decorative, for the same reasons as
#: :data:`qbiocode.evaluation.mfe_features.MFE_COLUMN_PREFIX`: it marks provenance
#: in a wide results table, cannot collide with the native column names or the
#: ``mfe.`` block, and lets a consumer select or drop this entire block with one
#: ``startswith`` test. ``detect_complexity_schema`` relies on that.
TASK_COLUMN_PREFIX = "task."

#: The five features that depend on the k-NN graph, so are aggregated over
#: :data:`GRAPH_NEIGHBOURS` and are the ones ``stability=True`` reports a
#: coefficient of variation for.
GRAPH_FEATURES: Tuple[str, ...] = (
    "graph_dirichlet",
    "graph_hf_mass",
    "graph_spec_entropy",
    "graph_bandwidth90",
    "diffusion_half_life",
)

#: The three features with no k-NN graph in them, hence no ``k`` to aggregate over.
#: ``pca_tail_signal`` is a Gram-matrix spectrum, and both ``h0_fragmentation`` and
#: ``purity_auc`` are defined directly on the distance matrix.
GEOMETRY_FEATURES: Tuple[str, ...] = (
    "pca_tail_signal",
    "h0_fragmentation",
    "purity_auc",
)

#: Every feature this module computes, in output order.
TASK_FEATURES: Tuple[str, ...] = GRAPH_FEATURES + GEOMETRY_FEATURES

#: Smallest sample count the construction is defined for: ``k >= 2`` after clipping
#: to ``n - 2`` needs ``n >= 4``.
MIN_SAMPLES: int = 4

_EPS = 1e-12


# ==========================================================================
# Graph construction
# ==========================================================================


def _pairwise_sq_distances(X: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix, symmetric with an exactly zero diagonal.

    The Gram-matrix expansion is used rather than a loop because ``p`` is often in
    the thousands: it costs one ``n x p`` by ``p x n`` product. Its known weakness
    is catastrophic cancellation for near-identical rows, which shows up as small
    negative values; those are clipped to zero, and the diagonal is written
    explicitly so that a self-distance is exactly 0 rather than ``1e-13``.
    """
    sq_norms = np.einsum("ij,ij->i", X, X)
    D2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    D2 = 0.5 * (D2 + D2.T)
    np.fill_diagonal(D2, 0.0)
    return D2


def _mst_total_weight(D: np.ndarray) -> float:
    """Total edge weight of the Euclidean minimum spanning tree of ``D``.

    This is the whole of the H0 persistent homology this module needs. In a
    single-linkage (Vietoris-Rips) filtration the finite H0 death times are
    exactly the MST edge weights, so the total MST weight is the total finite H0
    persistence -- no ``ripser`` or ``gudhi`` dependency is required to get it, and
    the result is exact rather than an approximation on a subsampled filtration.

    Args:
        D (numpy.ndarray): Square, symmetric distance matrix.

    Returns:
        float: Sum of the MST edge weights. 0.0 for fewer than two points.

    Note:
        ``minimum_spanning_tree`` reads a zero entry as "no edge", so genuinely
        coincident points -- duplicate rows, which do occur in omics matrices --
        would silently disconnect the graph. Their distance is nudged to the
        smallest positive value present instead, which keeps them adjacent at
        negligible cost. If *every* off-diagonal distance is zero (all rows
        identical) the MST weight is 0, which is correct.
    """
    n = D.shape[0]
    if n < 2:
        return 0.0
    positive = D[D > 0]
    if positive.size == 0:
        return 0.0
    graph = D.copy()
    off_diagonal = ~np.eye(n, dtype=bool)
    graph[off_diagonal & (graph <= 0)] = positive.min() * _EPS
    return float(minimum_spanning_tree(csr_matrix(graph)).sum())


def _knn_mask(D: np.ndarray, k: int) -> np.ndarray:
    """Boolean ``M[i, j] = "j is among i's k nearest neighbours"``, self excluded.

    ``kind="stable"`` makes the tie-breaking reproducible: with duplicate distances
    -- common after discretization or on integer-valued omics counts -- an unstable
    sort would put a different neighbour in the k-th slot on different runs, which
    changes ``sigma_i`` and hence every feature in the block.
    """
    masked = D.copy()
    np.fill_diagonal(masked, np.inf)
    order = np.argsort(masked, axis=1, kind="stable")[:, :k]
    mask = np.zeros(D.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=1)
    return mask


def _mutual_knn_affinity(D2: np.ndarray, k: int) -> np.ndarray:
    r"""Locally-scaled mutual k-NN affinity matrix, repaired to one component.

    Args:
        D2 (numpy.ndarray): Squared Euclidean distance matrix.
        k (int): Neighbourhood size, already clipped to ``[2, n - 2]``.

    Returns:
        numpy.ndarray: Symmetric affinity ``W`` with a zero diagonal, non-negative,
        and connected.

    Note:
        Local scaling is Zelnik-Manor and Perona's self-tuning choice
        :math:`\sigma_i = \lVert x_i - x_{(k)} \rVert`, so the affinity adapts to
        the local density instead of imposing one global bandwidth. It has a
        property worth relying on: because :math:`\sigma_i \sigma_j` carries the
        same units as :math:`\lVert x_i - x_j \rVert^2`, ``W`` is **invariant under
        a global rescaling** ``X -> cX``. This module therefore does not standardize
        ``X``, matching the rest of :mod:`~qbiocode.evaluation.dataset_evaluation`
        (``get_complexity``'s Isomap and ``get_log_density``'s KDE also read the
        frame as given); only the *relative* per-feature scaling matters, and where
        that has been normalized upstream by ``scale_train_test`` it is already
        handled.

        A ``sigma_i`` of exactly zero -- k or more rows coincident with row ``i`` --
        would divide by zero. It is floored at a small multiple of the smallest
        positive neighbour distance, which sends those affinities to 1 (coincident
        points are maximally similar), the correct limit.
    """
    n = D2.shape[0]
    D = np.sqrt(D2)

    masked = D.copy()
    np.fill_diagonal(masked, np.inf)
    kth = np.partition(masked, k - 1, axis=1)[:, k - 1]

    positive = kth[kth > 0]
    floor = positive.min() * 1e-6 if positive.size else 1.0
    sigma = np.maximum(kth, floor)

    scale = np.outer(sigma, sigma)
    W = np.exp(-D2 / scale)

    mask = _knn_mask(D, k)
    # Mutual rather than union: an edge survives only if BOTH endpoints elected the
    # other. That is the stricter, more faithful notion of local neighbourhood --
    # it refuses the edges by which a single outlier attaches itself to a dense
    # cluster it is not part of -- at the cost of disconnecting the graph, which is
    # what the MST repair below exists to undo.
    mask &= mask.T
    W = np.where(mask, W, 0.0)
    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)

    n_components, labels = connected_components(csr_matrix(W > 0), directed=False)
    if n_components == 1:
        return W

    # Kruskal restricted to the Euclidean MST edges: walk them shortest-first and
    # keep only those merging two components of the mutual graph. Because the MST
    # spans every vertex, this always terminates at one component, and it adds
    # exactly (n_components - 1) edges -- the sparsest repair available.
    graph = D.copy()
    off_diagonal = ~np.eye(n, dtype=bool)
    strictly_positive = D[off_diagonal & (D > 0)]
    if strictly_positive.size:
        graph[off_diagonal & (graph <= 0)] = strictly_positive.min() * _EPS
    mst = minimum_spanning_tree(csr_matrix(graph)).tocoo()

    parent = list(range(n_components))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    # A repair edge's own locally-scaled affinity UNDERFLOWS TO EXACTLY ZERO for
    # well-separated components, which would leave the graph disconnected after all.
    # Three blobs at mutual distance 20 with sigma ~ 0.5 put the exponent at -5903,
    # against a double's underflow limit near -745: the edge is added, its weight is
    # 0.0, `W > 0` still reports three components, and `_laplacian_basis` then drops
    # one of three trivial modes and leaves two in the basis to absorb target power
    # at lambda = 0. Nothing about that failure is visible in the output.
    #
    # So a repair edge gets at least the smallest weight the graph already uses.
    # A weight of exactly 0 is not a weak edge, it is an absent one, and a weight of
    # 1e-300 is worse than either -- it makes lambda_2 numerically zero, which
    # reintroduces every problem the repair exists to solve. Flooring at an affinity
    # the graph already contains introduces no new constant, keeps the spectrum
    # well-conditioned, and still leaves the repair edges the weakest in the graph.
    existing = W[W > 0]
    floor = float(existing.min()) if existing.size else 1.0

    for edge in np.argsort(mst.data, kind="stable"):
        i, j = int(mst.row[edge]), int(mst.col[edge])
        root_i, root_j = find(labels[i]), find(labels[j])
        if root_i == root_j:
            continue
        parent[root_i] = root_j
        # max(): a component pair close enough for a representable affinity keeps its
        # real one, so the floor only ever applies where the alternative is zero.
        weight = max(float(np.exp(-D2[i, j] / scale[i, j])), floor)
        W[i, j] = W[j, i] = weight

    return W


def _laplacian_basis(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Non-trivial eigenpairs of :math:`L_{\mathrm{sym}} = I - D^{-1/2} W D^{-1/2}`.

    Args:
        W (numpy.ndarray): Symmetric, connected, non-negative affinity matrix.

    Returns:
        tuple: ``(eigenvalues, eigenvectors)`` with the single trivial mode removed
        -- ``eigenvalues`` is length ``n - 1``, ascending, clipped to ``[0, 2]``, and
        ``eigenvectors`` is ``n x (n - 1)`` with the basis vectors as columns.

    Note:
        The trivial mode is dropped by index, which is only valid because ``W`` is
        connected: then :math:`\lambda_1 = 0` has multiplicity exactly one and
        ``eigh`` returns it first. This is the mode that carries no oscillation and
        whose eigenvector is :math:`D^{1/2}\mathbf{1}` -- **not** :math:`\mathbf{1}`,
        which is why centring ``y`` does not remove it and it has to be projected out
        here.

        :math:`L_{\mathrm{sym}}` is symmetric with spectrum in :math:`[0, 2]`, so
        ``eigh`` is exact up to rounding and the clip only absorbs eigenvalues like
        ``-2e-16``. Bounding the spectrum is what makes ``graph_dirichlet`` bounded
        by 2 rather than by the arbitrary scale of an unnormalized Laplacian.
    """
    degree = W.sum(axis=1)
    # Connectivity means every degree is positive; the floor guards only against a
    # vertex whose every affinity underflowed to 0, which the MST repair can produce
    # for a point extremely far from the rest of the cloud.
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, _EPS))
    L = np.eye(W.shape[0]) - (inv_sqrt[:, None] * W * inv_sqrt[None, :])
    L = 0.5 * (L + L.T)
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Dropping the trivial mode by index is only correct at multiplicity one, and a
    # second zero eigenvalue means `W` arrived disconnected -- which the caller is
    # responsible for preventing and which produced exactly this bug once already
    # (see the underflow note in `_mutual_knn_affinity`). Refuse rather than return a
    # basis with residual trivial modes in it: they sit at lambda = 0, so they absorb
    # target power into the smoothest possible bin and every feature reads low.
    if eigenvalues.size > 1 and eigenvalues[1] <= 1e-8:
        n_trivial = int((eigenvalues <= 1e-8).sum())
        raise ValueError(
            f"The affinity graph has {n_trivial} connected components, not 1: "
            f"L_sym has {n_trivial} zero eigenvalues. The target spectrum needs a "
            "connected graph, which _mutual_knn_affinity is supposed to guarantee by "
            "MST repair; reaching here means a repair edge carried no usable weight."
        )

    return np.clip(eigenvalues[1:], 0.0, 2.0), eigenvectors[:, 1:]


# ==========================================================================
# The graph-spectral features
# ==========================================================================


def _target_power(basis: np.ndarray, y_centred: np.ndarray) -> np.ndarray:
    r"""Normalized target power :math:`p_j = \alpha_j^2 / \sum_l \alpha_l^2`.

    Returns a uniform distribution when the centred target has no component
    outside the trivial mode at all. That is not a silent failure: it is the
    honest answer that the expansion carries no information, and a uniform ``p``
    puts ``graph_spec_entropy`` at its maximum of 1 and ``graph_hf_mass`` at the
    fraction of the spectrum lying above the threshold, which is exactly
    "indistinguishable from broadband".
    """
    alpha = basis.T @ y_centred
    power = alpha * alpha
    total = power.sum()
    if not np.isfinite(total) or total <= _EPS:
        return np.full(power.shape, 1.0 / power.size)
    return power / total


def _spectral_entropy(power: np.ndarray) -> float:
    r"""Shannon entropy of ``p`` normalized by :math:`\log(n-1)`, so in ``[0, 1]``.

    ``n - 1`` is the number of non-trivial modes, hence the entropy of a target
    spread uniformly over the whole geometric spectrum -- the maximum attainable.
    The effective spectral rank the module docstring mentions is recoverable as
    ``exp(graph_spec_entropy * log(n - 1))``.
    """
    if power.size < 2:
        return 0.0
    nonzero = power[power > 0]
    return float(-(nonzero * np.log(nonzero)).sum() / np.log(power.size))


def _bandwidth(power: np.ndarray, threshold: float = POWER_THRESHOLD) -> float:
    """Fraction of the frequency-ordered modes needed to hold ``threshold`` power.

    ``power`` arrives in ascending-eigenvalue order, so the cumulative sum walks
    the spectrum from smooth to oscillatory and the answer is a normalized
    *bandwidth*: near 0 means a handful of low-frequency modes reconstruct the
    target, near 1 means the whole spectrum is needed.
    """
    reached = int(np.searchsorted(np.cumsum(power), threshold) + 1)
    return float(min(reached, power.size) / power.size)


def _diffusion_half_life(eigenvalues: np.ndarray, power: np.ndarray) -> float:
    r"""Half-life of the target under heat diffusion, in units of :math:`\ln 2/\lambda_2`.

    Solves :math:`R_y(t) = \sum_j p_j e^{-t\lambda_j} = 1/2` for ``t`` and returns
    :math:`\lambda_2 t / \ln 2 \in (0, 1]`. See the module docstring for why the
    normalization is not optional: the raw ``t`` is measured in units of
    :math:`1/\lambda` and would rank datasets by ``n`` and ``k``.

    The bracket is exact rather than searched for. :math:`R_y` is strictly
    decreasing with :math:`R_y(0) = 1`, and every :math:`\lambda_j \ge \lambda_2`
    gives :math:`R_y(t) \le e^{-t\lambda_2}`, so :math:`R_y(\ln 2/\lambda_2) \le 1/2`
    and the root always lies in :math:`[0, \ln 2/\lambda_2]`.
    """
    lambda_2 = float(eigenvalues[0])
    if lambda_2 <= _EPS:
        # A near-disconnected graph the MST repair joined by one very weak edge:
        # lambda_2 ~ 0 makes the normalization meaningless. Report the maximum,
        # which is the truthful reading -- the target survives all the smoothing
        # this graph is capable of applying on its slowest timescale.
        return 1.0

    upper = np.log(2.0) / lambda_2

    def residual(t: float) -> float:
        return float((power * np.exp(-t * eigenvalues)).sum() - 0.5)

    if residual(upper) >= 0.0:
        # Only reachable when rounding puts R(upper) a hair above 1/2, at which
        # point the root is the endpoint itself.
        return 1.0
    return float(brentq(residual, 0.0, upper, xtol=1e-12, rtol=1e-10) / upper)


def _graph_statistics(
    eigenvalues: np.ndarray,
    basis: np.ndarray,
    y_centred: np.ndarray,
    hf_threshold: float,
) -> Dict[str, float]:
    """The five graph-spectral features for one target vector on one fixed graph.

    Split out from the graph construction because this is the only part a
    permutation changes: ``eigenvalues``, ``basis`` and ``hf_threshold`` are
    properties of ``X`` and are computed once per ``k``, then reused across every
    permutation of ``y``. That is what makes the negative control affordable.
    """
    power = _target_power(basis, y_centred)
    return {
        # Spectral centroid of the target: sum_j lambda_j p_j, in [0, 2]. For the
        # UNNORMALIZED Laplacian on uncentred +/-1 labels this is exactly
        # 4 * (total affinity crossing the class boundary); with L_sym on centred
        # labels the quantity is sum_{i<j} W_ij (y_i/sqrt(d_i) - y_j/sqrt(d_j))^2,
        # which reads the same direction -- low means labels vary slowly along the
        # geometry -- without that clean identity.
        "graph_dirichlet": float((eigenvalues * power).sum()),
        "graph_hf_mass": float(power[eigenvalues >= hf_threshold].sum()),
        "graph_spec_entropy": _spectral_entropy(power),
        "graph_bandwidth90": _bandwidth(power),
        "diffusion_half_life": _diffusion_half_life(eigenvalues, power),
    }


# ==========================================================================
# The geometry features -- no k-NN graph, so no k to aggregate over
# ==========================================================================


def _gram_basis(X: np.ndarray) -> Tuple[np.ndarray, int]:
    r"""Row-space PCA basis of ``X`` and the count of PCs holding 90 % of variance.

    Args:
        X (numpy.ndarray): Feature matrix, observations in rows.

    Returns:
        tuple: ``(basis, r)`` where ``basis`` is ``n x rho`` with the PCA score
        directions as orthonormal columns in descending-variance order, ``rho`` is
        the numerical rank, and ``r`` is the number of leading columns whose
        eigenvalues reach :data:`POWER_THRESHOLD` of the total.

    Note:
        The eigendecomposition is of the ``n x n`` centred Gram matrix
        :math:`X_c X_c^\top`, never of the ``p x p`` covariance. For QBioCode's
        ``p >> n`` regime that is the difference between ``O(n^3)`` and ``O(p^3)`` --
        the same reason ``mfe_features`` excludes pyMFE's ``eigenvalues`` measure,
        which spent 957 s of a 1007 s run at ``p = 20000``. The eigenvalues are the
        squared singular values of :math:`X_c`, so the variance ordering is
        identical.

        **The kernel is excluded.** ``rho`` can be far below ``n - 1``: with ``p = 2``
        embedded features and ``n = 200`` samples, ``rho = 2``. Directions in the
        kernel of :math:`X_c X_c^\top` are not low-variance PCs, they are an
        arbitrary basis of a zero-variance subspace that ``X`` cannot express at all,
        and any target power there is unlearnable rather than merely hard. Including
        it would make ``pca_tail_signal`` read ~1.0 for every embedded dataset and
        measure nothing but ``p / n``. So ``q_j`` is normalized over the row space
        only.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    gram = Xc @ Xc.T
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    total = eigenvalues.sum()
    if total <= _EPS:
        return eigenvectors[:, :0], 0
    keep = eigenvalues > max(total, 0.0) * 1e-10
    eigenvalues = eigenvalues[keep]
    basis = eigenvectors[:, keep]

    explained = np.cumsum(eigenvalues) / eigenvalues.sum()
    r = int(np.searchsorted(explained, POWER_THRESHOLD) + 1)
    return basis, min(r, basis.shape[1])


def _pca_tail_signal(basis: np.ndarray, r: int, y_centred: np.ndarray) -> float:
    """Share of target power outside the PCs explaining 90 % of ``X``'s variance.

    High means the discriminative direction is a low-variance one -- exactly the
    situation in which unsupervised dimensionality reduction discards the signal,
    and one of the few readings here that is directly actionable about the
    embedding step rather than about the model.
    """
    if basis.shape[1] == 0 or r >= basis.shape[1]:
        return 0.0
    projection = basis.T @ y_centred
    power = projection * projection
    total = power.sum()
    if total <= _EPS:
        return 0.0
    return float(power[r:].sum() / total)


def _h0_fragmentation(D: np.ndarray, y: np.ndarray) -> float:
    """Class-conditioned total H0 persistence, relative to the pooled value.

    The ratio of the summed within-class MST weights to the MST weight of the
    pooled cloud. Both are exact total finite H0 persistences of a single-linkage
    filtration -- see :func:`_mst_total_weight`.

    Reads roughly 1 when each class occupies its own contiguous region, because
    connecting a class costs about what connecting the whole cloud costs. Rises
    towards 2 and beyond when classes interleave, because a within-class MST edge
    then has to step *over* points of the other class, so every hop costs about
    twice a pooled hop. That is the disconnected-label-regions signature, measured
    without reference to any neighbourhood size ``k``.
    """
    pooled = _mst_total_weight(D)
    if pooled <= _EPS:
        return 0.0
    within = 0.0
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        if members.size > 1:
            within += _mst_total_weight(D[np.ix_(members, members)])
    return float(within / pooled)


def _purity_scales(n: int) -> np.ndarray:
    """Neighbourhood sizes ``purity_auc`` samples, geometric from 1 to ``n/2``.

    Geometric rather than linear because interleaving is a scale-*ratio*
    phenomenon: the interesting structure in a checkerboard target lives between
    ``m = 1`` and ``m = 10``, and a linear grid to ``n/2`` would spend most of its
    points where purity has already flattened to chance.
    """
    n_eff = n - 1
    upper = max(2, n_eff // 2)
    fractions = np.geomspace(1.0 / n_eff, upper / n_eff, num=PURITY_SCALES)
    return np.unique(np.clip(np.ceil(fractions * n_eff).astype(int), 1, n_eff))


def _purity_auc(neighbour_order: np.ndarray, scales: np.ndarray, y: np.ndarray) -> float:
    """Chance-corrected neighbourhood label purity, integrated over log scale.

    At each scale ``m``, purity is the mean fraction of a point's ``m`` nearest
    neighbours carrying its own label. Chance level under random relabelling with
    the class proportions fixed is ``sum_c (n_c/n)((n_c-1)/(n-1))``, and purity is
    rescaled so that chance maps to 0 and perfect purity to 1. The integral is
    taken against ``log m`` so that each octave of scale contributes equally.

    The correction is signed on purpose. An alternating target has *fewer*
    same-label neighbours than chance, so this goes negative -- which is the
    strongest and most direct fingerprint of the alternating-sign phenomenon
    anywhere in this block, and clipping it at 0 would throw that away.
    """
    n = y.size
    counts = np.bincount(np.unique(y, return_inverse=True)[1])
    chance = float(((counts / n) * ((counts - 1) / (n - 1))).sum())
    if chance >= 1.0 - _EPS:
        return 0.0

    agreement = (y[neighbour_order] == y[:, None]).astype(float)
    cumulative = np.cumsum(agreement, axis=1)
    purity = cumulative[:, scales - 1].mean(axis=0) / scales
    corrected = (purity - chance) / (1.0 - chance)

    if scales.size < 2:
        return float(corrected[0])
    log_scales = np.log(scales)
    return float(
        np.trapezoid(corrected, x=log_scales) / (log_scales[-1] - log_scales[0])
    )


# ==========================================================================
# Public API
# ==========================================================================


def _clipped_neighbours(n: int, neighbours: Iterable[int]) -> Tuple[int, ...]:
    """The requested ``k`` values clipped to ``[2, n - 2]``, deduplicated, ordered.

    Clipping rather than raising keeps the block defined on the small-``n`` datasets
    QBioCode does profile: at ``n = 12`` the requested ``(5, 10, 20)`` becomes
    ``(5, 10)``, and at ``n = 6`` it collapses to ``(4,)`` -- a single graph, so the
    median is that one value and the ``.cv_k`` columns are 0.
    """
    upper = max(2, n - 2)
    return tuple(sorted({int(min(max(k, 2), upper)) for k in neighbours}))


def _centred(y_numeric: np.ndarray) -> np.ndarray:
    """Mean-centred target. On balanced binary labels this is a no-op."""
    return y_numeric - y_numeric.mean()


def get_task_spectrum_features(
    X: Any,
    y: Any,
    random_state: int = 0,
    neighbours: Sequence[int] = GRAPH_NEIGHBOURS,
    hf_threshold: float = HF_EIGENVALUE_THRESHOLD,
    n_permutations: int = N_PERMUTATIONS,
    stability: bool = False,
) -> Dict[str, float]:
    r"""Describe how the target sits in the geometric spectrum of the features.

    Computes the eight features named in :data:`TASK_FEATURES`, each aggregated
    over :data:`GRAPH_NEIGHBOURS` where it depends on a graph, plus a permutation
    z-score for each. See the module docstring for the construction, the
    interpretation table, and why the multi-``k`` aggregation and the permutation
    null are both load-bearing rather than decorative.

    Args:
        X (array-like): Feature matrix, observations in rows. Converted to a float
            :class:`numpy.ndarray`; a :class:`pandas.DataFrame` is fine. Not
            standardized -- local scaling makes the affinity invariant to a global
            rescaling, see :func:`_mutual_knn_affinity`.
        y (array-like): Class labels, one per row of ``X``. Any two or more distinct
            values; they are mapped to consecutive integers and centred, so
            ``{-1, +1}``, ``{0, 1}`` and string labels all behave identically.
        random_state (int): Seed for the label permutations. Not optional in
            practice: without it the z-score columns change between two runs on
            identical input. Default 0.
        neighbours (Sequence[int]): Neighbourhood sizes to aggregate the
            graph-spectral features over, clipped to ``[2, n - 2]`` and
            deduplicated. Defaults to :data:`GRAPH_NEIGHBOURS`. A single value is
            accepted and makes the ``.cv_k`` columns identically 0 -- which is the
            reason not to pass one.
        hf_threshold (float): Eigenvalue of :math:`L_{\mathrm{sym}}` above which
            target power counts as high-frequency, for ``graph_hf_mass``. Defaults to
            :data:`HF_EIGENVALUE_THRESHOLD` (1.0), the point at which a mode's
            neighbour-weighted autocorrelation turns negative. That constant is
            derived rather than chosen, and a quantile of the spectrum is measurably
            the wrong thing here -- see :data:`HF_EIGENVALUE_THRESHOLD`.
        n_permutations (int): Permutations drawn for the negative control. Defaults
            to :data:`N_PERMUTATIONS` (100). Set to 0 to skip it, which **omits**
            the eight ``_z`` columns rather than filling them with ``NaN`` -- an
            always-``NaN`` column is worse than an absent one, for the reasons
            :mod:`qbiocode.evaluation.mfe_features` documents. Must be 0 or at
            least 2, since the null SD needs two draws.
        stability (bool): Also emit the ``.cv_k`` coefficient of variation across
            ``neighbours`` for each of the five :data:`GRAPH_FEATURES`. Default
            False: these are a diagnostic for the analyst -- a descriptor reading
            0.1, 1.5, 0.2 across ``k`` is not measuring a property of the dataset --
            rather than a signal worth spending five columns of a small meta-dataset
            on.

    Returns:
        dict: Column name -> value, every key prefixed with
        :data:`TASK_COLUMN_PREFIX`. Eight feature columns, plus eight ``_z``
        columns when ``n_permutations`` is non-zero, plus five ``_cv_k`` columns
        when ``stability`` is True. Every value is finite.

    Raises:
        ValueError: If ``y`` has fewer than two distinct values (the target
            spectrum of a constant label is not defined), if ``X`` and ``y``
            disagree in length, if there are fewer than :data:`MIN_SAMPLES`
            observations, if ``X`` holds non-finite values, or if
            ``n_permutations`` is 1.

    Example:
        >>> import numpy as np
        >>> angle = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        >>> X = np.column_stack([np.cos(angle), np.sin(angle)])
        >>> smooth = get_task_spectrum_features(X, (angle < np.pi).astype(int))
        >>> alternating = get_task_spectrum_features(X, np.arange(120) % 2)
        >>> smooth["task.graph_hf_mass"] < alternating["task.graph_hf_mass"]
        True
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y).ravel()

    if X.shape[0] != y.size:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {y.size} labels; they must describe "
            "the same observations."
        )
    n = X.shape[0]
    if n < MIN_SAMPLES:
        raise ValueError(
            f"The target spectrum needs at least {MIN_SAMPLES} observations to build "
            f"a k-NN graph with k >= 2; got {n}."
        )
    if not np.isfinite(X).all():
        raise ValueError(
            "X holds NaN or inf. The target spectrum is built from a Euclidean "
            "distance matrix, which a single non-finite entry makes non-finite "
            "everywhere. Impute or drop those rows or columns first."
        )
    classes, y_encoded = np.unique(y, return_inverse=True)
    if classes.size < 2:
        raise ValueError(
            "The target spectrum describes how y varies over the geometry of X, "
            f"which is undefined for a constant target (y takes the single value "
            f"{classes[0]!r})."
        )
    if n_permutations == 1:
        raise ValueError(
            "n_permutations=1 cannot give a null standard deviation. Pass 0 to skip "
            "the permutation control, or 2 or more to compute it."
        )

    # ---- everything that depends only on X, computed once ----------------
    # This is the whole reason the permutation null is affordable: a permutation
    # changes y and nothing else, so the O(n^3) work below is never repeated.
    D2 = _pairwise_sq_distances(X)
    D = np.sqrt(D2)

    graphs = [
        _laplacian_basis(_mutual_knn_affinity(D2, k))
        for k in _clipped_neighbours(n, neighbours)
    ]

    gram, r = _gram_basis(X)

    neighbour_order = np.argsort(
        np.where(np.eye(n, dtype=bool), np.inf, D), axis=1, kind="stable"
    )[:, : n - 1]
    scales = _purity_scales(n)

    def statistics(labels: np.ndarray) -> Dict[str, np.ndarray]:
        """All eight features for one labelling, graph features per ``k``."""
        centred = _centred(labels.astype(float))
        per_k = [
            _graph_statistics(eigenvalues, basis, centred, float(hf_threshold))
            for eigenvalues, basis in graphs
        ]
        values = {name: np.array([row[name] for row in per_k]) for name in GRAPH_FEATURES}
        values["pca_tail_signal"] = np.array([_pca_tail_signal(gram, r, centred)])
        values["h0_fragmentation"] = np.array([_h0_fragmentation(D, labels)])
        values["purity_auc"] = np.array([_purity_auc(neighbour_order, scales, labels)])
        return values

    observed = statistics(y_encoded)
    features = {name: float(np.median(observed[name])) for name in TASK_FEATURES}

    result = {f"{TASK_COLUMN_PREFIX}{name}": features[name] for name in TASK_FEATURES}

    if stability:
        for name in GRAPH_FEATURES:
            across_k = observed[name]
            mean = float(np.abs(across_k).mean())
            result[f"{TASK_COLUMN_PREFIX}{name}_cv_k"] = (
                float(across_k.std(ddof=0) / mean) if mean > _EPS else 0.0
            )

    if n_permutations:
        rng = np.random.default_rng(random_state)
        null = {name: np.empty(n_permutations) for name in TASK_FEATURES}
        for draw in range(n_permutations):
            permuted = statistics(rng.permutation(y_encoded))
            for name in TASK_FEATURES:
                null[name][draw] = float(np.median(permuted[name]))
        for name in TASK_FEATURES:
            samples = null[name]
            spread = float(samples.std(ddof=1))
            # A zero-spread null means the statistic is constant under relabelling,
            # so the observed value carries no information beyond that constant and
            # the standardized deviation is 0 -- not inf, which would propagate
            # silently into QSage's regressors.
            standardized = (
                float((features[name] - samples.mean()) / spread) if spread > _EPS else 0.0
            )
            result[f"{TASK_COLUMN_PREFIX}{name}_z"] = float(
                np.clip(standardized, -Z_CLIP, Z_CLIP)
            )

    return result


def task_column_names(
    n_permutations: int = N_PERMUTATIONS,
    stability: bool = False,
) -> tuple:
    """Column names :func:`get_task_spectrum_features` produces, without computing them.

    Unlike :func:`qbiocode.evaluation.mfe_features.mfe_column_names`, this is exact
    rather than a superset: the eight features are all scalar by construction, so no
    name depends on the data. Use it to build an empty results frame with the right
    schema, or to assert the schema in a test without paying for an extraction.

    Args:
        n_permutations (int): Must match what will be passed to
            :func:`get_task_spectrum_features`; 0 omits the ``_z`` columns.
        stability (bool): Must match what will be passed to
            :func:`get_task_spectrum_features`; True adds the ``_cv_k`` columns.

    Returns:
        tuple: Column names, features first in :data:`TASK_FEATURES` order, then the
        ``_cv_k`` columns, then the ``_z`` columns.
    """
    names = [f"{TASK_COLUMN_PREFIX}{name}" for name in TASK_FEATURES]
    if stability:
        names += [f"{TASK_COLUMN_PREFIX}{name}_cv_k" for name in GRAPH_FEATURES]
    if n_permutations:
        names += [f"{TASK_COLUMN_PREFIX}{name}_z" for name in TASK_FEATURES]
    return tuple(names)
