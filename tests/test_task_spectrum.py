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

"""The target-spectrum block measures what it claims to, on targets we can label by hand.

The block exists to separate four kinds of target -- smooth, complex-smooth,
structured-oscillatory, and broadband -- and that claim is testable rather than
aspirational, because targets of each kind can be constructed to order. Points on
a circle with labels ``+++++-----`` are smooth by construction; the same points
with ``+-+-+-+-`` are structured-oscillatory; a random relabelling is broadband.
``TestTheInterpretationTable`` asserts the block reads each one correctly, and it
is the test that matters: everything else here guards a mechanism, but that one
guards the point.

Two failures found during development are pinned explicitly, because both were
silent and both would come back from a plausible "simplification":

``TestHighFrequencyMassIsNotAQuantile`` -- defining high-frequency mass as power
above the 75th percentile of the spectrum, which is the obvious reading of "the
highest-frequency quarter", reports the alternating target as *low*-frequency at
``k=5``. The shape of the k-NN Laplacian spectrum changes drastically with ``k``
(``[0.002, 1.513]`` at ``k=5`` against ``[0.006, 1.231]`` at ``k=10`` on the same
160 points), so a quantile of it does not track the geometry. The threshold is
absolute and derived instead -- see ``HF_EIGENVALUE_THRESHOLD``.

``TestTheTrivialModeIsProjectedOut`` -- centring ``y`` does *not* remove the
trivial mode, because the null eigenvector of ``L_sym`` is ``D^(1/2) 1`` rather
than ``1``. On a graph with non-uniform degrees a centred target keeps a real
component along it, and leaving it in would put spurious power at ``lambda = 0``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from qbiocode.evaluation.dataset_evaluation import (
    NATIVE_COMPLEXITY_COLUMNS,
    complexity_feature_columns,
    detect_complexity_schema,
    evaluate,
    task_columns,
)
from qbiocode.evaluation.task_spectrum import (
    GRAPH_FEATURES,
    GRAPH_NEIGHBOURS,
    HF_EIGENVALUE_THRESHOLD,
    MIN_SAMPLES,
    TASK_COLUMN_PREFIX,
    TASK_FEATURES,
    Z_CLIP,
    _centred,
    _laplacian_basis,
    _mutual_knn_affinity,
    _pairwise_sq_distances,
    get_task_spectrum_features,
    task_column_names,
)

N = 160


def _circle(n=N):
    """``n`` points equally spaced on the unit circle -- a 1-D manifold in 2-D.

    The reference geometry throughout, because its graph Fourier basis is the
    discrete Fourier basis, so "low frequency" and "high frequency" have an
    unambiguous meaning that does not depend on the construction being right.
    """
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angle), np.sin(angle)]), angle


def _targets(n=N):
    """One target of each kind from the interpretation table, on the same geometry."""
    _, angle = _circle(n)
    rng = np.random.default_rng(0)
    return {
        "smooth": (angle < np.pi).astype(int),
        "blocks": ((np.arange(n) // 8) % 2).astype(int),
        "alternating": (np.arange(n) % 2).astype(int),
        "random": rng.permutation(np.arange(n) % 2),
    }


@pytest.fixture(scope="module")
def spectra():
    """The block computed for each hand-labelled target, with stability columns on."""
    X, _ = _circle()
    return {
        name: get_task_spectrum_features(X, y, stability=True)
        for name, y in _targets().items()
    }


def _value(spectrum, feature):
    return spectrum[f"{TASK_COLUMN_PREFIX}{feature}"]


def _z(spectrum, feature):
    return spectrum[f"{TASK_COLUMN_PREFIX}{feature}_z"]


class TestSchema:
    """What the block promises its callers about the shape of its output."""

    def test_the_default_is_sixteen_columns(self, spectra):
        """Eight features and eight z-scores; the cv_k columns are opt-in.

        Pinned because the column budget was a deliberate decision, not an accident:
        the meta-dataset QSage trains on has a few hundred rows, so five more columns
        of a diagnostic that is mostly noise is a real cost.
        """
        X, _ = _circle()
        default = get_task_spectrum_features(X, _targets()["smooth"])
        assert len(default) == 16
        assert sum(1 for name in default if name.endswith("_z")) == 8
        assert not any(name.endswith("_cv_k") for name in default)

    def test_every_column_carries_the_prefix(self, spectra):
        for spectrum in spectra.values():
            assert all(name.startswith(TASK_COLUMN_PREFIX) for name in spectrum)

    @pytest.mark.parametrize(
        "n_permutations,stability", [(100, False), (100, True), (0, False), (0, True)]
    )
    def test_the_advertised_names_are_exactly_the_emitted_ones(self, n_permutations, stability):
        """``task_column_names`` is exact, not a superset, so it can be asserted against.

        Unlike the pyMFE block -- where whether a measure is scalar or vector-valued
        depends on the data -- every feature here is scalar by construction, so a
        caller building an empty results frame gets the real schema.
        """
        X, _ = _circle(60)
        emitted = get_task_spectrum_features(
            X, _targets(60)["smooth"], n_permutations=n_permutations, stability=stability
        )
        assert set(emitted) == set(task_column_names(n_permutations, stability))

    def test_skipping_the_null_omits_the_columns_rather_than_nanning_them(self):
        """An always-NaN column is worse than an absent one -- see ``mfe_features``."""
        X, _ = _circle(60)
        without = get_task_spectrum_features(X, _targets(60)["smooth"], n_permutations=0)
        assert len(without) == 8
        assert not any(name.endswith("_z") for name in without)

    def test_nothing_is_nan_or_inf(self, spectra):
        for name, spectrum in spectra.items():
            bad = {k: v for k, v in spectrum.items() if not np.isfinite(v)}
            assert not bad, f"non-finite columns on the {name} target: {bad}"


class TestTheInterpretationTable:
    """The block's actual claim: it separates the four kinds of target.

    =========================== ================== ================
    target                      ``graph_hf_mass``  ``spec_entropy``
    =========================== ================== ================
    smooth / simple             low                low
    complex smooth              low / moderate     high
    structured oscillatory      high               low / moderate
    noisy / broadband           high               high
    =========================== ================== ================

    Rows three and four are the pair that matters. Both are high-frequency, so
    ``graph_hf_mass`` cannot tell them apart, and the difference between them is the
    whole scientific point -- a structured oscillatory target is a candidate for a
    representation-driven advantage, a broadband one is noise. The entropy is what
    splits them.
    """

    def test_smooth_labels_are_low_frequency(self, spectra):
        smooth = spectra["smooth"]
        assert _value(smooth, "graph_dirichlet") < 0.2
        assert _value(smooth, "graph_hf_mass") < 0.1
        assert _value(smooth, "graph_bandwidth90") < 0.1

    def test_alternating_labels_are_high_frequency(self, spectra):
        """The alternating-sign phenomenon, generalized off the torus and onto a graph."""
        alternating = spectra["alternating"]
        assert _value(alternating, "graph_dirichlet") > 1.0
        assert _value(alternating, "graph_hf_mass") > 0.9

    def test_frequency_increases_monotonically_with_label_oscillation(self, spectra):
        """``+++++-----`` then period-8 blocks then ``+-+-+-`` is increasingly oscillatory."""
        energies = [_value(spectra[name], "graph_dirichlet")
                    for name in ("smooth", "blocks", "alternating")]
        assert energies == sorted(energies), energies

    def test_structured_and_broadband_are_both_high_frequency(self, spectra):
        """Neither ``graph_hf_mass`` alone nor ``graph_dirichlet`` alone can split them."""
        assert _value(spectra["alternating"], "graph_hf_mass") > 0.5
        assert _value(spectra["random"], "graph_hf_mass") > 0.5

    def test_entropy_is_what_splits_structured_from_broadband(self, spectra):
        """The load-bearing assertion of the whole block.

        An alternating target is a *single* geometric mode, so its power distribution
        has near-zero entropy. Random labels spread power over the entire spectrum, so
        theirs is near the maximum of 1. If this ever fails, the block has stopped
        being able to distinguish parity-like structure from noise and the four-row
        table above collapses to two rows.
        """
        structured = _value(spectra["alternating"], "graph_spec_entropy")
        broadband = _value(spectra["random"], "graph_spec_entropy")
        assert structured < 0.2, structured
        assert broadband > 0.7, broadband

    def test_smoothing_destroys_an_oscillatory_target_and_spares_a_smooth_one(self, spectra):
        """``diffusion_half_life``, in units of the graph's own slowest timescale."""
        assert _value(spectra["smooth"], "diffusion_half_life") > 0.5
        assert _value(spectra["alternating"], "diffusion_half_life") < 0.05

    def test_purity_goes_negative_for_an_alternating_target(self, spectra):
        """The most direct fingerprint of alternation in the block.

        An alternating target has *fewer* same-label neighbours than chance, so the
        chance-corrected purity is negative. Clipping it at zero -- an easy
        "tidying" -- would throw the signature away, which is why the sign is pinned.
        """
        assert _value(spectra["alternating"], "purity_auc") < -0.1
        assert _value(spectra["smooth"], "purity_auc") > 0.5
        assert abs(_value(spectra["random"], "purity_auc")) < 0.1

    def test_interleaved_labels_double_the_class_conditioned_persistence(self, spectra):
        """``h0_fragmentation``: a within-class hop over a foreign point costs double.

        Contiguous label regions cost about what the pooled cloud costs, so the ratio
        sits near 1. Alternating labels force every within-class MST edge to step over
        a point of the other class, so it approaches 2.
        """
        assert 0.9 < _value(spectra["smooth"], "h0_fragmentation") < 1.1
        assert _value(spectra["alternating"], "h0_fragmentation") > 1.8


class TestHighFrequencyMassIsNotAQuantile:
    """Pins the threshold that a quantile of the spectrum gets wrong.

    "The fraction of label power in the highest-frequency quarter of the spectrum"
    reads naturally as ``power[lambda >= quantile(lambda, 0.75)]``, and that was the
    first implementation. It is wrong, and wrong in the worst direction: it reported
    the alternating target -- the single case the measure exists to detect -- as
    *less* high-frequency than random labels (0.009 against 0.231).

    The cause is that the k-NN Laplacian spectrum changes shape with ``k``, not just
    scale. On these 160 points the non-trivial spectrum runs ``[0.002, 1.513]`` with
    a 75th percentile of 1.38 at ``k=5``, and ``[0.006, 1.231]`` with a 75th
    percentile of 1.18 at ``k=10``. The alternating mode sits at ``lambda ~ 1.19``
    both times, so the quantile rule calls the identical mode low-frequency at one
    ``k`` and maximally high-frequency at the other.
    """

    @pytest.mark.parametrize("k", GRAPH_NEIGHBOURS)
    def test_the_alternating_target_reads_high_frequency_at_every_k(self, k):
        """The regression guard. A quantile threshold fails this at ``k=5`` and ``k=20``."""
        X, _ = _circle()
        spectrum = get_task_spectrum_features(
            X, _targets()["alternating"], neighbours=[k], n_permutations=0
        )
        assert _value(spectrum, "graph_hf_mass") > 0.9, (
            f"alternating target reads low-frequency at k={k}; the high-frequency "
            "threshold has become sensitive to the shape of the spectrum again"
        )

    def test_the_threshold_is_where_neighbour_autocorrelation_turns_negative(self):
        r"""``lambda > 1`` iff ``v'Wv < 0`` for ``v = D^(-1/2) u`` -- the derivation.

        This is why the threshold is 1.0 and not a tunable constant: it is the exact
        point at which a mode stops correlating adjacent observations and starts
        anti-correlating them. Verified directly on the eigenvectors.
        """
        X, _ = _circle(80)
        W = _mutual_knn_affinity(_pairwise_sq_distances(X), 8)
        eigenvalues, basis = _laplacian_basis(W)
        degree = W.sum(axis=1)
        for value, vector in zip(eigenvalues, basis.T):
            v = vector / np.sqrt(degree)
            autocorrelation = float(v @ W @ v)
            if value > HF_EIGENVALUE_THRESHOLD + 1e-6:
                assert autocorrelation < 0, (value, autocorrelation)
            elif value < HF_EIGENVALUE_THRESHOLD - 1e-6:
                assert autocorrelation > 0, (value, autocorrelation)


class TestGraphConstruction:
    """The mutual k-NN graph must be connected, or the whole expansion is ill-posed."""

    @pytest.mark.parametrize("k", [2, 3, 5, 10])
    def test_the_mst_repair_leaves_exactly_one_component(self, k):
        """Mutual k-NN disconnects readily; three separate things break if it does.

        A multi-dimensional null space makes "remove the trivial constant" ambiguous,
        an isolated vertex has degree zero so ``D^(-1/2)`` is singular, and
        ``log(n - 1)`` stops being the entropy normalizer. Tested on three separated
        blobs, which is the geometry that disconnects most eagerly.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        rng = np.random.default_rng(1)
        X = np.vstack([rng.normal(centre, 0.3, (20, 4)) for centre in (0, 20, 40)])
        W = _mutual_knn_affinity(_pairwise_sq_distances(X), k)
        n_components, _ = connected_components(csr_matrix(W > 0), directed=False)
        assert n_components == 1
        assert (W.sum(axis=1) > 0).all(), "a zero-degree vertex survived the repair"

    def test_the_spectrum_has_exactly_one_trivial_mode(self):
        """Connectivity means multiplicity one at zero, which is what lets us drop by index."""
        rng = np.random.default_rng(2)
        X = np.vstack([rng.normal(centre, 0.3, (20, 4)) for centre in (0, 20, 40)])
        eigenvalues, basis = _laplacian_basis(_mutual_knn_affinity(_pairwise_sq_distances(X), 5))
        assert eigenvalues.size == X.shape[0] - 1
        assert basis.shape == (X.shape[0], X.shape[0] - 1)
        assert eigenvalues[0] > 1e-8, "a second near-zero mode survived: still disconnected"

    def test_the_spectrum_is_bounded_by_two(self):
        """``L_sym``'s range is what makes ``graph_dirichlet`` bounded rather than scale-free."""
        X, _ = _circle(60)
        eigenvalues, _ = _laplacian_basis(_mutual_knn_affinity(_pairwise_sq_distances(X), 6))
        assert eigenvalues.min() >= 0.0
        assert eigenvalues.max() <= 2.0 + 1e-9

    def test_the_affinity_is_invariant_to_a_global_rescaling_of_x(self):
        """Local scaling makes ``sigma_i sigma_j`` carry the units of the squared distance.

        This is the reason the module does not standardize ``X``, so if it ever stops
        holding that decision needs revisiting.
        """
        X, _ = _circle(60)
        baseline = _mutual_knn_affinity(_pairwise_sq_distances(X), 6)
        scaled = _mutual_knn_affinity(_pairwise_sq_distances(X * 1000.0), 6)
        np.testing.assert_allclose(baseline, scaled, rtol=1e-9, atol=1e-12)


class TestTheTrivialModeIsProjectedOut:
    """Centring ``y`` is not enough, and the difference is measurable.

    The null eigenvector of ``L_sym`` is ``D^(1/2) 1``, not ``1``. Centring makes the
    target orthogonal to ``1``, which on a graph with non-uniform degrees leaves a
    real component along ``D^(1/2) 1``. That component sits at ``lambda = 0``, so
    leaving it in would inflate the low-frequency power of every target and shrink
    ``graph_dirichlet`` towards zero.
    """

    def test_a_centred_target_still_has_power_in_the_trivial_mode(self):
        rng = np.random.default_rng(3)
        # Unequal cluster sizes give unequal degrees, which is the whole point.
        X = np.vstack([rng.normal(0, 0.3, (12, 3)), rng.normal(6, 1.5, (48, 3))])
        y = np.r_[np.zeros(12), np.ones(48)]
        W = _mutual_knn_affinity(_pairwise_sq_distances(X), 5)
        degree = W.sum(axis=1)

        inv_sqrt = 1.0 / np.sqrt(degree)
        L = np.eye(len(y)) - (inv_sqrt[:, None] * W * inv_sqrt[None, :])
        _, eigenvectors = np.linalg.eigh(0.5 * (L + L.T))
        trivial = eigenvectors[:, 0]

        centred = _centred(y)
        assert abs(float(centred @ np.ones_like(centred))) < 1e-9, "not centred"
        assert abs(float(trivial @ centred)) > 1e-3, (
            "the trivial mode happens to be orthogonal to this centred target, so the "
            "test geometry no longer demonstrates why the explicit projection is needed"
        )

    def test_the_returned_basis_excludes_it(self):
        """So ``p_j`` is normalized over the ``n - 1`` non-trivial modes and nothing else."""
        X, _ = _circle(60)
        eigenvalues, basis = _laplacian_basis(_mutual_knn_affinity(_pairwise_sq_distances(X), 6))
        assert eigenvalues.size == 59
        assert basis.shape[1] == 59


class TestBounds:
    """Every feature is bounded, so it degrades to a constant rather than to ``inf``.

    ``mfe_features`` documents why that matters: an ``inf`` propagates silently into
    QSage's regressors, where a ``NaN`` would at least be caught as missing. The
    bounds are checked over a sweep of shapes and label structures rather than one
    dataset, because several of them are only reachable in a degenerate regime.
    """

    @pytest.mark.parametrize("n,p", [(20, 2), (40, 3), (60, 200), (30, 1)])
    @pytest.mark.parametrize("structure", ["blocks", "alternating", "random"])
    def test_all_features_stay_in_range(self, n, p, structure):
        rng = np.random.default_rng(abs(hash((n, p, structure))) % 2**32)
        X = rng.normal(size=(n, p))
        y = {
            "blocks": (np.arange(n) < n // 2).astype(int),
            "alternating": np.arange(n) % 2,
            "random": rng.integers(0, 2, n),
        }[structure]
        spectrum = get_task_spectrum_features(X, y, n_permutations=10, stability=True)

        assert 0.0 <= _value(spectrum, "graph_dirichlet") <= 2.0
        for bounded in ("graph_hf_mass", "graph_spec_entropy", "graph_bandwidth90",
                        "pca_tail_signal"):
            assert 0.0 <= _value(spectrum, bounded) <= 1.0, bounded
        assert 0.0 < _value(spectrum, "diffusion_half_life") <= 1.0
        assert _value(spectrum, "h0_fragmentation") >= 0.0
        assert -1.0 <= _value(spectrum, "purity_auc") <= 1.0
        for feature in TASK_FEATURES:
            assert abs(_z(spectrum, feature)) <= Z_CLIP

    def test_the_z_scores_are_clipped_rather_than_allowed_to_explode(self):
        """A tight null on a bounded statistic can put the raw z-score in the thousands.

        Measured 95764 for ``diffusion_half_life`` on two well-separated blobs before
        the clip. With 100 draws no tail probability finer than ~1/100 is resolvable,
        so those digits are noise -- and a column reaching 1e5 dominates any linear
        model exactly as an ``inf`` would.
        """
        rng = np.random.default_rng(4)
        X = np.vstack([rng.normal(0, 1, (60, 5)), rng.normal(12, 1, (60, 5))])
        y = np.r_[np.zeros(60), np.ones(60)].astype(int)
        spectrum = get_task_spectrum_features(X, y)
        assert max(abs(_z(spectrum, f)) for f in TASK_FEATURES) == pytest.approx(Z_CLIP)


class TestThePermutationNull:
    """The negative control, which is what makes the raw values interpretable.

    In sparse high-dimensional data almost nothing is smooth, so random labels look
    high-frequency by default and a raw ``graph_hf_mass`` of 0.5 means nothing on its
    own. Measured during development: a ``p=501`` dataset whose discriminative
    direction was real and low-variance scored ``graph_hf_mass`` 0.53 and
    ``graph_spec_entropy`` 0.85 -- indistinguishable from broadband noise by the raw
    values, and correctly flagged by every z-score coming back below 1.1.
    """

    def test_random_labels_sit_inside_their_own_null(self):
        """The control validating itself: a permuted target must not look atypical."""
        X, _ = _circle()
        spectrum = get_task_spectrum_features(X, _targets()["random"])
        extreme = {f: _z(spectrum, f) for f in TASK_FEATURES if abs(_z(spectrum, f)) > 3.0}
        assert not extreme, f"random labels flagged as atypical: {extreme}"

    def test_structured_labels_sit_far_outside_it(self):
        X, _ = _circle()
        for name in ("smooth", "alternating"):
            spectrum = get_task_spectrum_features(X, _targets()[name])
            assert max(abs(_z(spectrum, f)) for f in TASK_FEATURES) > 4.0, name

    def test_a_smooth_and_an_alternating_target_deviate_in_opposite_directions(self):
        """Both are atypical; the sign is what says which kind of atypical."""
        X, _ = _circle()
        smooth = _z(get_task_spectrum_features(X, _targets()["smooth"]), "graph_hf_mass")
        alternating = _z(
            get_task_spectrum_features(X, _targets()["alternating"]), "graph_hf_mass"
        )
        assert smooth < 0 < alternating, (smooth, alternating)

    def test_the_null_preserves_the_class_proportions(self):
        """A permutation, not a resample -- otherwise imbalance leaks into the null."""
        rng = np.random.default_rng(5)
        X = rng.normal(size=(40, 4))
        y = np.r_[np.zeros(34), np.ones(6)].astype(int)
        spectrum = get_task_spectrum_features(X, y, n_permutations=20)
        assert all(np.isfinite(_z(spectrum, f)) for f in TASK_FEATURES)


class TestPcaTailSignal:
    """Label power outside the PCs carrying 90 % of ``X``'s variance."""

    def test_it_is_high_when_the_discriminative_direction_is_low_variance(self):
        """The case that makes unsupervised dimensionality reduction discard the signal."""
        rng = np.random.default_rng(6)
        n = 120
        y = (np.arange(n) < n // 2).astype(int)
        nuisance = rng.normal(0, 5.0, (n, 200))
        signal = np.where(y == 1, 0.3, -0.3) + rng.normal(0, 0.05, n)
        spectrum = get_task_spectrum_features(
            np.column_stack([nuisance, signal]), y, n_permutations=10
        )
        assert _value(spectrum, "pca_tail_signal") > 0.1

    def test_it_is_zero_when_the_discriminative_direction_is_the_leading_pc(self):
        rng = np.random.default_rng(7)
        n = 120
        y = (np.arange(n) < n // 2).astype(int)
        nuisance = rng.normal(0, 0.5, (n, 200))
        signal = np.where(y == 1, 8.0, -8.0) + rng.normal(0, 0.1, n)
        spectrum = get_task_spectrum_features(
            np.column_stack([nuisance, signal]), y, n_permutations=10
        )
        assert _value(spectrum, "pca_tail_signal") < 0.05

    def test_the_kernel_of_the_gram_matrix_is_excluded(self):
        """Otherwise the feature reads ~1 for every embedded dataset and measures ``p/n``.

        With ``p=2`` and ``n=80`` the row space is 2-dimensional and the other 78
        directions are an arbitrary basis of a zero-variance subspace that ``X``
        cannot express at all. Counting target power there would make this a
        restatement of the dataset's aspect ratio.
        """
        X, _ = _circle(80)
        spectrum = get_task_spectrum_features(X, _targets(80)["random"], n_permutations=10)
        assert _value(spectrum, "pca_tail_signal") <= 1.0
        assert _value(spectrum, "pca_tail_signal") < 0.9


class TestRobustnessAcrossK:
    """A geometric property has to survive a modest change of neighbourhood scale."""

    def test_the_reported_value_is_the_median_over_k(self):
        X, _ = _circle(80)
        y = _targets(80)["smooth"]
        per_k = [
            get_task_spectrum_features(X, y, neighbours=[k], n_permutations=0)[
                f"{TASK_COLUMN_PREFIX}graph_dirichlet"
            ]
            for k in GRAPH_NEIGHBOURS
        ]
        aggregated = get_task_spectrum_features(X, y, n_permutations=0)
        assert aggregated[f"{TASK_COLUMN_PREFIX}graph_dirichlet"] == pytest.approx(
            float(np.median(per_k))
        )

    def test_stability_reports_one_cv_per_graph_feature_and_none_for_the_rest(self):
        """``pca_tail_signal``, ``h0_fragmentation`` and ``purity_auc`` have no ``k`` in them."""
        X, _ = _circle(80)
        spectrum = get_task_spectrum_features(
            X, _targets(80)["smooth"], n_permutations=0, stability=True
        )
        cv = {name for name in spectrum if name.endswith("_cv_k")}
        assert cv == {f"{TASK_COLUMN_PREFIX}{f}_cv_k" for f in GRAPH_FEATURES}

    def test_a_single_k_gives_zero_variation_which_is_why_not_to_pass_one(self):
        X, _ = _circle(80)
        spectrum = get_task_spectrum_features(
            X, _targets(80)["smooth"], neighbours=[7], n_permutations=0, stability=True
        )
        assert all(
            spectrum[f"{TASK_COLUMN_PREFIX}{f}_cv_k"] == 0.0 for f in GRAPH_FEATURES
        )

    def test_k_is_clipped_rather_than_raising_on_a_small_dataset(self):
        """``n=8`` cannot support ``k=20``; the block stays defined instead of failing."""
        rng = np.random.default_rng(8)
        spectrum = get_task_spectrum_features(
            rng.normal(size=(8, 3)), [0, 0, 0, 0, 1, 1, 1, 1], n_permutations=10
        )
        assert len(spectrum) == 16
        assert all(np.isfinite(v) for v in spectrum.values())


class TestDeterminism:
    """Two runs on identical input must agree, or the whole meta-dataset is noise."""

    def test_the_same_seed_reproduces_every_column(self):
        rng = np.random.default_rng(9)
        X, y = rng.normal(size=(60, 20)), rng.integers(0, 2, 60)
        first = get_task_spectrum_features(X, y, random_state=3)
        second = get_task_spectrum_features(X, y, random_state=3)
        assert first == second

    def test_the_seed_reaches_only_the_permutation_null(self):
        """The features themselves are deterministic functions of ``(X, y)``."""
        rng = np.random.default_rng(10)
        X, y = rng.normal(size=(60, 20)), rng.integers(0, 2, 60)
        first = get_task_spectrum_features(X, y, random_state=3)
        second = get_task_spectrum_features(X, y, random_state=4)
        assert all(first[f"{TASK_COLUMN_PREFIX}{f}"] == second[f"{TASK_COLUMN_PREFIX}{f}"]
                   for f in TASK_FEATURES)
        assert any(first[f"{TASK_COLUMN_PREFIX}{f}_z"] != second[f"{TASK_COLUMN_PREFIX}{f}_z"]
                   for f in TASK_FEATURES)

    def test_label_encoding_does_not_matter(self):
        """``{-1,+1}``, ``{0,1}`` and strings describe the same partition, so must agree."""
        rng = np.random.default_rng(11)
        X = rng.normal(size=(50, 6))
        partition = rng.integers(0, 2, 50)
        as_signs = np.where(partition == 1, 1, -1)
        as_strings = np.where(partition == 1, "case", "control")
        baseline = get_task_spectrum_features(X, partition, n_permutations=10)
        for alternative in (as_signs, as_strings):
            # approx, not equality: {0,1} centres to +/-0.5 where {-1,+1} centres to
            # +/-1, so the expansions differ by a scale that p_j normalizes away
            # mathematically but not bitwise.
            emitted = get_task_spectrum_features(X, alternative, n_permutations=10)
            assert emitted == pytest.approx(baseline, rel=1e-9, abs=1e-12)


class TestDegenerateInput:
    """Refuse what is undefined; stay defined for what is merely awkward."""

    @pytest.mark.parametrize(
        "labels,match",
        [
            ([1] * 20, "constant target"),
            ([0, 1], "same observations"),
        ],
    )
    def test_undefined_input_is_refused_with_the_reason(self, labels, match):
        rng = np.random.default_rng(12)
        with pytest.raises(ValueError, match=match):
            get_task_spectrum_features(rng.normal(size=(20, 3)), labels)

    def test_too_few_observations_is_refused(self):
        rng = np.random.default_rng(13)
        with pytest.raises(ValueError, match=f"at least {MIN_SAMPLES}"):
            get_task_spectrum_features(rng.normal(size=(3, 3)), [0, 1, 1])

    def test_a_non_finite_x_is_refused_rather_than_poisoning_every_distance(self):
        """One NaN makes the entire distance matrix non-finite, so every column dies."""
        rng = np.random.default_rng(14)
        X = rng.normal(size=(20, 3))
        X[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or inf"):
            get_task_spectrum_features(X, [0] * 10 + [1] * 10)

    def test_one_permutation_cannot_give_a_null_sd(self):
        rng = np.random.default_rng(15)
        with pytest.raises(ValueError, match="cannot give a null standard deviation"):
            get_task_spectrum_features(
                rng.normal(size=(20, 3)), [0] * 10 + [1] * 10, n_permutations=1
            )

    @pytest.mark.parametrize(
        "name,build",
        [
            ("duplicate rows drive sigma to zero", "duplicates"),
            ("every row identical", "constant"),
            ("severe class imbalance", "imbalanced"),
            ("more than two classes", "multiclass"),
        ],
    )
    def test_awkward_input_stays_finite(self, name, build):
        """None of these is undefined, so none may produce a NaN column."""
        rng = np.random.default_rng(16)
        if build == "duplicates":
            X = rng.normal(size=(30, 4))
            X[5:16] = X[4]
            y = [0] * 15 + [1] * 15
        elif build == "constant":
            X, y = np.ones((20, 4)), [0] * 10 + [1] * 10
        elif build == "imbalanced":
            X, y = rng.normal(size=(30, 4)), [0] * 28 + [1, 1]
        else:
            X, y = rng.normal(size=(30, 4)), [0] * 10 + [1] * 10 + [2] * 10
        spectrum = get_task_spectrum_features(X, y, n_permutations=10, stability=True)
        bad = {k: v for k, v in spectrum.items() if not np.isfinite(v)}
        assert not bad, f"{name}: {bad}"


class TestIntegrationWithEvaluate:
    """``evaluate`` ships the block, and the schema helpers account for it."""

    @staticmethod
    def _evaluated(**kwargs):
        rng = np.random.default_rng(17)
        X = pd.DataFrame(rng.normal(size=(40, 30)))
        y = (np.arange(40) % 2).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return evaluate(X, y, "synthetic.csv", **kwargs)

    def test_the_block_is_present_prefixed_and_numeric(self):
        frame = self._evaluated()
        block = task_columns(frame)
        assert len(block) == 16
        assert all(name.startswith(TASK_COLUMN_PREFIX) for name in block)
        assert set(block) == set(task_column_names())
        assert frame[block].apply(pd.to_numeric, errors="coerce").notna().all().all()

    def test_it_can_be_switched_off_for_a_tall_dataset(self):
        """The cost is ``O(n^3)`` in the sample count, where the pyMFE block is not."""
        assert task_columns(self._evaluated(task_spectrum=False)) == []

    def test_task_kwargs_reaches_the_block(self):
        frame = self._evaluated(task_kwargs={"n_permutations": 0, "stability": True})
        block = task_columns(frame)
        assert set(block) == set(task_column_names(n_permutations=0, stability=True))

    def test_the_schema_helpers_return_the_block_as_features(self):
        frame = self._evaluated()
        schema, features = detect_complexity_schema(frame)
        assert schema == "pymfe"
        assert set(task_columns(frame)) <= set(features)
        assert set(NATIVE_COMPLEXITY_COLUMNS) <= set(features)
        assert set(task_columns(frame)) <= set(complexity_feature_columns(frame.columns))

    def test_a_pymfe_table_without_the_block_still_trains(self):
        """Every RawDataEvaluation.csv written before this block existed is that table."""
        frame = self._evaluated(task_spectrum=False)
        schema, features = detect_complexity_schema(frame)
        assert schema == "pymfe"
        assert features and not any(str(f).startswith(TASK_COLUMN_PREFIX) for f in features)

    def test_mixing_a_pre_block_table_with_a_post_block_one_is_refused(self):
        """The same silent-zeros hazard the legacy/pyMFE guard exists for, one level down.

        Concatenating an older results table with a fresh run leaves the older rows
        NaN across the whole task block, and training maps those to zeros -- a value
        the target spectrum can genuinely take, so nothing downstream can notice.
        """
        mixed = pd.concat(
            [self._evaluated(task_spectrum=False), self._evaluated()], ignore_index=True
        )
        with pytest.raises(ValueError, match="two generations of the pyMFE schema"):
            detect_complexity_schema(mixed)

    def test_the_refusal_counts_the_rows(self):
        mixed = pd.concat(
            [self._evaluated(task_spectrum=False), self._evaluated()], ignore_index=True
        )
        with pytest.raises(ValueError) as failure:
            detect_complexity_schema(mixed)
        assert "1 of 2" in str(failure.value)

    def test_a_block_that_is_empty_in_every_row_is_dropped_not_trained_on(self):
        """Unlike the mixed case there is no ambiguity, so it needs no exception."""
        frame = self._evaluated()
        frame[task_columns(frame)] = np.nan
        schema, features = detect_complexity_schema(frame)
        assert schema == "pymfe"
        assert not any(str(f).startswith(TASK_COLUMN_PREFIX) for f in features)
