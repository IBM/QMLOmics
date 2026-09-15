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

"""Dataset complexity evaluation for QBioCode.

:func:`evaluate` summarizes one tabular (samples x features) dataset as a one-row
:class:`pandas.DataFrame` of complexity measures. Those measures are the feature
matrix QSage trains on to predict which model will win on a dataset, so what they
measure bounds what QSage can learn.

The bulk of them come from `pyMFE <https://github.com/ealcobaca/pymfe>`_, via
:func:`qbiocode.evaluation.mfe_features.get_mfe_features`, which contributes the
Lorena et al. (2019) data-complexity suite, the landmarking family, decision-tree
model-based measures, clustering and concept measures, and the standard
statistical and information-theoretic ones. That module documents which of pyMFE's
~105 meta-features QBioCode uses and why the rest are excluded.

A third block, ``task.``-prefixed, comes from
:func:`qbiocode.evaluation.task_spectrum.get_task_spectrum_features`. Everything
pyMFE and this module contribute describes ``X`` alone, or describes ``y`` only
through a classifier's view of it; that block describes where ``y`` sits in the
*geometric spectrum* of ``X``, which is what separates a structured
high-frequency target (parity, checkerboard, alternating-sign) from ordinary
difficulty and from broadband noise. See that module's docstring.

This module keeps the measures pyMFE does **not** cover:

============================== =============================================
``Intrinsic_Dimension``        ``skdim`` lPCA. pyMFE's ``t3``/``t4`` are also
                               PCA-based but use a 95 %-explained-variance
                               criterion, a different estimator; QSage's
                               ``SLGH`` feature is defined on this column.
``Condition number``           No pyMFE equivalent.
``Fisher Discriminant Ratio``  See the note below -- pyMFE does not cover it.
``Coefficient of Variation %`` pyMFE has ``mean`` and ``sd`` separately but
``std_co_of_v``                not their ratio.
``# Low variance features``    No pyMFE equivalent.
``# Non-zero entries``         pyMFE's ``sparsity`` counts distinct values per
                               attribute, which is a different quantity.
``Mean Log Kernel Density``    pyMFE's ``density`` is graph-based, unrelated.
``Isomap Reconstruction Error`` No pyMFE equivalent.
``Fractal dimension``          Higuchi. No pyMFE equivalent.
============================== =============================================

**Why ``Fisher Discriminant Ratio`` is still computed here.** It looks like
pyMFE's ``f1`` covers it, and it does not. ``f1`` is ``1 / (1 + r_j)`` evaluated
*per feature* ``j`` and then summarized, so it is univariate and it is inverted --
a larger ``f1`` means a *harder* problem. :func:`get_fdr` is the multivariate
``trace(S_between) / trace(S_within)``, where larger means easier. pyMFE's
faithful analogue is ``f1v``, the directional-vector version, which raises
``ValueError`` on every dataset under NumPy >= 2 (see ``mfe_features``). Feeding
``f1`` into :func:`qbiocode.apps.sage.sage.calculate_SLGH` in place of this column
would flip the sign of that term and silently change what QSage learns, so the
native measure stays and ``mfe.f1.mean`` ships alongside it as the univariate
complement.
"""

import warnings

import hfda
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull as CH
from skdim import id
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import Isomap
from sklearn.neighbors import KernelDensity

from .mfe_features import MFE_COLUMN_PREFIX, get_mfe_features
from .task_spectrum import TASK_COLUMN_PREFIX, get_task_spectrum_features

#: Complexity columns computed in this module rather than by pyMFE, in output
#: order. ``Dataset`` is metadata and is deliberately not part of this tuple.
#:
#: :class:`qbiocode.apps.sage.sage.QuantumSage` uses this to assemble its feature
#: list, so keep it in step with what :func:`evaluate` actually emits.
NATIVE_COMPLEXITY_COLUMNS = (
    "Intrinsic_Dimension",
    "Condition number",
    "Fisher Discriminant Ratio",
    "# Non-zero entries",
    "# Low variance features",
    "Coefficient of Variation %",
    "std_co_of_v",
    "Mean Log Kernel Density",
    "Isomap Reconstruction Error",
    "Fractal dimension",
)


#: The dataset-complexity columns QProfiler wrote before the pyMFE integration.
#:
#: Still supported because ``tutorial/QSage/data/qprofiler_benchmarks.csv`` -- the
#: 576-row table the QSage tutorial trains on -- carries exactly these, and cannot
#: be regenerated from the repository: it covers ``class_data-{1..16}.csv`` and only
#: ``class_data-1``, ``-2`` and ``-3`` are committed. A table in this schema keeps
#: training QSage exactly as it did before.
LEGACY_COMPLEXITY_COLUMNS = [
    '# Features', '# Samples',
    'Feature_Samples_ratio', 'Intrinsic_Dimension', 'Condition number',
    'Fisher Discriminant Ratio', 'Total Correlations', 'Mutual information',
    '# Non-zero entries', '# Low variance features', 'Variation', 'std_var',
    'Coefficient of Variation %', 'std_co_of_v', 'Skewness', 'std_skew',
    'Kurtosis', 'std_kurt', 'Mean Log Kernel Density',
    'Isomap Reconstruction Error', 'Fractal dimension', 'Entropy',
    'std_entropy',
]

#: Column holding the sample count, per schema. ``calculate_SLGH`` needs it.
SAMPLE_COUNT_COLUMN = {'legacy': '# Samples', 'pymfe': 'mfe.nr_inst'}


def detect_complexity_schema(columns):
    """Identify which dataset-complexity schema a results table carries.

    QProfiler's complexity block was replaced by a pyMFE-backed one (see
    :mod:`qbiocode.evaluation.mfe_features`), so a results table in the wild is in
    one of two schemas. Rather than pick one and orphan the other, QSage reads
    whichever it is handed: tables written before the change still train, and
    tables written after also carry the landmarking, model-based and Lorena
    complexity families that the legacy block lacked entirely.

    Args:
        columns (Iterable[str] or pandas.DataFrame): Column names of the results
            table, or the table itself. **Prefer passing the frame.** Column names
            alone cannot detect a table that concatenates rows of both schemas --
            see the note below -- and only the frame carries the per-row evidence
            that does.

    Returns:
        tuple: ``(schema, feature_columns)`` where ``schema`` is ``'pymfe'`` or
        ``'legacy'`` and ``feature_columns`` is the list of complexity columns to
        train on, in table order.

    Raises:
        ValueError: If the table matches neither schema, carries pyMFE columns
            without the natively-computed ones, or -- when a frame is passed --
            concatenates rows of both schemas.

    Note:
        **Why a frame is better than a column list.** All ten
        :data:`NATIVE_COMPLEXITY_COLUMNS` are a subset of
        :data:`LEGACY_COMPLEXITY_COLUMNS`, so concatenating a legacy table with
        pyMFE rows produces a frame that satisfies the ``'pymfe'`` test on names
        alone -- pyMFE columns are present, and so is every native column, because
        the legacy rows supplied them. The legacy rows then hold ``NaN`` for all 115
        pyMFE columns, and :meth:`QuantumSage.train_sub_sages` turns those into
        zeros, so the model trains on a mostly-zero feature block while the 13
        legacy-only features most of its rows actually have are dropped. Measured on
        the committed 576-row benchmark table plus 8 fresh rows: 98.6 % of rows were
        entirely ``NaN`` across the pyMFE block, and nothing warned. Passing the
        frame lets that be caught and refused.
    """
    # A DataFrame is iterable over its column names, so `list()` handles both inputs;
    # keep the frame itself when we were given one, for the row-level check below.
    frame = columns if hasattr(columns, "isna") and hasattr(columns, "columns") else None
    columns = list(columns.columns) if frame is not None else list(columns)
    present = set(columns)

    # A pyMFE-prefixed column is unambiguous -- nothing else in a results table
    # uses that prefix -- so it decides the schema on its own. The same holds for the
    # task-spectrum prefix, which a table written by a current evaluate() also
    # carries; either one alone is enough to rule out the legacy schema.
    mfe = [name for name in columns if str(name).startswith(MFE_COLUMN_PREFIX)]
    task = [name for name in columns if str(name).startswith(TASK_COLUMN_PREFIX)]
    if mfe or task:
        native = [name for name in NATIVE_COMPLEXITY_COLUMNS if name in present]
        missing = [name for name in NATIVE_COMPLEXITY_COLUMNS if name not in present]
        if missing:
            raise ValueError(
                f"Table carries {len(mfe)} pyMFE complexity columns but is missing "
                f"{len(missing)} of the natively-computed ones: {missing}. Both halves "
                "come from a single qbiocode.evaluation.evaluate() call, so a table "
                "with one and not the other has been subset or concatenated across "
                "incompatible QProfiler versions."
            )
        # Rows that hold NaN across the ENTIRE pyMFE block did not come from a
        # pyMFE-era evaluate() call -- they are legacy rows carried in by a
        # concatenation. Column names cannot show this (see the Note above), and the
        # consequence is silent: those rows train as zeros. Refuse instead.
        if frame is not None and len(frame):
            legacy_rows = int(frame[mfe].isna().all(axis=1).sum())
            if legacy_rows:
                raise ValueError(
                    f"Table mixes both complexity schemas: {legacy_rows} of "
                    f"{len(frame)} rows are NaN across all {len(mfe)} pyMFE columns, "
                    f"so they predate the pyMFE block and only the remaining "
                    f"{len(frame) - legacy_rows} carry it.\n\n"
                    "This is almost always a concatenation of a pre-pyMFE results "
                    "table with a fresh QProfiler run. It cannot be trained on as "
                    "either schema: read as 'pymfe' the legacy rows contribute a "
                    "block of NaN that training silently turns into zeros, and the "
                    "legacy-only features the majority of rows do have are dropped.\n\n"
                    "Either train on one schema at a time (filter the rows), or "
                    "re-run QProfiler over the older datasets so every row carries "
                    "the current complexity block."
                )
        # The task-spectrum block postdates the pyMFE one, so a table can legitimately
        # carry pyMFE columns and no task columns -- every RawDataEvaluation.csv
        # written before it existed does. That case is fine: `task` is empty and the
        # feature list is the pyMFE one, exactly as before. What is NOT fine is a
        # concatenation of a pre-task table with a post-task one, which reproduces the
        # legacy/pyMFE hazard documented in the Note above one level down -- the older
        # rows hold NaN across the whole task block and train as zeros. Same detection,
        # same refusal.
        if task and frame is not None and len(frame):
            blank = frame[task].isna().all(axis=1)
            n_blank = int(blank.sum())
            if 0 < n_blank < len(frame):
                raise ValueError(
                    f"Table mixes two generations of the pyMFE schema: {n_blank} of "
                    f"{len(frame)} rows are NaN across all {len(task)} "
                    f"{TASK_COLUMN_PREFIX} target-spectrum columns, so they were "
                    f"written before that block existed, and only the remaining "
                    f"{len(frame) - n_blank} carry it.\n\n"
                    "This is a concatenation of an older QProfiler results table with a "
                    "fresh run. Training on it silently turns the missing block into "
                    "zeros for the older rows, which is a value the target spectrum can "
                    "genuinely take -- so nothing downstream can notice.\n\n"
                    "Either train on one generation at a time (filter the rows), or "
                    "re-run QProfiler over the older datasets so every row carries the "
                    "current block."
                )
            if n_blank == len(frame):
                # Present but empty in every row: nothing to train on, and unlike the
                # mixed case there is no ambiguity about what the table is. Drop the
                # block rather than feed QSage a column of zeros.
                task = []
        return 'pymfe', native + mfe + task

    legacy_missing = [name for name in LEGACY_COMPLEXITY_COLUMNS if name not in present]
    if not legacy_missing:
        return 'legacy', list(LEGACY_COMPLEXITY_COLUMNS)

    raise ValueError(
        "Table matches neither dataset-complexity schema. Expected either "
        f"{MFE_COLUMN_PREFIX}-prefixed columns from the current "
        "qbiocode.evaluation.evaluate(), or all "
        f"{len(LEGACY_COMPLEXITY_COLUMNS)} legacy columns (missing "
        f"{len(legacy_missing)}: {legacy_missing}). Pass a QProfiler ModelResults.csv "
        "or compiled_results.csv."
    )


def complexity_feature_columns(columns):
    """The complexity feature columns in ``columns``, tolerating an unknown schema.

    The permissive counterpart to :func:`detect_complexity_schema`. Use it where a
    narrower-than-expected answer is acceptable and an exception is not -- notably
    :func:`qbiocode.visualization.compute_results_correlation`, which correlates
    whatever complexity columns a results table happens to carry.

    Prefer :func:`detect_complexity_schema` where selecting the wrong columns would be
    a correctness problem rather than a cosmetic one, as it is when training QSage.

    Args:
        columns (Iterable[str]): Column names of a results table.

    Returns:
        list: Complexity feature columns, in table order. Empty if none are recognized.
    """
    columns = list(columns)
    present = set(columns)
    mfe = [name for name in columns if str(name).startswith(MFE_COLUMN_PREFIX)]
    task = [name for name in columns if str(name).startswith(TASK_COLUMN_PREFIX)]
    if mfe or task:
        return [n for n in NATIVE_COMPLEXITY_COLUMNS if n in present] + mfe + task
    return [n for n in LEGACY_COMPLEXITY_COLUMNS if n in present]


def get_intrinsic_dim(df):
    """Get intrinsic dimension of the data using lPCA from skdim.

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns

    Returns:
        float: Intrinsic dimension of the data
    """
    # Intrinsic dimension, calculated via scikit-dimension's PCA method
    pca = id.lPCA()  # Initialize the PCA estimator from skdim
    pca.fit(df)  # Fit the estimator to your data
    return pca.dimension_


def get_condition_number(df):
    """Get the condition number of a matrix.

    A high condition number indicates that the matrix is ill-conditioned and
    can produce large output errors even for small input perturbations. A low
    condition number indicates a more stable matrix.

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns

    Returns:
        float: Condition number of the matrix represented in ``df``.
    """
    return np.linalg.cond(df)


def get_fdr(df, y):
    """Calculate Fisher Discriminant Ratio for a given dataset.

    This is the multivariate ``trace(S_between) / trace(S_within)``, where a
    larger value means the classes are more separable. It is deliberately not
    replaced by pyMFE's ``f1`` -- see the module docstring for why.

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns
        y (int): supervised binary class label

    Returns:
        float: Fisher Discriminant ratio
    """
    X = df.values
    class_labels = np.unique(y)
    n_classes = len(class_labels)
    FDR = 0

    if n_classes != 2:
        warnings.warn("WARNING: Fisher Discriminant Ratio is only defined for binary classes. ")
    else:
        mean1 = np.mean(X[y == class_labels[0]], axis=0)  # mean for class1
        mean2 = np.mean(X[y == class_labels[1]], axis=0)  # mean for class2

        # calculate within-class scatter matrices
        scatter_within = np.zeros((X.shape[1], X.shape[1]))
        for label in class_labels:
            X_class = X[y == label]
            scatter_within += np.cov(X_class.T)

        # calculate between-class scatter matrix
        scatter_between = np.outer(mean1 - mean2, mean1 - mean2)

        # compute FDR
        FDR = np.trace(scatter_between) / np.trace(scatter_within)

    return FDR


def get_coefficient_var(df):
    """Get coefficient of variance

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns

    Returns:
        avg_co_of_v (float): Mean coefficient of variance
        std_var (float): Standard deviation of coefficient of variance
    """
    co_of_v = (df.std() / df.mean()) * 100
    avg_co_of_v = co_of_v.mean()
    std_co_of_v = co_of_v.std()

    return avg_co_of_v, std_co_of_v


def get_nnz(df):
    """Calculate nonzero values in the data

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns

    Returns:
        int: nonzero count
    """
    return np.count_nonzero(df.values)


def get_low_var_features(df, num_features):
    """Calculate get count of low variance features

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns
        num_features (int): number of features in the dataset

    Raises:
        ValueError: If no feature is strong enough to keep

    Returns:
        int: count of features with low variance
    """

    threshold = np.percentile(df.var(), 25)

    try:
        low_var_features = num_features - VarianceThreshold(threshold).fit(df).get_support().sum()
    except ValueError:
        print("No feature is strong enough to keep")
        low_var_features = None

    return low_var_features


def get_log_density(df):
    """Calculate the mean log density of the data

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns

    Returns:
        float: mean log kernel density
    """
    kde = KernelDensity(bandwidth=0.2, kernel="gaussian").fit(
        df
    )  # Create a KernelDensity estimator and fit the estimator to the data
    log_density = kde.score_samples(df)

    return log_density.mean()


def get_fractal_dim(df, k_max):
    """Calculate the fractal dimension of the data using Higuchi's method

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns
        k_max (int): Maximum number of k values to use in the calculation

    Returns:
        float: Fractal dimension of the data
    """
    FD = hfda.measure(df, k_max)

    return FD


def get_volume(df):
    """Get volume of the data from Convex Hull

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns

    Returns:
        volume (float): Volume of the space spanned by the features of the data
    """

    vol = 0
    if df.shape[0] <= df.shape[1]:
        warnings.warn("Convex Hull requires number of observations > number of features")
    else:
        vol = CH(df, qhull_options="QJ").volume

    return vol


def get_complexity(df, n_neighbors=10, n_components=2):
    """Measure manifold complexity via Isomap's geodesic-vs-Euclidean residual.

    This function computes the reconstruction error of the Isomap algorithm, which
    serves as an indicator of the complexity of the manifold represented by the data.

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns
        n_neighbors: Number of neighbors for the Isomap algorithm. Default value 10
        n_components: Number of components (dimensions) for Isomap projection.  Default value 2

    Returns:
        float: The reconstruction error of the fitted Isomap model -- the
            residual between geodesic and Euclidean distances, which indicates
            the complexity of the manifold.
    """
    # Both arguments are forwarded. They used to be accepted and then ignored in
    # favour of hardcoded 10 and 2, so passing anything else silently did nothing.
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    isomap.fit(df.values)

    # reconstruction error - an indicator of complexity
    reconstruction_error = isomap.reconstruction_error()

    return reconstruction_error


def evaluate(df, y, file, random_state=0, mfe_features=None, task_spectrum=True,
             task_kwargs=None):
    """Summarize a dataset's complexity as a one-row DataFrame.

    Combines the measures pyMFE does not provide (computed by the helpers in this
    module) with the curated pyMFE meta-feature block from
    :func:`qbiocode.evaluation.mfe_features.get_mfe_features`. The result is
    intended to be concatenated across datasets and correlated against model
    performance -- it is what QProfiler writes to ``RawDataEvaluation.csv`` and
    what QSage trains on.

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows,
            features in columns. Non-numeric columns are dropped.
        y (int): supervised binary class label
        file (str): Name of the dataset file, recorded in the ``Dataset`` column
            for identification in the summary DataFrame
        random_state (int): Seed for the pyMFE block, whose landmarking measures
            cross-validate and whose clustering measures run k-means. Fixed by
            default so two runs on identical input agree. Default 0.
        mfe_features (Iterable[str], optional): Override the pyMFE measure list.
            Defaults to the curated
            :data:`qbiocode.evaluation.mfe_features.MFE_FEATURES`.
        task_spectrum (bool): Include the ``task.`` target-spectrum block from
            :func:`qbiocode.evaluation.task_spectrum.get_task_spectrum_features`.
            Default True. It is cheap next to the pyMFE block -- measured 0.18 s at
            ``n=100, p=20000``, against about 50 s for the curated pyMFE set at the
            same shape -- but it is ``O(n^3)`` in the sample count where pyMFE is
            not, so it is worth switching off for an unusually tall dataset
            (measured 12 s at ``n=800``).
        task_kwargs (dict, optional): Forwarded to
            :func:`~qbiocode.evaluation.task_spectrum.get_task_spectrum_features`,
            for the neighbourhood sizes, the permutation count, and the
            ``stability=True`` coefficient-of-variation columns. ``random_state`` is
            passed through from this call and should not be duplicated here.

    Returns:
        pandas.DataFrame: One row. Columns are ``Dataset``, then
        :data:`NATIVE_COMPLEXITY_COLUMNS`, then the pyMFE block prefixed ``mfe.``,
        then -- unless ``task_spectrum`` is False -- the target-spectrum block
        prefixed ``task.``.
    """
    # Select only numeric columns from the DataFrame
    df_numeric = df.select_dtypes(include=[np.number])
    y = np.asarray(y).ravel()

    n_features = df_numeric.shape[1]

    # ---- measures with no pyMFE equivalent (see module docstring) ----
    intrinsic_dim = get_intrinsic_dim(df_numeric)
    condition_number = get_condition_number(df_numeric)
    fdr = get_fdr(df_numeric, y)
    count_nonzero = get_nnz(df)
    num_low_variance_features = get_low_var_features(df_numeric, n_features)
    avg_co_of_v, std_co_of_v = get_coefficient_var(df_numeric)
    mean_log_density = get_log_density(df_numeric)
    complexity = get_complexity(df_numeric)
    fractal_dim = get_fractal_dim(df_numeric, k_max=5)

    summary = {
        "Dataset": file,
        "Intrinsic_Dimension": intrinsic_dim,
        "Condition number": condition_number,
        "Fisher Discriminant Ratio": fdr,
        "# Non-zero entries": count_nonzero,
        "# Low variance features": num_low_variance_features,
        "Coefficient of Variation %": avg_co_of_v,
        "std_co_of_v": std_co_of_v,
        "Mean Log Kernel Density": mean_log_density,
        "Isomap Reconstruction Error": complexity,
        "Fractal dimension": fractal_dim,
    }

    # ---- the pyMFE block ----
    # This supplies the dataset shape (mfe.nr_inst, mfe.nr_attr,
    # mfe.attr_to_inst), the distribution moments, the correlation and
    # information-theoretic measures, and the complexity, landmarking,
    # model-based, clustering and concept families -- everything the hand-rolled
    # implementations used to cover, plus what they did not.
    summary.update(
        get_mfe_features(
            df_numeric,
            y,
            random_state=random_state,
            features=mfe_features,
        )
    )

    # ---- the target-spectrum block ----
    # The only block that describes y RELATIVE TO the geometry of X rather than
    # describing either one alone. It is what distinguishes a structured
    # high-frequency target from a merely hard one; see the task_spectrum module
    # docstring for why that distinction is the point of the block.
    if task_spectrum:
        summary.update(
            get_task_spectrum_features(
                df_numeric,
                y,
                random_state=random_state,
                **(task_kwargs or {}),
            )
        )

    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    transposed = summary_df.T

    # Every measure column is numeric, but building the frame from a dict that also
    # holds the `Dataset` string makes the whole row object-dtype. That used to be
    # invisible because the only consumer wrote it straight to CSV and read it back,
    # which restores the dtypes -- but it means an in-memory caller gets a frame
    # whose float columns reject `numpy` ufuncs ("loop of ufunc does not support
    # argument 0 of type numpy.float64"), which is what QSage's calculate_SLGH does.
    # Coerce here so the returned frame is usable without a CSV round-trip.
    # `errors="coerce"` maps the None that get_low_var_features returns when no
    # feature clears the variance threshold to NaN.
    measures = [name for name in transposed.columns if name != "Dataset"]
    transposed[measures] = transposed[measures].apply(pd.to_numeric, errors="coerce")

    return transposed


def mfe_columns(frame):
    """The pyMFE-contributed column names present in ``frame``, in order.

    Args:
        frame (pandas.DataFrame): Any frame produced by, or derived from,
            :func:`evaluate`.

    Returns:
        list: Column names carrying the ``mfe.`` prefix.
    """
    return [name for name in frame.columns if str(name).startswith(MFE_COLUMN_PREFIX)]


def task_columns(frame):
    """The target-spectrum column names present in ``frame``, in order.

    The counterpart to :func:`mfe_columns` for the ``task.`` block. Empty for a
    frame produced with ``task_spectrum=False``, or by an ``evaluate()`` predating
    that block.

    Args:
        frame (pandas.DataFrame): Any frame produced by, or derived from,
            :func:`evaluate`.

    Returns:
        list: Column names carrying the ``task.`` prefix.
    """
    return [name for name in frame.columns if str(name).startswith(TASK_COLUMN_PREFIX)]
