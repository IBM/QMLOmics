"""Correlation analysis and publication-quality figures for QProfiler results."""

import logging
import os
import re
import sys
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import spearmanr

from qbiocode.evaluation.dataset_evaluation import complexity_feature_columns
from qbiocode.evaluation.mfe_features import MFE_COLUMN_PREFIX
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

#: Publication defaults for scientific journals, applied **per figure** through
#: :func:`matplotlib.pyplot.rc_context`.
#:
#: These were previously assigned straight into ``plt.rcParams`` at module scope,
#: so ``import qbiocode`` -- which imports this module transitively -- silently
#: reconfigured the importing program's matplotlib: font family, tick geometry,
#: spine visibility and a 600-dpi ``savefig`` default leaked into every unrelated
#: figure the caller drew afterwards, with nothing in the traceback or the call
#: stack to attribute it to. Call :func:`publication_style` to opt in globally.
PUBLICATION_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
}

#: Model-name prefixes drawn as quantum rather than classical.
_QML_MODELS = ("QNN", "PQK", "VQC", "QSVC")

_SAVEFIG_KWARGS = {
    "dpi": 600,
    "bbox_inches": "tight",
    "facecolor": "white",
    "edgecolor": "none",
}


class CorrelationFigures(NamedTuple):
    """The three figures :func:`plot_results_correlation` produces.

    Returned so a caller can compose, restyle or save them without re-running the
    computation, and so a notebook can display exactly the one it wants. Every
    figure is closed before the function returns -- a `Figure` object stays fully
    usable after `plt.close`, including `fig.savefig` and Jupyter's inline
    display, it is only removed from pyplot's global figure manager. That is what
    keeps figures from accumulating when QProfiler calls this once per iteration.
    """

    scatter: Figure
    scatter_ax: Axes
    clustered_heatmap: "sns.matrix.ClusterGrid"
    ordered_heatmap: "sns.matrix.ClusterGrid"


def publication_style() -> dict:
    """Return a copy of :data:`PUBLICATION_STYLE` for use as a matplotlib style.

    The plotting functions here apply it per figure, so callers need this only to
    style figures of their own::

        import matplotlib.pyplot as plt
        import qbiocode as qbc

        with plt.rc_context(qbc.visualization.publication_style()):
            fig, ax = plt.subplots()

    Mutating the returned dict does not affect the module default.
    """
    return {k: (list(v) if isinstance(v, list) else v) for k, v in PUBLICATION_STYLE.items()}


def _can_show() -> bool:
    """Whether ``plt.show()`` would display a window rather than do nothing or fail.

    Library code must never block a headless run, and the previous unconditional
    ``plt.show()`` did exactly that: it hung a batch QProfiler run under a GUI
    backend, and under ``Agg`` it emitted ``UserWarning: FigureCanvasAgg is
    non-interactive, so cannot be shown`` three times per call.

    Note this deliberately does *not* call ``matplotlib.use("Agg")``. Forcing a
    backend at import time is the same class of hidden global mutation this module
    just stopped doing with ``rcParams``, and it is unnecessary: matplotlib already
    resolves its backend lazily and falls back to ``Agg`` when no GUI toolkit is
    importable. Asking whether showing works is enough, and it leaves the caller's
    backend choice alone.
    """
    backend = mpl.get_backend().lower()
    if backend in ("agg", "pdf", "ps", "svg", "cairo", "template"):
        return False
    # An inline/widget backend in Jupyter has no window server requirement.
    if "inline" in backend or "ipympl" in backend or "nbagg" in backend or "widget" in backend:
        return True
    # A remaining GUI backend needs somewhere to draw. Windows and macOS always
    # have a window server when a GUI backend resolved at all; X11/Wayland do not.
    if os.name == "posix" and sys.platform != "darwin":
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _derive_path(save_file_path: str, suffix: str) -> str:
    """Insert ``suffix`` before the extension of ``save_file_path``.

    Replaces ``re.sub(".pdf", "_heatmap.pdf", save_file_path)``, which had two
    faults. The ``.`` was an unescaped regex wildcard, so a path containing any
    character followed by ``pdf`` was rewritten in the wrong place
    (``out/spdf_x.pdf`` became ``out/_heatmap.pdf_x_heatmap.pdf``). And for any
    extension other than ``.pdf`` the pattern simply did not match, so the derived
    path *equalled the original* -- meaning a caller who passed ``corr.png`` had
    the scatter plot overwritten by the clustered heatmap and that overwritten in
    turn by the non-clustered one, ending with one file where three were
    requested and no indication anything had been lost.
    """
    root, ext = os.path.splitext(save_file_path)
    return f"{root}{suffix}{ext}"


def _save_figure(fig: Figure, path: str, label: str) -> None:
    """Write ``fig`` to ``path``, creating the parent directory if needed."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fmt = os.path.splitext(path)[1].lstrip(".").lower() or None
    fig.savefig(path, format=fmt, **_SAVEFIG_KWARGS)
    # logging, not print(): this is library code called once per metric per
    # embedding per iteration, and a caller that wants the paths has them in the
    # return value.
    logger.info("%s saved to: %s", label, path)


def compute_results_correlation(results_df, correlation="spearman", thresh=0.7):
    """This function takes in as input a Pandas Dataframe containing the results and data evaluations for
    a given dataset.  It then produces a spearman correlation between the data evaluation characteristics (features)
    and instances where an F1 score was observed above a certain threshold (thresh).
    The function returns the input DataFrame with additional columns for datatype and model_embed_datatype,
    as well as a new DataFrame containing the computed correlations between metrics and features.
    The correlation is computed for each model-embedding-dataset combination, and the results are aggregated.
    The features correlated are whichever dataset-complexity columns the table carries, as identified by :func:`qbiocode.evaluation.dataset_evaluation.complexity_feature_columns` -- the ``mfe.``-prefixed pyMFE block, the ``task.``-prefixed target-spectrum block, and the natively-computed measures, or the legacy column names for a table written before the pyMFE integration.
    The metrics considered for correlation include 'accuracy', 'f1_score', 'time', and 'auc'.
    The function also calculates the median metric value and the fraction of instances above the specified threshold for each combination.
    The resulting DataFrame contains the model-embedding-dataset, metric, feature, median metric value, fraction above threshold, and the computed correlation.
    This function is useful for understanding how different data characteristics relate to model performance metrics, particularly in the context of machine learning models applied to datasets.

    Args:
        results_df (pd.DataFrame): A DataFrame containing the results and data evaluations.
        correlation (str): The type of correlation to compute, default is 'spearman'.
        thresh (float): The threshold for F1 score to consider, default is 0.7.

    Returns:
        results_df (pd.DataFrame): The input DataFrame with additional columns for datatype and model_embed_datatype.
        correlations_df (pd.DataFrame): A DataFrame containing the computed correlations between metrics and features.

    """

    # Refining datasrame
    results_df["datatype"] = [
        re.sub(r"\.csv", "", re.sub(r"-.*", "", x)) for x in results_df["Dataset"]
    ]
    results_df["model_embed_datatype"] = [
        "_".join([str(row.model), str(row.embeddings), str(row.datatype)])
        for idx, row in results_df.iterrows()
    ]

    correlations = []
    # Derived from the table, not hardcoded. This list used to name the 21 columns of
    # the pre-pyMFE complexity block and skip any that were absent -- so once
    # QProfiler's block became pyMFE-backed it would have silently correlated the ten
    # surviving names and none of the new ones, narrowing the analysis with nothing in
    # the output saying so.
    features = complexity_feature_columns(results_df.columns)
    if not features:
        logger.warning(
            "No dataset-complexity columns recognized in results_df, so no "
            "feature correlations will be computed. Expected either %s-prefixed "
            "columns from qbiocode.evaluation.evaluate() or the legacy complexity "
            "column names.",
            MFE_COLUMN_PREFIX,
        )
    metrics = ["accuracy", "f1_score", "time", "auc"]

    keys = list(set(results_df["model_embed_datatype"]))
    for m in keys:
        dat_temp_m = results_df[results_df["model_embed_datatype"] == m]
        if len(dat_temp_m) > 0:
            for s in metrics:
                for f in features:
                    if f in dat_temp_m.columns:
                        if correlation == "spearman":
                            correlations.append(
                                [
                                    m,
                                    s,
                                    f,
                                    np.median(dat_temp_m[s]),
                                    sum(dat_temp_m[s] > thresh) / len(dat_temp_m[s]),
                                    spearmanr(dat_temp_m[s], dat_temp_m[f])[0],
                                ]
                            )

    correlations_df = pd.DataFrame(
        correlations,
        columns=[
            "model_embed_datatype",
            "metric",
            "feature",
            "median_metric",
            "frac_gt_thresh",
            "correlation",
        ],
    )

    return results_df, correlations_df


def plot_results_correlation(
    correlations_df,
    metric="f1_score",
    title="",
    correlation_type="Spearman ρ",
    figsize=(6.5, 10),
    save_file_path="",
    size="median_metric",
    xticks=True,
    key="model_embed_datatype",
    legend_offset=1.0,
    show_plots=True,
    colorbar_label="Correlation coefficient",
    size_label="Median metric value",
):
    """Plot publication-quality correlation figures from a ``correlations_df``.

    Draws three figures from the frame produced by
    :func:`compute_results_correlation`: a dot plot, a row/column-clustered
    heatmap, and a heatmap with quantum models ordered first. In the dot plot the
    larger the circle, the higher the metric value for that data set; circle
    colour is the correlation between the data characteristic and the metric --
    red positive, blue anti-correlated, darker meaning stronger.

    Args:
        correlations_df (pd.DataFrame): A DataFrame containing the computed correlations between metrics and features.
        metric (str): The metric to plot, default is 'f1_score'.
        title (str): The title of the plot, default is an empty string.
        correlation_type (str): The type of correlation to display in the legend, default is 'Spearman ρ'.
        figsize (tuple): The size of the dot-plot figure; the heatmaps are scaled from it.
        save_file_path (str): Where to write the dot plot. The two heatmaps are
            written alongside it with ``_heatmap`` and ``_noncluster_heatmap``
            inserted before the extension, so any image format works and the
            three never collide. Nothing is written when this is ``""``.
        size (str): The column name to use for the size of the dots, default is 'median_metric'.
        xticks (bool): Whether to label the heatmap x-axis.
        key (str): Column identifying each model/embedding/datatype combination.
        legend_offset (float): Horizontal offset of the size legend.
        show_plots (bool): Whether to call ``plt.show()``. Default ``True`` for
            notebook use; it is a no-op under a non-interactive backend or with no
            display, so a headless or batch run neither blocks nor warns.
        colorbar_label (str): Label for the colorbar, default is 'Correlation coefficient'.
        size_label (str): Label for the size legend, default is 'Median metric value'.

    Returns:
        CorrelationFigures: the three figures, as
        ``(scatter, scatter_ax, clustered_heatmap, ordered_heatmap)``. All are
        already closed -- pyplot no longer tracks them, which is what stops
        figures accumulating across QProfiler's iterations -- but each remains
        usable for ``fig.savefig(...)`` or inline display.

    Notes:
        :data:`PUBLICATION_STYLE` is applied for the duration of this call only,
        via ``plt.rc_context``, and is restored even if plotting raises. The
        caller's ``plt.rcParams`` are never modified; use
        :func:`publication_style` to adopt the same settings deliberately.
    """
    # rc_context here rather than inside the implementation, so the caller's
    # rcParams are restored even when plotting raises partway through.
    with plt.rc_context(PUBLICATION_STYLE):
        return _plot_correlation_figures(
            correlations_df,
            metric=metric,
            title=title,
            correlation_type=correlation_type,
            figsize=figsize,
            save_file_path=save_file_path,
            size=size,
            xticks=xticks,
            key=key,
            legend_offset=legend_offset,
            show_plots=show_plots,
            colorbar_label=colorbar_label,
            size_label=size_label,
        )


def _plot_correlation_figures(
    correlations_df,
    metric="f1_score",
    title="",
    correlation_type="Spearman ρ",
    figsize=(6.5, 10),
    save_file_path="",
    size="median_metric",
    xticks=True,
    key="model_embed_datatype",
    legend_offset=1.0,
    show_plots=True,
    colorbar_label="Correlation coefficient",
    size_label="Median metric value",
):
    """Body of :func:`plot_results_correlation`; see there for the documented API.

    Separate only so the public function can own the ``plt.rc_context`` and
    guarantee the caller's rcParams are restored on the exception path too.
    """

    # Use enhanced professional diverging colormap
    from matplotlib.colors import LinearSegmentedColormap

    colors_custom = [
        "#053061",
        "#2166ac",
        "#4393c3",
        "#92c5de",
        "#d1e5f0",
        "#f7f7f7",
        "#fddbc7",
        "#f4a582",
        "#d6604d",
        "#b2182b",
        "#67001f",
    ]
    cmap_custom = LinearSegmentedColormap.from_list("custom_diverging", colors_custom, N=256)
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    # Sample data
    data = correlations_df[correlations_df["metric"] == metric].copy()
    data["feature"] = [
        re.sub(
            "std",
            "Std. dev. of",
            re.sub(
                "co of v",
                "coefficient of variation",
                re.sub(
                    "kurt$",
                    "kurtosis",
                    re.sub(
                        "skew$",
                        "skewness",
                        re.sub("var$", "variation", re.sub("%", "", re.sub("_", " ", x))),
                    ),
                ),
            ),
        )
        for x in data["feature"]
    ]

    if key == "model_datatype":
        data["datatype"] = ["_".join(x.split("_")[1:]) for x in data[key]]
        key_column = "Model / Dataset"
    else:
        data["datatype"] = ["_".join(x.split("_")[2:]) for x in data[key]]
        key_column = "Model / Embedding / Dataset"

    data = data.sort_values(["feature", "datatype"], ascending=False)
    data["model"] = [re.sub("_.*", "", x) for x in data[key]]
    data["model"] = [x.upper() for x in data["model"]]
    data = pd.concat(
        [
            data[~data["model"].isin(["QSVC", "QNN", "VQC", "PQK"])],
            data[data["model"].isin(["QSVC", "QNN", "VQC", "PQK"])],
        ]
    )
    fm = dict(zip(list(set(data["feature"])), range(len(set(data["feature"])))))
    data["feature_map"] = [fm[x] for x in data["feature"]]

    # Fill NaN values before scaling to avoid errors
    data = data.fillna(0)

    # Scale dot size based on actual data range for meaningful representation
    # Reduced sizes to minimize overlap
    epsilon = 25

    # Get actual min/max from the data to scale appropriately
    min_val = data[size].min()
    max_val = data[size].max()

    # Normalize to 0-1 based on actual data range, then scale to pixel sizes
    if max_val > min_val:
        normalized_values = (data[size] - min_val) / (max_val - min_val)
    else:
        normalized_values = np.ones_like(data[size]) * 0.5

    # Size formula: normalized value in [0,1] → size in [epsilon, 150+epsilon] (reduced from 200)
    data["norm_size"] = (normalized_values * 150 + epsilon).astype(float)

    data[key] = [re.sub("_", " / ", x) for x in data[key]]

    # Create figure with very compact design
    fig, ax = plt.subplots(figsize=figsize, facecolor="white", dpi=100)
    ax.set_facecolor("white")

    # Create scatter plot with enhanced professional styling
    scatter = ax.scatter(
        data[key],
        data["feature"],
        s=data["norm_size"],
        c=data["correlation"],
        cmap=cmap_custom,
        norm=norm,
        alpha=0.92,
        edgecolors="#34495E",
        linewidths=1.2,
        zorder=3,
    )

    # Add colorbar with enhanced professional styling
    cbar = plt.colorbar(scatter, ax=ax, pad=0.018, aspect=28, shrink=0.88)
    cbar.set_label(colorbar_label, rotation=270, labelpad=22, fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10, width=1.3, length=5, pad=4)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_edgecolor("#34495E")

    # Set labels with clean formatting
    ax.set_xlabel(key_column, fontweight="bold", fontsize=13, labelpad=10)
    ax.set_ylabel("Data Feature", fontweight="bold", fontsize=13, labelpad=10)

    # Add title if provided
    if title:
        ax.set_title(title, fontweight="bold", pad=20, fontsize=14)

    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha="right", va="top", fontsize=10)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=10)

    # Add professional grid for better readability
    ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.8, color="#95A5A6", zorder=0)
    ax.set_axisbelow(True)

    # Proper margins to prevent cropping while keeping columns close
    ax.margins(x=0.025, y=0.035)

    # Clean tick parameters
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.2, length=5)

    # Remove top and right spines for cleaner look
    sns.despine(ax=ax)

    # Create size legend with 4 dots showing ACTUAL median metric values from data
    handles_size, labels_size = scatter.legend_elements(
        prop="sizes", alpha=0.75, num=4, markeredgecolor="#34495E", markeredgewidth=1.2
    )

    # Use REAL median metric values from the data
    smin = np.min(data[size])
    smax = np.max(data[size])
    labels_size = [f"{x:.2f}" for x in np.linspace(smin, smax, 4)]

    # Position legend on the right side, well below the colorbar with proper spacing
    legend = ax.legend(
        handles_size,
        labels_size,
        title=size_label,
        loc="upper left",
        bbox_to_anchor=(1.15, -0.05),
        ncol=1,
        frameon=True,
        fancybox=False,
        title_fontsize=9,
        fontsize=8,
        edgecolor="#34495E",
        framealpha=0.98,
        labelspacing=0.8,
        handletextpad=0.5,
    )
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_facecolor("white")
    legend.get_title().set_fontweight("bold")

    # Adjust layout with reduced horizontal spacing between subplots.
    # fig.tight_layout, not plt.tight_layout: the module-level call acts on
    # whatever figure happens to be current, which is not this one as soon as a
    # caller draws anything between construction and here.
    fig.tight_layout(pad=0.8, w_pad=1.8)

    if save_file_path != "":
        _save_figure(fig, save_file_path, "Scatter plot")

    if show_plots and _can_show():
        plt.show()
    # plt.close(fig), not plt.close(): the bare form closes the *current* figure,
    # which after plt.show() under some backends is a different one -- leaving this
    # figure open to accumulate across QProfiler's iterations.
    plt.close(fig)

    model_qml = list(_QML_MODELS)

    data[key_column] = data[key]
    data["Data feature"] = data["feature"]
    to_plot = data.pivot_table(columns=key_column, index="Data feature", values="correlation")

    # Define professional color scheme for model types
    ccolors = [
        "#7B68EE" if re.sub(" .*", "", x) in model_qml else "#FF8C00" for x in to_plot.columns
    ]

    # Create custom diverging colormap
    from matplotlib.colors import LinearSegmentedColormap

    colors_heatmap = [
        "#2166ac",
        "#4393c3",
        "#92c5de",
        "#d1e5f0",
        "#f7f7f7",
        "#fddbc7",
        "#f4a582",
        "#d6604d",
        "#b2182b",
    ]
    cmap_heatmap = LinearSegmentedColormap.from_list("custom_heatmap", colors_heatmap, N=256)

    # Create professional heatmap with better proportions
    heatmap_height = figsize[1] * 0.95  # Much taller to reduce space above colorbar
    heatmap_width = min(figsize[0] * 0.9, 10)  # Narrower columns

    g = sns.clustermap(
        to_plot.fillna(0),
        figsize=(heatmap_width, heatmap_height),
        col_colors=ccolors,
        cmap=cmap_heatmap,
        method="average",
        metric="euclidean",
        center=0,
        xticklabels=xticks,
        yticklabels=True,
        cbar_kws={"label": colorbar_label, "orientation": "horizontal"},
        linewidths=1.0,
        linecolor="white",
        vmin=-1,
        vmax=1,
        dendrogram_ratio=0.05,
        cbar_pos=(0.55, 0.01, 0.4, 0.015),
    )

    # Hide dendrograms for cleaner appearance
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)

    # Improve axis labels with better styling
    g.ax_heatmap.set_xlabel(
        key_column, fontweight="bold", fontsize=11, labelpad=12, color="#2C3E50"
    )
    g.ax_heatmap.set_ylabel(
        "Data Feature", fontweight="bold", fontsize=11, labelpad=12, color="#2C3E50"
    )

    # Rotate x-labels 45 degrees for readability
    plt.setp(
        g.ax_heatmap.xaxis.get_majorticklabels(),
        rotation=45,
        ha="right",
        fontsize=9,
        color="#2C3E50",
    )
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=9, color="#2C3E50")

    # Improve tick parameters with better styling
    g.ax_heatmap.tick_params(
        axis="both", which="major", width=1.2, length=5, pad=4, colors="#2C3E50"
    )

    # Style heatmap spines
    for spine in g.ax_heatmap.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor("#34495E")

    # Enhance horizontal colorbar styling at bottom
    if g.cax is not None:
        g.cax.set_xlabel(
            colorbar_label, fontsize=10, fontweight="bold", labelpad=10, color="#2C3E50"
        )
        g.cax.tick_params(labelsize=9, width=1.2, length=4, colors="#2C3E50")
        for spine in g.cax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor("#34495E")

    if save_file_path != "":
        _save_figure(g.figure, _derive_path(save_file_path, "_heatmap"), "Clustered heatmap")

    if show_plots and _can_show():
        plt.show()
    plt.close(g.figure)

    # Create non-clustered heatmap with quantum models first
    qml_col = [x for x in to_plot.columns if re.sub(" .*", "", x) in model_qml]
    cml_col = [x for x in to_plot.columns if re.sub(" .*", "", x) not in model_qml]
    to_plot_ordered = to_plot.loc[:, qml_col + cml_col]
    ccolors_ordered = [
        "#7B68EE" if re.sub(" .*", "", x) in model_qml else "#FF8C00"
        for x in to_plot_ordered.columns
    ]

    g2 = sns.clustermap(
        to_plot_ordered.fillna(0),
        figsize=(heatmap_width, heatmap_height),
        col_colors=ccolors_ordered,
        col_cluster=False,
        row_cluster=True,
        cmap=cmap_heatmap,
        center=0,
        xticklabels=xticks,
        yticklabels=True,
        cbar_kws={"label": colorbar_label, "orientation": "horizontal"},
        linewidths=1.0,
        linecolor="white",
        vmin=-1,
        vmax=1,
        dendrogram_ratio=0.05,
        cbar_pos=(0.55, 0.01, 0.4, 0.015),
        method="average",
        metric="euclidean",
    )

    # Improve axis labels with better styling
    g2.ax_heatmap.set_xlabel(
        key_column, fontweight="bold", fontsize=11, labelpad=12, color="#2C3E50"
    )
    g2.ax_heatmap.set_ylabel(
        "Data Feature", fontweight="bold", fontsize=11, labelpad=12, color="#2C3E50"
    )

    # Rotate x-labels 45 degrees for readability
    plt.setp(
        g2.ax_heatmap.xaxis.get_majorticklabels(),
        rotation=45,
        ha="right",
        fontsize=9,
        color="#2C3E50",
    )
    plt.setp(g2.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=9, color="#2C3E50")

    # Improve tick parameters with better styling
    g2.ax_heatmap.tick_params(
        axis="both", which="major", width=1.2, length=5, pad=4, colors="#2C3E50"
    )

    # Style heatmap spines
    for spine in g2.ax_heatmap.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor("#34495E")

    # Enhance horizontal colorbar styling at bottom
    if g2.cax is not None:
        g2.cax.set_xlabel(
            colorbar_label, fontsize=10, fontweight="bold", labelpad=10, color="#2C3E50"
        )
        g2.cax.tick_params(labelsize=9, width=1.2, length=4, colors="#2C3E50")
        for spine in g2.cax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor("#34495E")

    if save_file_path != "":
        _save_figure(
            g2.figure,
            _derive_path(save_file_path, "_noncluster_heatmap"),
            "Non-clustered heatmap",
        )

    if show_plots and _can_show():
        plt.show()
    plt.close(g2.figure)

    return CorrelationFigures(
        scatter=fig, scatter_ax=ax, clustered_heatmap=g, ordered_heatmap=g2
    )
