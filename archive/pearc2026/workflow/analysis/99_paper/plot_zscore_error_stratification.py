"""Generate the Z-score error stratification figure (paper Figure 1).

Two output modes:

    python analysis/99_paper/plot_zscore_error_stratification.py
        -> analysis/99_paper/results/figures/
           zscore_error_stratification.{pdf,png}          CPUTime + maxRSS (as published)

    python analysis/99_paper/plot_zscore_error_stratification.py --cpu-only
        -> analysis/99_paper/results/figures/
           zscore_error_stratification_cputime.{pdf,png}  CPUTime row only, for slides

The CPUTime-only mode keeps the "Maximum absolute Z-score" axis label, which is lost
when the published two-row figure is cropped to its top row.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter


DEFAULT_OUTDIR = Path(__file__).resolve().parent / "results" / "figures"

# -----------------------
# Real data (your values)
# -----------------------
cpu_data = {
    '2DSA': {'median': [14.5, 29.4, 338.5, 46.8], 'p90': [14.9, 102.4, 2113.8, 370.0]},
    'GA': {'median': [5256.1, 6611.0, 5926.6, 16669.0], 'p90': [12401.8, 19365.8, 37455.7, 93592.6]},
    'PCSA': {'median': [91.3, 306.8, 509.7], 'p90': [238.5, 1168.5, 1683.7]}
}

mem_data = {
    '2DSA': {'median': [27.8, 295.0, 932.6, 146.7], 'p90': [53.7, 3446.6, 4589.7, 334.0]},
    'GA': {'median': [2067.3, 5464.7, 1396.8, 43458.8], 'p90': [147587.3, 192923.8, 344939.8, 136189.6]},
    'PCSA': {'median': [301.3, 2991.9, 4933.8], 'p90': [531.7, 7634.1, 13379.9]}
}

methods = ["2DSA", "GA", "PCSA"]

# Compact bin labels (avoids crowding)
z_bin_labels = [r"$\leq 1$", r"$(1,2]$", r"$(2,3]$", r"$>3$"]
x = np.arange(len(z_bin_labels))


def comma_ticks(x, _pos):
    return f"{int(x):,}" if x >= 1 else f"{x:g}"


def pad_to_4(vals, method):
    """PCSA has length 3 (no z<=1). Pad with NaN so x-bins align across methods."""
    vals = list(vals)
    if len(vals) == 4:
        return np.array(vals, dtype=float)
    if method == "PCSA" and len(vals) == 3:
        return np.array([np.nan] + vals, dtype=float)
    raise ValueError(f"Unexpected bin count for {method}: {len(vals)}")


def plain_log_ticks(y, _pos):
    """Format powers of 10 as plain numbers: 10, 100, 1,000, ..."""
    if y <= 0:
        return ""
    e = np.log10(y)
    if abs(e - round(e)) < 1e-8:
        return f"{int(y):,}"
    return ""


def series_for(data, method):
    return {
        "median": pad_to_4(data[method]["median"], method),
        "p90": pad_to_4(data[method]["p90"], method),
    }


def plot_panel(ax, series, title=None, y_label=None, yscale="linear"):
    # Median: solid + circles
    ax.plot(
        x, series["median"],
        linestyle="-", marker="o",
        linewidth=2.3, markersize=6.5,
        label="Median",
    )

    # 90th percentile: dashed + triangles
    ax.plot(
        x, series["p90"],
        linestyle="--", marker="^",
        linewidth=2.3, markersize=6.5,
        label="90th percentile",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(z_bin_labels, fontsize=12)  # reduce crowding
    ax.tick_params(axis="y", labelsize=13, pad=4)

    if y_label:
        ax.set_ylabel(y_label, fontsize=15, fontweight="bold", labelpad=10)

    if title is not None and title != "":
        ax.set_title(title, fontsize=16, fontweight="bold")

    ax.set_yscale(yscale)

    # Keep points off borders a bit (prevents "hits neighbor axis" look)
    ax.margins(x=0.06, y=0.10)
    ax.set_xlim(-0.5, len(z_bin_labels) - 0.5)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.55)


def style_cpu_axes(axes_row):
    """Shared log scale + plain tick labels across the CPUTime panels."""
    cpu_all = []
    for m in methods:
        cpu_all += cpu_data[m]["median"] + cpu_data[m]["p90"]
    cpu_all = np.asarray(cpu_all, dtype=float)
    cpu_all = cpu_all[np.isfinite(cpu_all) & (cpu_all > 0)]

    ymin = 10 ** np.floor(np.log10(cpu_all.min()))
    ymax = 10 ** np.ceil(np.log10(cpu_all.max()))

    for ax in axes_row:
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(plain_log_ticks))
        ax.yaxis.set_minor_formatter(NullFormatter())


def build_figure(cpu_only=False):
    nrows = 1 if cpu_only else 2
    figsize = (14, 4.6) if cpu_only else (14, 8)

    fig, axes = plt.subplots(
        nrows, 3,
        figsize=figsize,
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )

    for col, m in enumerate(methods):
        plot_panel(
            axes[0, col],
            series_for(cpu_data, m),
            title=m,
            y_label="Absolute error (s)" if col == 0 else None,
            yscale="log",
        )

        if not cpu_only:
            plot_panel(
                axes[1, col],
                series_for(mem_data, m),
                title="",
                y_label="Absolute error (MB)" if col == 0 else None,
                yscale="linear",
            )

        if col == 2:
            axes[0, col].legend(loc="lower right", frameon=True, fontsize=12)

    style_cpu_axes(axes[0, :])

    if not cpu_only:
        # --- Format maxRSS (bottom row) ticks with commas ---
        for ax in axes[1, :]:
            ax.yaxis.set_major_formatter(FuncFormatter(comma_ticks))

    # Align left-column y-labels so they share the same x-position
    fig.align_ylabels(axes[:, 0])

    # Shared x label (figure-level) — retained in cpu-only mode, unlike a crop
    fig.supxlabel("Maximum absolute Z-score", fontsize=15, fontweight="bold")

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu-only", action="store_true",
                        help="plot only the CPUTime row (for slides)")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help=f"directory for output files (default: {DEFAULT_OUTDIR})")
    args = parser.parse_args()

    stem = "zscore_error_stratification_cputime" if args.cpu_only else "zscore_error_stratification"
    fig = build_figure(cpu_only=args.cpu_only)
    args.outdir.mkdir(parents=True, exist_ok=True)

    for ext, kwargs in (("pdf", {}), ("png", {"dpi": 300})):
        path = args.outdir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"wrote {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
