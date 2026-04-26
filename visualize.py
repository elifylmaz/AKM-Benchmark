import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = "/Users/elifyilmaz/Desktop/pattern_mining"
OUT_DIR  = os.path.join(BASE_DIR, "results")
FIG_DIR  = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

COLORS  = {"CMRules": "#2176AE", "RuleGrowth": "#E07A5F", "ERMiner": "#3D405B"}
MARKERS = {"CMRules": "o",       "RuleGrowth": "s",       "ERMiner": "^"}
ALGOS   = list(COLORS.keys())

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.grid":      True,
    "grid.alpha":     0.35,
    "grid.linestyle": "--",
    "figure.dpi":     300,
})


def sup_label(v: float) -> str:
    return f"{v * 100:.4g}%"


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(FIG_DIR, f"{name}.{ext}"), bbox_inches="tight"
        )
    plt.close(fig)
    print(f"  Saved: {name}.pdf / .png")


def _mem_column(df: pd.DataFrame) -> str:

    if "mean_mem_mb" in df.columns:
        # Check that at least one non-zero, non-NaN value exists across all
        # algorithms (not just CMRules).
        non_trivial = df["mean_mem_mb"].replace(0, np.nan).dropna()
        if len(non_trivial) > len(df[df["algorithm"] == "CMRules"]):
            return "mean_mem_mb"
    if "mean_mem_rss_mb" in df.columns:
        return "mean_mem_rss_mb"
    return "mean_mem_mb"


# ── Figure 2: Execution Time vs. Min Support ─────────────────────────────────
def plot_exec_time(df: pd.DataFrame) -> None:

    for ds in df["dataset"].unique():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sub = df[df["dataset"] == ds].sort_values("min_support")
        for algo in ALGOS:
            g = sub[sub["algorithm"] == algo]
            if g.empty:
                continue
            ax.errorbar(
                [sup_label(v) for v in g["min_support"]],
                g["mean_time_s"],
                yerr=g["ci95_s"].fillna(0),
                label=algo,
                color=COLORS[algo],
                marker=MARKERS[algo],
                linewidth=1.8,
                markersize=6,
                capsize=4,
                elinewidth=1.2,
            )
        ax.set_title(
            f"Execution Time vs. Minimum Support\n({ds})", fontweight="bold"
        )
        ax.set_xlabel("Minimum Support Threshold")
        ax.set_ylabel("Mean Execution Time (s) ± 95% CI")
        ax.legend(title="Algorithm", frameon=True)
        fig.tight_layout()
        save(fig, f"fig2_exec_time_{ds}")


# ── Figure 3: Discovered Rules vs. Min Support ───────────────────────────────
def plot_pattern_count(df: pd.DataFrame) -> None:

    for ds in df["dataset"].unique():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sub = df[df["dataset"] == ds].sort_values("min_support")
        for algo in ALGOS:
            g = sub[sub["algorithm"] == algo]
            if g.empty:
                continue
            ax.errorbar(
                [sup_label(v) for v in g["min_support"]],
                g["mean_n_rules"],
                yerr=g["std_n_rules"].fillna(0),
                label=algo,
                color=COLORS[algo],
                marker=MARKERS[algo],
                linewidth=1.8,
                markersize=6,
                capsize=4,
                elinewidth=1.2,
            )
        ax.set_yscale("symlog")
        ax.set_title(
            f"Number of Discovered Rules vs. Minimum Support\n({ds})",
            fontweight="bold",
        )
        ax.set_xlabel("Minimum Support Threshold")
        ax.set_ylabel("Rule Count (mean ± std, symlog scale)")
        ax.legend(title="Algorithm", frameon=True)
        fig.tight_layout()
        save(fig, f"fig3_pattern_count_{ds}")


# ── Figure 4: Scalability Analysis ───────────────────────────────────────────
def plot_scalability(df: pd.DataFrame) -> None:

    for ds in df["base_dataset"].unique():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sub = df[df["base_dataset"] == ds].sort_values("fraction")
        for algo in ALGOS:
            g = sub[sub["algorithm"] == algo]
            if g.empty:
                continue
            ax.plot(
                (g["fraction"] * 100).astype(int),
                g["mean_time_s"],
                label=algo,
                color=COLORS[algo],
                marker=MARKERS[algo],
                linewidth=1.8,
                markersize=6,
            )
        ax.set_title(
            f"Scalability: Execution Time vs. Dataset Size\n({ds})",
            fontweight="bold",
        )
        ax.set_xlabel("Dataset Size (% of full dataset)")
        ax.set_ylabel("Execution Time (s)")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d%%"))
        ax.legend(title="Algorithm", frameon=True)
        fig.tight_layout()
        save(fig, f"fig4_scalability_{ds}")


# ── Figure 5: Pairwise Speedup Heatmap ───────────────────────────────────────
def plot_speedup(df: pd.DataFrame) -> None:

    for ds in df["dataset"].unique():
        sup_vals = sorted(df[df["dataset"] == ds]["min_support"].unique())
        ncols    = min(3, len(sup_vals))
        nrows    = -(-len(sup_vals) // ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows)
        )
        axes = np.array(axes).flatten()
        for idx, sup in enumerate(sup_vals):
            sub   = df[(df["dataset"] == ds) & (df["min_support"] == sup)]
            times = {
                a: float(sub[sub["algorithm"] == a]["mean_time_s"].iloc[0])
                if not sub[sub["algorithm"] == a].empty
                else np.nan
                for a in ALGOS
            }
            mat = pd.DataFrame(
                {
                    b: [
                        times[a] / times[b]
                        if times.get(b, 0) > 0
                        else np.nan
                        for a in ALGOS
                    ]
                    for b in ALGOS
                },
                index=ALGOS,
            )
            sns.heatmap(
                mat.astype(float),
                annot=True,
                fmt=".2f",
                cmap="RdYlGn_r",
                center=1.0,
                linewidths=0.5,
                ax=axes[idx],
                cbar_kws={"label": "time(row) / time(col)"},
            )
            axes[idx].set_title(f"minsup = {sup_label(sup)}")
        for idx in range(len(sup_vals), len(axes)):
            axes[idx].set_visible(False)
        fig.suptitle(
            f"Pairwise Speedup Matrix — {ds}\n"
            f"(values >1 mean row algorithm is slower)",
            fontsize=12,
            fontweight="bold",
            y=1.01,
        )
        fig.tight_layout()
        save(fig, f"fig5_speedup_{ds}")


# ── Figure 6: Timing Stability (CV) ──────────────────────────────────────────
def plot_cv(df: pd.DataFrame) -> None:

    for ds in df["dataset"].unique():
        sub      = df[df["dataset"] == ds].sort_values("min_support")
        sup_vals = sorted(sub["min_support"].unique())
        x        = np.arange(len(sup_vals))
        w        = 0.25
        offsets  = (
            np.linspace(-(len(ALGOS) - 1) / 2, (len(ALGOS) - 1) / 2, len(ALGOS))
            * w
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for algo, off in zip(ALGOS, offsets):
            g = sub[sub["algorithm"] == algo].sort_values("min_support")
            ax.bar(
                x + off,
                g["cv_time"].fillna(0),
                width=w,
                label=algo,
                color=COLORS[algo],
                alpha=0.85,
                edgecolor="white",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([sup_label(v) for v in sup_vals])
        ax.set_xlabel("Minimum Support Threshold")
        ax.set_ylabel("Coefficient of Variation (CV = std / mean)")
        ax.set_title(
            f"Timing Stability across Repetitions — {ds}", fontweight="bold"
        )
        ax.legend(title="Algorithm", frameon=True)
        fig.tight_layout()
        save(fig, f"fig6_cv_{ds}")


# ── Figure 7: Memory Usage vs. Min Support ───────────────────────────────────
def plot_memory(df: pd.DataFrame) -> None:

    mem_col = _mem_column(df)
    source_note = (
        "Memory source: SPMF-reported (CMRules) / psutil RSS (RuleGrowth, ERMiner)"
        if mem_col == "mean_mem_mb"
        else f"Memory source: {mem_col}"
    )

    for ds in df["dataset"].unique():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sub = df[df["dataset"] == ds].sort_values("min_support")

        any_data = False
        for algo in ALGOS:
            g = sub[sub["algorithm"] == algo]
            if g.empty:
                continue
            vals = g[mem_col].replace(0, np.nan)
            if vals.isna().all():
                continue
            any_data = True
            ax.plot(
                [sup_label(v) for v in g["min_support"]],
                vals,
                label=algo,
                color=COLORS[algo],
                marker=MARKERS[algo],
                linewidth=1.8,
                markersize=6,
            )

        if not any_data:
            plt.close(fig)
            print(f"  [SKIP] fig7_memory_{ds} — no memory data available")
            continue

        ax.set_title(
            f"Memory Usage vs. Minimum Support\n({ds})", fontweight="bold"
        )
        ax.set_xlabel("Minimum Support Threshold")
        ax.set_ylabel("Mean Memory Usage (MB)")
        ax.legend(title="Algorithm", frameon=True)
        fig.text(
            0.5, -0.04,
            source_note,
            ha="center",
            fontsize=8,
            style="italic",
            color="grey",
        )
        fig.tight_layout()
        save(fig, f"fig7_memory_{ds}")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading results ...")
    df_main  = pd.read_csv(os.path.join(OUT_DIR, "main_results.csv"))
    df_scale = pd.read_csv(os.path.join(OUT_DIR, "scalability_results.csv"))

    print(f"\nMemory column selected: {_mem_column(df_main)}")
    print("\nGenerating figures ...")
    plot_exec_time(df_main)
    plot_pattern_count(df_main)
    plot_scalability(df_scale)
    plot_speedup(df_main)
    plot_cv(df_main)
    plot_memory(df_main)

    print(f"\nAll figures saved to: {FIG_DIR}")