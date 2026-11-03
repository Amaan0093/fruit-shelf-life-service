"""
Generate all figures for the report from the CSV artefacts produced by
`evaluate_all.py` and `ablation_study.py`.

Reads (from config.ARTIFACTS, i.e. `outputs/`):
    metrics_summary.csv         - per-model overall MAE/RMSE/R^2 + size
    per_fruit_metrics.csv       - per-model per-fruit metrics
    latency.csv                 - per-image inference latency
    ablation.csv                - unified vs per-fruit specialists
    predictions_svr.csv         - test-set predictions (path, fruit, stage, days, y_pred)
    predictions_rf.csv
    predictions_mobilenet.csv
    rf.joblib                   - (optional) loaded for feature-group importance

Writes to:
    outputs/figures/*.png

All figures use a consistent colour palette and are saved at 200 dpi
(suitable for IEEE two-column inclusion). Each figure can be regenerated
independently by re-running this script.

Usage:
    python scripts/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow `python scripts/make_figures.py` to find project modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ARTIFACTS, FRUIT_LIST, FRUITS

FIG_DIR = ARTIFACTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent palette across every figure.
MODEL_COLOURS = {
    "SVR":          "#1f77b4",
    "RandomForest": "#2ca02c",
    "MobileNetV2":  "#d62728",
}
MODEL_ORDER = ["SVR", "RandomForest", "MobileNetV2"]

# Common matplotlib settings for paper-ready output.
plt.rcParams.update({
    "figure.dpi":    100,
    "savefig.dpi":   200,
    "savefig.bbox":  "tight",
    "font.size":     10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# Helpers

def _load_csv(name: str) -> pd.DataFrame | None:
    p = ARTIFACTS / name
    if not p.exists():
        print(f"  warn: {p} not found, skipping figures that need it")
        return None
    return pd.read_csv(p)


def _save(fig: plt.Figure, name: str) -> None:
    out = FIG_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def _model_colour_list(models: list[str]) -> list[str]:
    return [MODEL_COLOURS.get(m, "#888") for m in models]


# Figure 1: Headline metrics bar chart (MAE / RMSE / R^2)

def fig_headline_metrics(summary: pd.DataFrame) -> None:
    summary = summary.set_index("model").reindex(MODEL_ORDER).dropna(how="all")
    metrics = ["mae", "rmse", "r2"]
    titles = ["MAE (days, lower is better)",
              "RMSE (days, lower is better)",
              "R$^2$ (higher is better)"]

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    for ax, metric, title in zip(axes, metrics, titles):
        vals = summary[metric].values
        bars = ax.bar(summary.index, vals,
                      color=[MODEL_COLOURS[m] for m in summary.index],
                      edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_xticks(range(len(summary.index)))
        ax.set_xticklabels(summary.index, rotation=15, ha="right")
        # Annotate bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.margins(y=0.15)
    fig.tight_layout()
    _save(fig, "fig1_headline_metrics.png")


# Figure 2: Per-fruit MAE (grouped bars)

def fig_per_fruit_mae(per_fruit: pd.DataFrame) -> None:
    pivot = (per_fruit.pivot(index="fruit", columns="model", values="mae")
                       .reindex(FRUIT_LIST)
                       .reindex(columns=MODEL_ORDER))

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.arange(len(pivot.index))
    width = 0.25
    for i, model in enumerate(pivot.columns):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, pivot[model].values, width,
                      label=model, color=MODEL_COLOURS[model],
                      edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, pivot[model].values):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Test MAE (days)")
    ax.set_title("Per-fruit test-set MAE by model")
    ax.legend(frameon=False)
    ax.margins(y=0.18)
    fig.tight_layout()
    _save(fig, "fig2_per_fruit_mae.png")


# Figure 3: Per-stage MAE (averaged across fruits, grouped bars)

def fig_per_stage_mae(predictions: dict[str, pd.DataFrame]) -> None:
    rows = []
    for model, df in predictions.items():
        df = df.copy()
        df["abs_err"] = (df["y_pred"] - df["days"]).abs()
        for stage, sub in df.groupby("stage"):
            rows.append({"model": model, "stage": int(stage),
                         "mae": sub["abs_err"].mean(),
                         "n": len(sub)})
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="stage", columns="model", values="mae").reindex(columns=MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.arange(len(pivot.index))
    width = 0.25
    for i, model in enumerate(pivot.columns):
        offset = (i - 1) * width
        ax.bar(x + offset, pivot[model].values, width,
               label=model, color=MODEL_COLOURS[model],
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Stage {s}" for s in pivot.index])
    ax.set_ylabel("MAE (days)")
    ax.set_title("Per-stage test MAE (averaged across fruits)")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig3_per_stage_mae.png")


# Figure 4: Predicted vs actual scatter (3-panel)

def fig_pred_vs_actual(predictions: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, len(predictions), figsize=(11, 3.5), sharey=True)
    if len(predictions) == 1:
        axes = [axes]

    # Global axis range
    all_max = max(df["days"].max() for df in predictions.values())
    lim = (-0.5, all_max + 1)

    fruit_markers = {f: m for f, m in zip(FRUIT_LIST, ["o", "s", "^", "D", "P", "X"])}

    for ax, (model, df) in zip(axes, predictions.items()):
        for fruit in df["fruit"].unique():
            sub = df[df["fruit"] == fruit]
            ax.scatter(sub["days"], sub["y_pred"],
                       s=12, alpha=0.45,
                       marker=fruit_markers.get(fruit, "o"),
                       label=fruit,
                       color=MODEL_COLOURS.get(model, "#666"),
                       edgecolor="white", linewidth=0.3)
        # y = x reference line
        ax.plot(lim, lim, ls="--", lw=0.8, color="black", alpha=0.6)
        mae = (df["y_pred"] - df["days"]).abs().mean()
        ax.set_title(f"{model}  (MAE = {mae:.2f}d)")
        ax.set_xlabel("Actual shelf life (days)")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    axes[0].set_ylabel("Predicted shelf life (days)")
    fig.tight_layout()
    _save(fig, "fig4_pred_vs_actual.png")


# Figure 5: Residual distributions (histograms)

def fig_residuals(predictions: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bins = np.linspace(-8, 8, 41)
    for model, df in predictions.items():
        residuals = df["y_pred"] - df["days"]
        ax.hist(residuals, bins=bins, alpha=0.45, label=model,
                color=MODEL_COLOURS.get(model, "#666"),
                edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel("Residual = predicted - actual (days)")
    ax.set_ylabel("Test images")
    ax.set_title("Residual distribution (negative = under-prediction)")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig5_residuals.png")


# Figure 6: Efficiency tradeoff -- latency vs MAE, marker size = model size

def fig_efficiency_tradeoff(summary: pd.DataFrame, latency: pd.DataFrame) -> None:
    summary = summary.set_index("model")
    latency = latency.set_index("model")
    common = [m for m in MODEL_ORDER if m in summary.index and m in latency.index]
    if len(common) < 2:
        print("  warn: not enough models with both metrics + latency, skipping fig 6")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for model in common:
        x = latency.loc[model, "mean_ms"]
        y = summary.loc[model, "mae"]
        size = summary.loc[model, "size_mb"]
        ax.scatter(x, y, s=max(60, min(size * 40, 600)),
                   color=MODEL_COLOURS[model], edgecolor="black",
                   alpha=0.85, label=f"{model} ({size:.1f} MB)")
        ax.annotate(model, (x, y),
                    xytext=(8, 8), textcoords="offset points", fontsize=9)
    ax.set_xlabel("CPU inference latency (ms / image, mean)")
    ax.set_ylabel("Test MAE (days)")
    ax.set_title("Accuracy / latency tradeoff   (marker area $\\propto$ model size)")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    _save(fig, "fig6_efficiency_tradeoff.png")


# Figure 7: RF feature-group importance (loaded from rf.joblib)

def fig_rf_feature_importance() -> None:
    rf_path = ARTIFACTS / "rf.joblib"
    if not rf_path.exists():
        print("  warn: rf.joblib not found, skipping fig 7")
        return
    try:
        from models.random_forest import load, grouped_importance
        rf, lengths = load()
        groups = grouped_importance(rf, lengths)
    except Exception as e:
        print(f"  warn: could not compute RF importance ({e!r}), skipping fig 7")
        return

    labels = ["Colour\n(HSV histogram)", "Texture\n(HOG)", "Fruit type\n(one-hot)"]
    values = [groups["colour"], groups["texture"], groups["fruit_type"]]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, values, color=["#4c72b0", "#dd8452", "#55a467"],
                  edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Aggregated Gini importance")
    ax.set_title("Random Forest: feature-group contribution")
    ax.margins(y=0.15)
    fig.tight_layout()
    _save(fig, "fig7_rf_feature_importance.png")


# Figure 8: Ablation -- unified vs per-fruit specialists

def fig_ablation(ablation: pd.DataFrame) -> None:
    pivot = ablation.pivot(index="fruit", columns="model", values="mae").reindex(FRUIT_LIST)
    # Pair the 4 columns: SVR_unified vs SVR_specialist, RF_unified vs RF_specialist
    pairs = [
        ("SVR",          ["SVR_unified", "SVR_specialist"]),
        ("RandomForest", ["RF_unified",  "RF_specialist"]),
    ]
    pairs = [(name, cols) for name, cols in pairs if all(c in pivot.columns for c in cols)]
    if not pairs:
        print("  warn: ablation.csv has no recognised columns, skipping fig 8")
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 3.5), sharey=True)
    if len(pairs) == 1:
        axes = [axes]
    width = 0.35
    x = np.arange(len(pivot.index))
    for ax, (name, cols) in zip(axes, pairs):
        unified, specialist = cols
        ax.bar(x - width/2, pivot[unified].values, width,
               label="Unified (all fruits)",
               color=MODEL_COLOURS.get(name, "#888"),
               edgecolor="black", linewidth=0.5)
        ax.bar(x + width/2, pivot[specialist].values, width,
               label="Specialist (per fruit)",
               color=MODEL_COLOURS.get(name, "#888"),
               edgecolor="black", linewidth=0.5,
               hatch="///", alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(pivot.index)
        ax.set_title(name)
        ax.set_ylabel("Test MAE (days)" if ax is axes[0] else "")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Unified vs per-fruit specialist models", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig8_ablation_unified_vs_specialist.png")


# Figure 9: Predicted-days distribution by true stage (boxplots)

def fig_stage_boxplot(predictions: dict[str, pd.DataFrame]) -> None:
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.8), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (model, df) in zip(axes, predictions.items()):
        stages = sorted(df["stage"].unique())
        data = [df[df["stage"] == s]["y_pred"].values for s in stages]
        # True days for each stage (per fruit) -- show as scatter overlay
        bp = ax.boxplot(data, positions=stages, widths=0.5, patch_artist=True,
                        showfliers=False,
                        boxprops=dict(facecolor=MODEL_COLOURS.get(model, "#aaa"),
                                      alpha=0.5, edgecolor="black"),
                        medianprops=dict(color="black"))
        # Overlay the actual target days for each (fruit, stage) cell
        for s in stages:
            actual = df[df["stage"] == s]["days"].unique()
            ax.scatter([s] * len(actual), actual, marker="x", color="black", s=30,
                       zorder=3, label="True days" if s == stages[0] else None)
        ax.set_title(model)
        ax.set_xlabel("True stage")
        ax.set_xticks(stages)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        if ax is axes[0]:
            ax.set_ylabel("Predicted shelf life (days)")
            ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("Predicted-days distribution per true stage", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig9_stage_boxplot.png")


# Main

def main() -> None:
    print(f"Reading from {ARTIFACTS}/")
    print(f"Writing to   {FIG_DIR}/\n")

    summary = _load_csv("metrics_summary.csv")
    per_fruit = _load_csv("per_fruit_metrics.csv")
    latency = _load_csv("latency.csv")
    ablation = _load_csv("ablation.csv")

    predictions: dict[str, pd.DataFrame] = {}
    for tag, model_name in [("svr", "SVR"),
                            ("rf",  "RandomForest"),
                            ("mobilenet", "MobileNetV2")]:
        df = _load_csv(f"predictions_{tag}.csv")
        if df is not None:
            predictions[model_name] = df

    if summary is not None:
        fig_headline_metrics(summary)
    if per_fruit is not None:
        fig_per_fruit_mae(per_fruit)
    if predictions:
        fig_per_stage_mae(predictions)
        fig_pred_vs_actual(predictions)
        fig_residuals(predictions)
        fig_stage_boxplot(predictions)
    if summary is not None and latency is not None:
        fig_efficiency_tradeoff(summary, latency)
    fig_rf_feature_importance()
    if ablation is not None:
        fig_ablation(ablation)

    print(f"\nDone. {len(list(FIG_DIR.glob('*.png')))} figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
