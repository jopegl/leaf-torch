import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# =========================
# CONFIG
# =========================
CSV_PATH = "dados_filtrados.csv"
OUTPUT_DIR = "./plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRIC_LABELS = {
    "area": "Leaf Area (cm²)",
    "perimeter": "Leaf Perimeter (cm)",
    "length": "Leaf Length (cm)"
}

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

numeric_cols = [
    "real", "annot", "pred",
    "annot_RER", "pred_RER", "method_RER",
    "pattern_size", "dist"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# remove valores inválidos globais básicos
df = df.replace(-1, np.nan)

if "method_RER" not in df.columns:
    df["method_RER"] = np.where(
        df["annot"] == 0,
        0,
        100 * np.abs(df["pred"] - df["annot"]) / df["annot"]
    )

# =========================
# FUNÇÕES
# =========================

def scatter_plot(data, metric, fold=None):
    d = data[data["metric"] == metric].copy()

    if fold is not None:
        d = d[d["fold"] == fold]

    # manter só valores válidos
    d = d.dropna(subset=["real", "annot", "pred"])
    d = d[d["real"] > 0]

    if d.empty:
        return

    label = METRIC_LABELS.get(metric, metric)

    plt.figure(figsize=(7, 5))

    plt.scatter(d["real"], d["annot"], alpha=0.5, label="Annotated Estimate")
    plt.scatter(d["real"], d["pred"], alpha=0.5, label="Model Prediction")

    x = np.array([d["real"].min(), d["real"].max()])
    plt.plot(x, x, linestyle="solid", label="Ideal (y = x)")

    if len(d) > 1:
        m_a, b_a = np.polyfit(d["real"], d["annot"], 1)
        plt.plot(x, m_a * x + b_a, linestyle="dashed", label="Annotated Fit")
        r2_a = r2_score(d["annot"], m_a * d["real"] + b_a)

        m_p, b_p = np.polyfit(d["real"], d["pred"], 1)
        plt.plot(x, m_p * x + b_p, linestyle="dotted", label="Prediction Fit")
        r2_p = r2_score(d["pred"], m_p * d["real"] + b_p)

        plt.text(
            0.05, 0.95,
            f"$R^2$ Annotated = {r2_a:.3f}\n$R^2$ Predicted = {r2_p:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment="top"
        )

    plt.title(f"{label}: Measured vs Estimated")
    plt.xlabel(f"Measured {label}")
    plt.ylabel(f"Estimated {label}")
    plt.legend()

    plt.savefig(f"{OUTPUT_DIR}/{metric}_scatter.png", bbox_inches="tight")
    plt.close()


def histogram(data, metric, col):
    d = data[data["metric"] == metric].copy()

    d = d.dropna(subset=[col])
    d = d[d["real"] > 0]

    if d.empty:
        return

    label = METRIC_LABELS.get(metric, metric)

    plt.figure(figsize=(7, 5))

    if col in ["annot", "pred"]:
        d2 = d.dropna(subset=["annot", "pred"])

        sns.histplot(d2["annot"], kde=True, label="Annotated", alpha=0.5)
        sns.histplot(d2["pred"], kde=True, label="Predicted", alpha=0.5)
        plt.legend()
        plt.title(f"{label}: Distribution of Estimated Values")

    elif "RER" in col:
        sns.histplot(d[col], kde=True)
        plt.title(f"{label}: Relative Error Distribution ({col})")
        plt.xlabel("Relative Error (%)")

    else:
        sns.histplot(d[col], kde=True)
        plt.title(f"{label}: Distribution of {col}")

    plt.savefig(f"{OUTPUT_DIR}/{metric}_hist_{col}.png", bbox_inches="tight")
    plt.close()


def boxplot(data, metric):
    d = data[data["metric"] == metric].copy()

    d = d.dropna(subset=["real", "annot", "pred"])
    d = d[d["real"] > 0]

    if d.empty:
        return

    label = METRIC_LABELS.get(metric, metric)

    plt.figure(figsize=(7, 5))
    d[["real", "annot", "pred"]].boxplot()
    plt.title(f"{label}: Value Distribution (Including Outliers)")
    plt.ylabel(label)
    plt.savefig(f"{OUTPUT_DIR}/{metric}_box_with_outliers.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    d[["real", "annot", "pred"]].boxplot(showfliers=False)
    plt.title(f"{label}: Value Distribution (Outliers Removed)")
    plt.ylabel(label)
    plt.savefig(f"{OUTPUT_DIR}/{metric}_box_no_outliers.png", bbox_inches="tight")
    plt.close()


def grouped_scatter(data, metric, group_col):
    d_all = data[data["metric"] == metric].copy()
    d_all = d_all.dropna(subset=["real", "annot", "pred"])
    d_all = d_all[d_all["real"] > 0]

    groups = d_all[group_col].dropna().unique()

    ncols = 2
    nrows = math.ceil(len(groups) / ncols)

    label = METRIC_LABELS.get(metric, metric)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows)
    )

    axes = axes.flatten()

    for i, g in enumerate(groups):
        d = d_all[d_all[group_col] == g]

        if d.empty:
            continue

        # --- SCATTERS ---
        axes[i].scatter(d["real"], d["annot"], alpha=0.5, s=10, label="Annot")
        axes[i].scatter(d["real"], d["pred"], alpha=0.5, s=10, label="Pred")

        # linha ideal
        axes[i].plot(
            [d["real"].min(), d["real"].max()],
            [d["real"].min(), d["real"].max()],
            linestyle="dashed"
        )

        axes[i].set_title(f"{group_col}: {g}", fontsize=9)
        axes[i].set_xlabel("Measured", fontsize=8)
        axes[i].set_ylabel("Estimated", fontsize=8)
        axes[i].tick_params(axis='both', labelsize=7)

        # legenda só no primeiro gráfico (pra não poluir)
        if i == 0:
            axes[i].legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        f"{label}: Prediction Performance by {group_col}",
        fontsize=11,
        y=1.02
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(f"{OUTPUT_DIR}/{metric}_scatter_by_{group_col}.png", bbox_inches="tight")
    plt.close()


# =========================
# RUN
# =========================

metrics = ["area", "perimeter", "length"]

for m in metrics:
    scatter_plot(df, m)

    histogram(df, m, "annot")
    histogram(df, m, "pred")
    histogram(df, m, "pred_RER")
    histogram(df, m, "method_RER")

    boxplot(df, m)

    grouped_scatter(df, m, "species")
    grouped_scatter(df, m, "dist")
    grouped_scatter(df, m, "pattern_size")

print("✅ Done generating plots")