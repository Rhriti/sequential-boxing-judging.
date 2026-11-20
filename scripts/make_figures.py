"""
Generate publication-quality figures for the boxing judging paper from results/.

Outputs:
  figures/fig1_attention_by_round.pdf & .png
  figures/fig2_heldout_metrics.pdf & .png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication style configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
})


def make_fig1_attention():
    csv_path = RESULTS_DIR / "stage3_attention_by_round.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(3.45, 2.65))

    rounds = df["round"].to_numpy()
    w_hist = df["w_historical_mean"].to_numpy()
    w_imm = df["w_immediate_mean"].to_numpy()

    ax.plot(
        rounds,
        w_hist,
        color="#1f77b4",
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        label=r"Historical Memory ($w_{\mathrm{hist}}$)",
    )
    ax.plot(
        rounds,
        w_imm,
        color="#d62728",
        marker="s",
        markersize=4.5,
        linewidth=1.6,
        linestyle="--",
        label=r"Immediate Round ($w_{\mathrm{imm}}$)",
    )

    # Highlight early rounds and championship rounds
    ax.axvspan(1.7, 3.3, color="#1f77b4", alpha=0.08, label="Early (Rounds 2–3, mean=0.418)")
    ax.axvspan(10.7, 12.3, color="#7f7f7f", alpha=0.12, label="Championship (Rounds 11–12, mean=0.307)")

    ax.set_xlabel("Round Number ($r$)")
    ax.set_ylabel("Attention Gate Weight")
    ax.set_title("CBSN Attention Gate Dynamics Across Rounds")
    ax.set_xticks(rounds)
    ax.set_ylim(0.20, 0.78)
    ax.legend(loc="center right", frameon=True, framealpha=0.9, edgecolor="none")

    fig.tight_layout()

    pdf_out = FIGURES_DIR / "fig1_attention_by_round.pdf"
    png_out = FIGURES_DIR / "fig1_attention_by_round.png"
    fig.savefig(pdf_out)
    fig.savefig(png_out)
    plt.close(fig)
    print(f"Wrote {pdf_out} and {png_out}")


def make_fig2_heldout_metrics():
    xgb_path = RESULTS_DIR / "stage2_xgboost.json"
    cbsn_path = RESULTS_DIR / "stage3_cv_metrics.json"

    with open(xgb_path, encoding="utf-8") as f:
        xgb_data = json.load(f)
    with open(cbsn_path, encoding="utf-8") as f:
        cbsn_data = json.load(f)

    folds = list(range(1, 6))
    xgb_loss = [f["log_loss"] for f in xgb_data["folds"]]
    cbsn_loss = [f["log_loss"] for f in cbsn_data["folds"]]
    xgb_acc = [f["accuracy"] * 100 for f in xgb_data["folds"]]
    cbsn_acc = [f["accuracy"] * 100 for f in cbsn_data["folds"]]

    pooled_xgb_loss = xgb_data["pooled"]["log_loss"]
    pooled_cbsn_loss = cbsn_data["pooled"]["log_loss"]
    pooled_xgb_acc = xgb_data["pooled"]["accuracy"] * 100
    pooled_cbsn_acc = cbsn_data["pooled"]["accuracy"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.5))

    # --- Subplot 1: Log Loss ---
    x = np.arange(len(folds))
    width = 0.35

    ax1.bar(x - width / 2, xgb_loss, width, label="XGBoost", color="#2ca02c", alpha=0.85)
    ax1.bar(x + width / 2, cbsn_loss, width, label="CBSN (LSTM)", color="#1f77b4", alpha=0.85)

    ax1.axhline(pooled_xgb_loss, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.8, label=f"XGB Pooled ({pooled_xgb_loss:.3f})")
    ax1.axhline(pooled_cbsn_loss, color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.8, label=f"CBSN Pooled ({pooled_cbsn_loss:.3f})")

    ax1.set_xlabel("Cross-Validation Fold")
    ax1.set_ylabel("Held-Out Log Loss (lower is better)")
    ax1.set_title("(a) Held-Out Log Loss by Fold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Fold {i}" for i in folds])
    ax1.set_ylim(0.40, 0.53)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.85, fontsize=7)

    # --- Subplot 2: Accuracy ---
    ax2.bar(x - width / 2, xgb_acc, width, label="XGBoost", color="#2ca02c", alpha=0.85)
    ax2.bar(x + width / 2, cbsn_acc, width, label="CBSN (LSTM)", color="#1f77b4", alpha=0.85)

    ax2.axhline(pooled_xgb_acc, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.8, label=f"XGB Pooled ({pooled_xgb_acc:.1f}%)")
    ax2.axhline(pooled_cbsn_acc, color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.8, label=f"CBSN Pooled ({pooled_cbsn_acc:.1f}%)")

    ax2.set_xlabel("Cross-Validation Fold")
    ax2.set_ylabel("Held-Out Accuracy (%)")
    ax2.set_title("(b) Held-Out Accuracy by Fold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Fold {i}" for i in folds])
    ax2.set_ylim(72.0, 79.5)
    ax2.legend(loc="lower left", frameon=True, framealpha=0.85, fontsize=7)

    fig.tight_layout()

    pdf_out = FIGURES_DIR / "fig2_heldout_metrics.pdf"
    png_out = FIGURES_DIR / "fig2_heldout_metrics.png"
    fig.savefig(pdf_out)
    fig.savefig(png_out)
    plt.close(fig)
    print(f"Wrote {pdf_out} and {png_out}")


def main():
    print("Generating figures for boxing judging paper...")
    make_fig1_attention()
    make_fig2_heldout_metrics()
    print("Done generating figures.")


if __name__ == "__main__":
    main()
