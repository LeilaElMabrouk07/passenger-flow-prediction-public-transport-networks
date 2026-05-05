"""
plot_convergence.py
-------------------
Génère la courbe de convergence (Validation MAE par epoch)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parents[1] / "cache" / "training_metrics_stgcn.csv"
OUT_PATH = Path(__file__).resolve().parents[1] / "cache" / "convergence_curve.png"

# Chargement 
df     = pd.read_csv(CSV_PATH)
val_df = df[df["split"] == "val"].sort_values("epoch").reset_index(drop=True)

epochs  = val_df["epoch"].values
mae_val = val_df["mae_global"].values

best_idx   = val_df["mae_global"].idxmin()
best_epoch = val_df.loc[best_idx, "epoch"]
best_mae   = val_df.loc[best_idx, "mae_global"]

plt.rcParams.update({
    "font.family"       : "serif",
    "font.serif"        : ["Times New Roman", "DejaVu Serif"],
    "font.size"         : 14,
    "axes.labelsize"    : 14,
    "axes.titlesize"    : 16,
    "xtick.labelsize"   : 12,
    "ytick.labelsize"   : 12,
    "legend.fontsize"   : 12,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

fig, ax = plt.subplots(figsize=(8, 6))

ax.set_title(
    "Training Convergence of STGCN on NUMBAT Dataset",
    fontsize=16,
    fontweight="bold",
    pad=10
)

# Courbe Validation MAE 
ax.plot(
    epochs, mae_val,
    color="#2166ac", linewidth=2,
    marker="o", markersize=4,
    markerfacecolor="white", markeredgewidth=1.2,
    label="Validation MAE",
    zorder=3
)

ax.axhline(y=best_mae, color="gray", linestyle=":",
           linewidth=1.0, alpha=0.6)
ax.axvline(x=best_epoch, color="#d6604d", linestyle="--",
           linewidth=1.5, alpha=0.85)

# Point meilleur checkpoint 
ax.scatter([best_epoch], [best_mae],
           color="#d6604d", zorder=5, s=100,
           label=f"Best checkpoint (epoch {int(best_epoch)}, MAE = {best_mae:.2f})")

ax.annotate(
    f"epoch {int(best_epoch)}\nMAE = {best_mae:.2f}",
    xy=(best_epoch, best_mae),
    xytext=(best_epoch + 2.5, best_mae + 2.8),
    fontsize=12,
    color="#d6604d",
    arrowprops=dict(
        arrowstyle="->",
        color="#d6604d",
        lw=1.2,
        connectionstyle="arc3,rad=0.2"
    )
)

ax.set_xlabel("Epoch", fontsize=14, labelpad=6)
ax.set_ylabel("Validation MAE", fontsize=14, labelpad=6)
ax.set_xlim(1, epochs[-1])
ax.set_ylim(0, mae_val.max() * 1.15)
ax.set_xticks(np.arange(0, epochs[-1] + 1, 5))
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35, color="gray")

ax.legend(
    loc="upper right",
    fontsize=12,
    framealpha=0.92,
    edgecolor="lightgray",
    borderpad=0.8
)

plt.tight_layout(pad=1.2)
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
print(f" Figure sauvegardée : {OUT_PATH}")
plt.show()