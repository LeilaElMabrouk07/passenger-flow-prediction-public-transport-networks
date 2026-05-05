"""
plot_scatter_bias.py
Scatter plot réel vs prédit pour chaque horizon (H1-H4)
"""
 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
 
from build_table import build_or_load_long_table
from graph_link import build_link_index, attach_link_ids, build_line_graph_edge_index
from graph_station import edge_index_to_adj
from link_dataset_stgcn import prepare_link_datasets, load_global_norm_link
from model_stgcn import STGCN
from config import BATCH_SIZE, CACHE_DIR, HORIZON, N_HISTORY
 
MODEL_PATH_LINK = CACHE_DIR / "stgcn_link_model_best.pt"
OUT_PATH        = CACHE_DIR / "scatter_bias.png"
 
HORIZON_LABELS = ["+15 min", "+30 min", "+45 min", "+60 min"]
 
 
def collect_predictions(loader, model, device, std, mean):
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            yhat = model(x)
            y_den    = y.cpu().numpy()    * std + mean
            yhat_den = yhat.cpu().numpy() * std + mean
            y_true_all.append(y_den)
            y_pred_all.append(yhat_den)
    yt = np.concatenate(y_true_all, axis=0)
    yp = np.concatenate(y_pred_all, axis=0)
    return yt, yp
 
 
def plot_scatter_bias(yt, yp):
    plt.rcParams.update({
        "font.family"       : "serif",
        "font.serif"        : ["Times New Roman", "DejaVu Serif"],
        "font.size"         : 10,
        "axes.labelsize"    : 10,
        "xtick.labelsize"   : 9,
        "ytick.labelsize"   : 9,
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
    })
 
    fig, axes = plt.subplots(2, 2, figsize=(7, 7.2))
    axes = axes.flatten()
 
    for h in range(HORIZON):
        ax = axes[h]
 
        y_true_h = yt[:, h, :].ravel()
        y_pred_h = yp[:, h, :].ravel()
 
        if len(y_true_h) > 5000:
            idx = np.random.choice(len(y_true_h), 5000, replace=False)
            y_true_h = y_true_h[idx]
            y_pred_h = y_pred_h[idx]
 
        vmax = max(y_true_h.max(), y_pred_h.max()) * 1.05
        vmin = 0
 
        # Scatter
        ax.scatter(y_true_h, y_pred_h,
                   alpha=0.15, s=4, color="#2166ac", rasterized=True)
 
        # Diagonale parfaite
        ax.plot([vmin, vmax], [vmin, vmax],
                color="red", linewidth=1.5, linestyle="--",
                label="Perfect prediction")
 
        # Régression
        coeffs = np.polyfit(y_true_h, y_pred_h, 1)
        x_reg  = np.linspace(vmin, vmax, 100)
        y_reg  = np.polyval(coeffs, x_reg)
        ax.plot(x_reg, y_reg,
                color="darkorange", linewidth=1.5, linestyle="-",
                label=f"Regression (slope={coeffs[0]:.2f})")
 
        # Biais
        bias_h = np.mean(y_pred_h - y_true_h)
        ax.text(
            0.05, 0.92,
            f"Bias = {bias_h:+.2f}",
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", alpha=0.8,
                      edgecolor="lightgray")
        )
 
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel("Real passenger flow")
        ax.set_ylabel("Predicted passenger flow")
        ax.set_title(f"Horizon {HORIZON_LABELS[h]}", fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
 
    fig.suptitle(
        "Prediction Bias Analysis — STGCN on NUMBAT 2024 Test Set",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Figure sauvegardée : {OUT_PATH}")
    plt.show()
 
 
def run_scatter_bias():
 
    print("Chargement des données...")
    df = build_or_load_long_table(force_rebuild=False)
    df = df[[
        "year", "day_code", "Line", "Dir",
        "time_index", "flow",
        "From Station", "To Station"
    ]].copy()
 
    TARGET_LINES = ["Bakerloo", "H&C and Circle"]
    df = df[df["Line"].isin(TARGET_LINES)].copy()
 
    links = build_link_index(df)
    df_l  = attach_link_ids(df, links)
    df_l  = df_l[[
        "year", "day_code", "Line", "Dir",
        "time_index", "flow", "link_id"
    ]].copy()
 
    df_l["year"]       = df_l["year"].astype("int16")
    df_l["day_code"]   = df_l["day_code"].astype("category")
    df_l["Line"]       = df_l["Line"].astype("category")
    df_l["Dir"]        = df_l["Dir"].astype("category")
    df_l["time_index"] = pd.to_numeric(
        df_l["time_index"], errors="coerce").astype("int16")
 
    edge_index = build_line_graph_edge_index(links)
    A_norm     = edge_index_to_adj(edge_index, num_nodes=len(links))
 
    print("Construction dataset test...")
    _, _, test_ds, _ = prepare_link_datasets(df_l, num_links=len(links))
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, drop_last=False
    )
    print(f"Test windows : {len(test_ds)}")
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_norm = A_norm.to(device)
 
    model = STGCN(
        num_nodes=len(links), A_norm=A_norm,
        in_channels=1, hidden_channels=32,
        kt=3, dropout=0.1, horizon=HORIZON
    ).to(device)
 
    ckpt = torch.load(MODEL_PATH_LINK, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Modèle chargé : {MODEL_PATH_LINK}")
 
    mean, std = load_global_norm_link()
 
    print("Collecte des prédictions sur le test set...")
    yt, yp = collect_predictions(test_loader, model, device, std, mean)
 
    print(f"\n{'='*55}")
    print(f"{'Horizon':<12} {'Bias (mean error)':>20}")
    print(f"{'-'*40}")
    for h in range(HORIZON):
        bias_h = np.mean(yp[:, h, :].ravel() - yt[:, h, :].ravel())
        print(f"{HORIZON_LABELS[h]:<12} {bias_h:>+20.2f}")
 
    print("\nGénération de la figure...")
    plot_scatter_bias(yt, yp)
 
 
if __name__ == "__main__":
    run_scatter_bias()