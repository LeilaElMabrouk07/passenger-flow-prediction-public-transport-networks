"""
frame_importance_stgcn.py
Analyse de l'importance temporelle des frames d'entrée
pour le modèle STGCN — perturbation analysis.
ΔMAE = MAE_masked - MAE_baseline (en passagers réels)
"""
 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
 
from build_table import build_or_load_long_table
from graph_link import build_link_index, attach_link_ids, build_line_graph_edge_index
from graph_station import edge_index_to_adj
from link_dataset_stgcn import prepare_link_datasets, load_global_norm_link
from model_stgcn import STGCN
from config import BATCH_SIZE, CACHE_DIR, HORIZON, N_HISTORY
 
MODEL_PATH_LINK = CACHE_DIR / "stgcn_link_model_best.pt"
OUT_CSV         = CACHE_DIR / "frame_importance_stgcn.csv"
 
 
def evaluate_with_mask(loader, model, device, std, mean, mask_frame=None):
    model.eval()
    y_true_all, y_pred_all = [], []
 
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)   
            y = y.to(device)
 
            if mask_frame is not None:
                x = x.clone()
                x[:, :, :, mask_frame] = 0.0
 
            yhat = model(x)
 
            # Dénormalisation 
            y_den    = y.cpu().numpy()    * std + mean
            yhat_den = yhat.cpu().numpy() * std + mean
 
            y_true_all.append(y_den)
            y_pred_all.append(yhat_den)
 
    yt = np.concatenate(y_true_all, axis=0)  
    yp = np.concatenate(y_pred_all, axis=0)
 
    mae_global = mean_absolute_error(yt.ravel(), yp.ravel())
    mse_global = mean_squared_error(yt.ravel(),  yp.ravel())
 
    mae_h, mse_h = [], []
    for h in range(yt.shape[1]):
        mae_h.append(mean_absolute_error(
            yt[:, h, :].ravel(), yp[:, h, :].ravel()))
        mse_h.append(mean_squared_error(
            yt[:, h, :].ravel(), yp[:, h, :].ravel()))
 
    return mae_global, mse_global, mae_h, mse_h
 
 
# Fonction principale 
def run_frame_importance():
 
    # Données
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
 
    # Graphe 
    edge_index = build_line_graph_edge_index(links)
    A_norm     = edge_index_to_adj(edge_index, num_nodes=len(links))
 
    # Datasets
    print("Construction des datasets...")
    _, _, test_ds, _ = prepare_link_datasets(df_l, num_links=len(links))
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, drop_last=False
    )
    print(f"Test windows : {len(test_ds)}")
 
    # Modèle 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_norm = A_norm.to(device)
 
    model = STGCN(
        num_nodes=len(links),
        A_norm=A_norm,
        in_channels=1,
        hidden_channels=32,
        kt=3,
        dropout=0.1,
        horizon=HORIZON
    ).to(device)
 
    ckpt = torch.load(MODEL_PATH_LINK, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Modèle chargé : {MODEL_PATH_LINK}")
 
    mean, std = load_global_norm_link()
 
    # Baseline sans masque 
    print("\nCalcul baseline (sans masque)...")
    base_mae, base_mse, base_mae_h, base_mse_h = evaluate_with_mask(
        test_loader, model, device, std, mean,
        mask_frame=None
    )
    print(f"Baseline — MAE={base_mae:.4f}  MSE={base_mse:.4f}")
    print(f"MAE par horizon : {['%.4f' % v for v in base_mae_h]}")
 
    # Importance par frame 
    print("\nAnalyse par frame...")
    rows = []
 
    for frame in range(N_HISTORY):
        if frame < N_HISTORY - 1:
            label = f"t-{N_HISTORY - 1 - frame}"
        else:
            label = "t (most recent)"
 
        mae_m, mse_m, mae_mh, mse_mh = evaluate_with_mask(
            test_loader, model, device, std, mean,
            mask_frame=frame
        )
 
        # ΔMAE = MAE_masked - MAE_baseline (en passagers réels)
        delta_mae = mae_m - base_mae
        delta_mse = mse_m - base_mse
 
        row = {
            "frame"     : frame,
            "label"     : label,
            "delta_mae" : round(delta_mae, 4),
            "delta_mse" : round(delta_mse, 4),
        }
 
        # ΔMAE par horizon
        for h in range(HORIZON):
            delta_mae_h = mae_mh[h] - base_mae_h[h]
            row[f"delta_mae_h{h+1}"] = round(delta_mae_h, 4)
 
        rows.append(row)
        print(
            f"Frame {frame:2d} ({label:15s}) | "
            f"ΔMAE={delta_mae:8.4f} | "
            f"ΔMSE={delta_mse:10.4f} | "
            f"H={['%.4f' % (mae_mh[h] - base_mae_h[h]) for h in range(HORIZON)]}",
            flush=True
        )
 
    # Sauvegarde CSV 
    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("frame").reset_index(drop=True)
    results_df.to_csv(OUT_CSV, index=False)
 
    print(f"\n Résultats sauvegardés : {OUT_CSV}")
    print("\n" + results_df.to_string(index=False))
 
    print(f"\n{'='*60}")
    print(f"Baseline MAE = {base_mae:.4f} passagers")
    print(f"{'='*60}")
    print(f"{'Frame':<15} {'ΔMAE':>10} {'ΔMSE':>12}")
    print(f"{'-'*40}")
    for row in rows:
        print(
            f"{row['label']:<15} "
            f"{row['delta_mae']:>10.4f} "
            f"{row['delta_mse']:>12.4f}"
        )
 
 
if __name__ == "__main__":
    run_frame_importance()