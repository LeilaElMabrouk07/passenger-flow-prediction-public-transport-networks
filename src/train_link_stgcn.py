
"""
train_link_stgcn.py
-------------------
Entraîne le modèle STGCN sur les liens NUMBAT avec le découpage :
  Train : 2016-2022
  Val   : 2023
  Test  : 2024  (évaluation finale après l'entraînement)
 
Sauvegarde d'un CSV complet :
  cache/training_metrics_stgcn.csv
"""
 
import csv
import time
from pathlib import Path
 
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
 
from build_table import build_or_load_long_table
from config import (
    BATCH_SIZE, CACHE_DIR, EPOCHS, HORIZON, LR, RESULTS_CSV
)
from graph_link import attach_link_ids, build_line_graph_edge_index, build_link_index
from graph_station import edge_index_to_adj
from link_dataset_stgcn import (
    load_global_norm_link, prepare_link_datasets
)
from model_stgcn import STGCN
 
MODEL_PATH_LINK = CACHE_DIR / "stgcn_link_model.pt"
BEST_MODEL_PATH = CACHE_DIR / "stgcn_link_model_best.pt"
 
 
def _eval_loader(loader, model, device, std, mean):
    """
    Évalue le modèle sur un DataLoader.
    Retourne : mae_global, rmse_global, mae_h (list), rmse_h (list)
    """
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
 
    yt = np.concatenate(y_true_all, axis=0)   # [N, H, E]
    yp = np.concatenate(y_pred_all, axis=0)   # [N, H, E]
 
    mae_global  = mean_absolute_error(yt.ravel(), yp.ravel())
    rmse_global = np.sqrt(mean_squared_error(yt.ravel(), yp.ravel()))
 
    mae_h, rmse_h = [], []
    for h in range(yt.shape[1]):
        mae_h.append(mean_absolute_error(yt[:, h, :].ravel(), yp[:, h, :].ravel()))
        rmse_h.append(np.sqrt(mean_squared_error(yt[:, h, :].ravel(), yp[:, h, :].ravel())))
 
    return mae_global, rmse_global, mae_h, rmse_h
 
 
def _write_csv_header(path: Path):
    horizon_cols = []
    for h in range(1, HORIZON + 1):
        horizon_cols += [f"mae_h{h}", f"rmse_h{h}"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "split",
            "mae_global", "rmse_global",
            *horizon_cols,
            "train_mse_epoch",
        ])
 
 
def _append_csv_row(path: Path, epoch: int, split: str,
                    mae_g: float, rmse_g: float,
                    mae_h: list, rmse_h: list,
                    train_mse: float):
    horizon_vals = []
    for h in range(len(mae_h)):
        horizon_vals += [round(mae_h[h], 4), round(rmse_h[h], 4)]
 
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, split,
            round(mae_g,  4), round(rmse_g,  4),
            *horizon_vals,
            round(train_mse, 6),
        ])
 
 
# Main training function

def train_link_model():
    # 1. Données
    df = build_or_load_long_table(force_rebuild=False)
    df = df[[
        "year", "day_code", "Line", "Dir",
        "time_index", "flow", "From Station", "To Station"
    ]].copy()
 
    TARGET_LINES = ["Bakerloo", "H&C and Circle"]
    df = df[df["Line"].isin(TARGET_LINES)].copy()
 
    links  = build_link_index(df)
    df_l   = attach_link_ids(df, links)
    df_l   = df_l[["year", "day_code", "Line", "Dir", "time_index", "flow", "link_id"]].copy()
 
    df_l["year"]       = df_l["year"].astype("int16")
    df_l["day_code"]   = df_l["day_code"].astype("category")
    df_l["Line"]       = df_l["Line"].astype("category")
    df_l["Dir"]        = df_l["Dir"].astype("category")
    df_l["time_index"] = pd.to_numeric(df_l["time_index"], errors="coerce").astype("int16")
 
    # 2. Graphe
    edge_index = build_line_graph_edge_index(links)
    A_norm     = edge_index_to_adj(edge_index, num_nodes=len(links))
 
    # 3. Datasets  (train / val / test)
    train_ds, val_ds, test_ds, common_codes = prepare_link_datasets(
        df_l, num_links=len(links)
    )
 
    print(f"Liens        : {len(links)}")
    print(f"Train windows: {len(train_ds)}")
    print(f"Val   windows: {len(val_ds)}")
    print(f"Test  windows: {len(test_ds)}")
    print(f"Common day_codes: {common_codes}")
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
 
    # 4. Modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_norm = A_norm.to(device)
 
    model = STGCN(
        num_nodes=len(links),
        A_norm=A_norm,
        in_channels=1,
        hidden_channels=32,
        kt=3,
        dropout=0.1,
        horizon=HORIZON,
    ).to(device)
 
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
 
    mean, std = load_global_norm_link()
 
    # 5. CSV 
    _write_csv_header(RESULTS_CSV)
 
    # 6. Boucle d'entraînement
    best_val_mae = float("inf")
 
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        t0_epoch = time.time()
 
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{EPOCHS}")
 
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)
 
            opt.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
 
            train_losses.append(loss.item())
 
            if batch_idx % 20 == 0 or batch_idx == 1:
                avg20 = np.mean(train_losses[-20:])
                print(
                    f"  batch {batch_idx:4d}/{len(train_loader)} "
                    f"| loss={loss.item():.4f} | avg20={avg20:.4f}",
                    flush=True,
                )
 
        train_mse = float(np.mean(train_losses))
        elapsed   = time.time() - t0_epoch
        print(f"Epoch {epoch} finie en {elapsed/60:.2f} min | train_MSE={train_mse:.4f}")
 
        # Validation 
        val_mae_g, val_rmse_g, val_mae_h, val_rmse_h = _eval_loader(
            val_loader, model, device, std, mean
        )
        print(
            f"  [VAL]  MAE={val_mae_g:.2f}  RMSE={val_rmse_g:.2f} "
            f"| MAE/h={[f'{v:.2f}' for v in val_mae_h]}"
        )
 
        _append_csv_row(
            RESULTS_CSV, epoch, "val",
            val_mae_g, val_rmse_g, val_mae_h, val_rmse_h,
            train_mse,
        )
 
        #Sauvegarde best model (sur val MAE)
        if val_mae_g < best_val_mae:
            best_val_mae = val_mae_g
            torch.save({"model": model.state_dict(), "epoch": epoch}, BEST_MODEL_PATH)
            print(f"  Nouveau meilleur modèle (val MAE={best_val_mae:.2f}) → {BEST_MODEL_PATH}")
 
    # 7. Évaluation finale sur Test avec le MEILLEUR modèle
    print(f"\n{'='*50}")
    print("Évaluation finale sur TEST avec le meilleur modèle (val)…")
 
    ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    best_epoch = ckpt.get("epoch", "?")
    print(f"  Meilleur checkpoint : epoch {best_epoch}")
 
    test_mae_g, test_rmse_g, test_mae_h, test_rmse_h = _eval_loader(
        test_loader, model, device, std, mean
    )
 
    print(
        f"  [TEST] MAE={test_mae_g:.2f}  RMSE={test_rmse_g:.2f} "
        f"| MAE/h={[f'{v:.2f}' for v in test_mae_h]}"
    )
 
    _append_csv_row(
        RESULTS_CSV, best_epoch, "test",
        test_mae_g, test_rmse_g, test_mae_h, test_rmse_h,
        train_mse=0.0,   # pas de train_mse pour l'éval finale
    )
 
    # Sauvegarde aussi le dernier modèle (epoch finale)
    torch.save({"model": model.state_dict()}, MODEL_PATH_LINK)
 
    print(f"\nModèle final       : {MODEL_PATH_LINK}")
    print(f"Meilleur modèle    : {BEST_MODEL_PATH}")
    print(f"Métriques CSV      : {RESULTS_CSV}")
 

if __name__ == "__main__":
    train_link_model()