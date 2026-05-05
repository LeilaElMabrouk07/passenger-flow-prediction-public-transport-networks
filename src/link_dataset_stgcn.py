"""
link_dataset_stgcn.py
---------------------
Construit les datasets train / val / test à partir des données NUMBAT.

  Train  : TRAIN_YEARS  (2016-2022)
  Val    : VAL_YEAR     (2023)
  Test   : TEST_YEAR    (2024)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import TRAIN_YEARS, VAL_YEAR, TEST_YEAR, N_HISTORY, HORIZON, STRIDE, CACHE_DIR

NORM_PATH_LINK = CACHE_DIR / "stgcn_link_norm.npz"



def build_link_tensor_day(df_link: pd.DataFrame, num_links: int) -> np.ndarray:
   
    df = df_link[["time_index", "link_id", "flow"]].dropna().copy()
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce")
    df = df.dropna(subset=["flow"])

    pivot = (
        df.pivot_table(index="time_index", columns="link_id", values="flow", aggfunc="sum")
        .sort_index()
    )
    pivot = pivot.reindex(columns=range(num_links))
    pivot = pivot.ffill().fillna(0.0)
    return pivot.to_numpy(dtype=np.float32)   


def compute_global_norm_link(df_link_ids: pd.DataFrame):
    """Calcule moyenne / écart-type sur les années d'entraînement uniquement."""
    s = ss = 0.0
    n = 0
    for year, g in df_link_ids.groupby("year", sort=False):
        if int(year) not in TRAIN_YEARS:
            continue
        v = pd.to_numeric(g["flow"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if v.size == 0:
            continue
        s  += v.sum()
        ss += (v * v).sum()
        n  += v.size

    mean = float(s / max(n, 1))
    var  = float(ss / max(n, 1) - mean * mean)
    std  = float(np.sqrt(max(var, 1e-12)))
    if std < 1e-6:
        std = 1.0

    np.savez(NORM_PATH_LINK, mean=mean, std=std)
    return mean, std


def load_global_norm_link():
    d = np.load(NORM_PATH_LINK)
    return float(d["mean"]), float(d["std"])


# Dataset
class STGCNLinkWindowDataset(Dataset):
    def __init__(self, tensors_by_key, mean: float, std: float):
        self.items: list = []
        self.mean  = mean
        self.std   = std

        for _key, X in tensors_by_key.items():
            T, E = X.shape
            Xn = (X - mean) / std

            for t in range(0, T - (N_HISTORY + HORIZON) + 1, STRIDE):
                xin   = Xn[t : t + N_HISTORY,          :]   
                y_seq = Xn[t + N_HISTORY : t + N_HISTORY + HORIZON, :]  
                self.items.append((xin, y_seq))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        xin, y_seq = self.items[idx]
        x = torch.tensor(xin,   dtype=torch.float32).T.unsqueeze(0) 
        y = torch.tensor(y_seq, dtype=torch.float32)               
        return x, y

# Prepare datasets  (train / val / test)

def prepare_link_datasets(df_link_ids: pd.DataFrame, num_links: int):
    """
    Retourne : train_ds, val_ds, test_ds, common_codes
    """
    ALL_EVAL_YEARS = {VAL_YEAR, TEST_YEAR}

    meta = (
        df_link_ids[["year", "day_code"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    train_codes = set(meta[meta["year"].isin(TRAIN_YEARS)]["day_code"].astype(str).unique())
    val_codes   = set(meta[meta["year"] == VAL_YEAR  ]["day_code"].astype(str).unique())
    test_codes  = set(meta[meta["year"] == TEST_YEAR ]["day_code"].astype(str).unique())

    common_codes = sorted(train_codes & val_codes & test_codes)
    if not common_codes:
        common_codes = sorted(train_codes & (val_codes | test_codes))
    if not common_codes:
        raise ValueError(
            "Aucun day_code commun entre train, val et test pour link-dataset."
        )
    
    tensors_train: dict = {}
    tensors_val:   dict = {}
    tensors_test:  dict = {}

    grouped = df_link_ids.groupby(["year", "day_code", "Line", "Dir"], sort=False)

    for (year, day_code, line, direction), g in grouped:
        year_i = int(year)
        if year_i not in TRAIN_YEARS and year_i not in ALL_EVAL_YEARS:
            continue
        if str(day_code) not in common_codes:
            continue

        X   = build_link_tensor_day(g, num_links)
        key = (year_i, str(day_code), str(line), str(direction))

        if year_i in TRAIN_YEARS:
            tensors_train[key] = X
        elif year_i == VAL_YEAR:
            tensors_val[key]   = X
        elif year_i == TEST_YEAR:
            tensors_test[key]  = X

    # Normalisation
    mean, std = compute_global_norm_link(df_link_ids)

    train_ds = STGCNLinkWindowDataset(tensors_train, mean, std)
    val_ds   = STGCNLinkWindowDataset(tensors_val,   mean, std)
    test_ds  = STGCNLinkWindowDataset(tensors_test,  mean, std)

    return train_ds, val_ds, test_ds, common_codes

