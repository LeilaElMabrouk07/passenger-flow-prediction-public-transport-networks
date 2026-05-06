import pandas as pd
import numpy as np
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

csv_path = "london_fri_16_24.csv"

K_IN = 8
H = 4
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3

HIDDEN = 64
DROPOUT = 0.2

DIFFUSION_STEPS = 2
USE_BIDIR = True
TEACHER_FORCING = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(0)
torch.manual_seed(0)

df = pd.read_csv(csv_path)


meta_cols = [
    "year",               # IMPORTANT
    "Link", "Line", "Dir", "Order",
    "From NLC", "From ASC", "From Station",
    "To NLC", "To ASC", "To Station",
]

time_cols_all = [c for c in df.columns if c not in meta_cols]

for col in time_cols_all:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("\u202f", "", regex=False)   # espace insécable fine
        .str.replace(" ", "", regex=False)        # espaces
        .str.replace(",", ".", regex=False)       # virgule décimale -> point
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

all_years = sorted(df["year"].unique().tolist())
print("Années dispo :", all_years)

test_years = [24]   
train_years = [y for y in all_years if y not in test_years]

print("Train years:", train_years)
print("Test years :", test_years)

ref_year = train_years[0]
df_ref = df[df["year"] == ref_year].copy()

link_id_cols = [
    "Link", "Line", "Dir",
    "From Station", "To Station"
]
df_ref = df_ref.sort_values(link_id_cols).reset_index(drop=True)

N = df_ref.shape[0]
print(f"Nombre de liens de référence N = {N}")

def make_link_key(df_in, cols):
    return df_in[cols].astype(str).agg("||".join, axis=1)

df_ref = df[df["year"] == ref_year].copy().sort_values(link_id_cols).reset_index(drop=True)
df_ref["link_key"] = make_link_key(df_ref, link_id_cols)

ref_keys = df_ref["link_key"].tolist()
N = len(ref_keys)

link_meta_cols = [
    "link_key",
    "Link", "Line", "Dir", "Order",
    "From NLC", "From ASC", "From Station",
    "To NLC", "To ASC", "To Station",
]

link_reference = (
    df_ref[link_meta_cols]
    .copy()
    .reset_index(drop=True)
)

link_reference["link_idx"] = np.arange(len(link_reference))

idx_to_linkkey = dict(zip(link_reference["link_idx"], link_reference["link_key"]))
linkkey_to_idx = dict(zip(link_reference["link_key"], link_reference["link_idx"]))

print(link_reference.head())

if df_ref["link_key"].duplicated().any():
    dups = df_ref.loc[df_ref["link_key"].duplicated(), "link_key"].tolist()
    raise ValueError(f"Doublons de liens dans l'année de référence {ref_year}: {dups[:10]}")

ref_keys = df_ref["link_key"].tolist()
N = len(ref_keys)

aligned_year_dfs = {}

for y in all_years:
    df_y = df[df["year"] == y].copy()
    df_y["link_key"] = make_link_key(df_y, link_id_cols)

    # sécurité : pas de doublon par année
    if df_y["link_key"].duplicated().any():
        dup_count = int(df_y["link_key"].duplicated().sum())
        dup_examples = df_y.loc[df_y["link_key"].duplicated(), "link_key"].tolist()[:10]
        raise ValueError(
            f"Année {y}: {dup_count} doublon(s) de liens. Exemples: {dup_examples}"
        )

    df_y = df_y.set_index("link_key")

    missing = [k for k in ref_keys if k not in df_y.index]
    extra = [k for k in df_y.index if k not in ref_keys]

    if missing:
        print(f"Année {y}: {len(missing)} lien(s) manquant(s) par rapport à la référence.")
        print("Exemples manquants:", missing[:10])

    if extra:
        print(f"Année {y}: {len(extra)} lien(s) en trop par rapport à la référence.")
        print("Exemples en trop:", extra[:10])

    if missing:
        raise ValueError(
            f"Année {y}: impossible d'aligner car certains liens de la référence sont absents."
        )

    df_y = df_y.loc[ref_keys].reset_index()

    aligned_year_dfs[y] = df_y

print("Toutes les années ont été réalignées sur le même ordre de liens.")


from_stations = df_ref["From Station"].astype(str).tolist()
to_stations = df_ref["To Station"].astype(str).tolist()

station_to_links = {}
for idx, (fs, ts) in enumerate(zip(from_stations, to_stations)):
    for st in (fs, ts):
        station_to_links.setdefault(st, []).append(idx)

edges = []
for st, link_idx in station_to_links.items():
    if len(link_idx) < 2:
        continue
    for i, j in combinations(link_idx, 2):
        edges.append((i, j))
        edges.append((j, i))

if len(edges) == 0:
    raise ValueError("Aucune arête construite : vérifier les stations.")

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
E = edge_index.shape[1]
print(f"E arêtes = {E}")

df_graph = df_ref.copy().reset_index(drop=True)

df_graph["Line"] = df_graph["Line"].astype(str)
df_graph["Dir"] = df_graph["Dir"].astype(str)
df_graph["Order"] = pd.to_numeric(df_graph["Order"], errors="coerce")

if df_graph["Order"].isna().any():
    bad_rows = df_graph[df_graph["Order"].isna()][["Link", "Line", "Dir", "From Station", "To Station"]]
    raise ValueError(
        "Certaines valeurs de 'Order' sont manquantes ou non numériques.\n"
        f"Exemples:\n{bad_rows.head()}"
    )

from_stations = df_graph["From Station"].astype(str).tolist()
to_stations   = df_graph["To Station"].astype(str).tolist()

edges = set()

for (line, direction), group in df_graph.groupby(["Line", "Dir"], sort=False):
    group = group.sort_values("Order").reset_index()
    idxs = group["index"].tolist()
    orders = group["Order"].tolist()

    for k in range(len(idxs) - 1):
        i = idxs[k]
        j = idxs[k + 1]
        if orders[k + 1] > orders[k]:
            edges.add((i, j))

for i in range(N):
    for j in range(N):
        if i == j:
            continue
        if to_stations[i] == from_stations[j]:
            edges.add((i, j))

if len(edges) == 0:
    raise ValueError("Aucune arête orientée construite.")

edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
E = edge_index.shape[1]
print(f"E arêtes orientées hybrides = {E}")


def build_random_walk_matrices(num_nodes: int, edge_index: torch.Tensor, add_self_loops: bool = True):
    src, dst = edge_index[0], edge_index[1]
    values = torch.ones(src.shape[0], dtype=torch.float32)

    if add_self_loops:
        self_idx = torch.arange(num_nodes, dtype=torch.long)
        src = torch.cat([src, self_idx])
        dst = torch.cat([dst, self_idx])
        values = torch.cat([values, torch.ones(num_nodes, dtype=torch.float32)])

    A = torch.sparse_coo_tensor(
        indices=torch.stack([src, dst], dim=0),
        values=values,
        size=(num_nodes, num_nodes),
    ).coalesce()

    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.0)
    inv_deg = 1.0 / deg

    idx = A.indices()
    val = A.values()
    row = idx[0]
    val_norm = val * inv_deg[row]
    P = torch.sparse_coo_tensor(idx, val_norm, size=A.size()).coalesce()

    AT = torch.sparse_coo_tensor(
        indices=torch.stack([dst, src], dim=0),
        values=values,
        size=(num_nodes, num_nodes),
    ).coalesce()

    deg_t = torch.sparse.sum(AT, dim=1).to_dense().clamp(min=1.0)
    inv_deg_t = 1.0 / deg_t

    idx_t = AT.indices()
    val_t = AT.values()
    row_t = idx_t[0]
    val_t_norm = val_t * inv_deg_t[row_t]
    P_T = torch.sparse_coo_tensor(idx_t, val_t_norm, size=AT.size()).coalesce()

    return P, P_T

def sparse_matmul(P: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    B, N, C = X.shape
    out = []
    for b in range(B):
        out.append(torch.sparse.mm(P, X[b]))
    return torch.stack(out, dim=0)

P_fwd, P_bwd = build_random_walk_matrices(N, edge_index, add_self_loops=True)
P_fwd = P_fwd.to(device)
P_bwd = P_bwd.to(device)

def get_year_matrix(df_all, year_value, link_sort_cols, time_cols):
    df_y = (
        df_all[df_all["year"] == year_value]
        .copy()
        .sort_values(link_sort_cols)
        .reset_index(drop=True)
    )
    return df_y[time_cols].to_numpy(dtype=np.float32)   # (N, Traw)

flows_by_year_raw = {}
for y in all_years:
    flows_by_year_raw[y] = get_year_matrix(df, y, link_id_cols, time_cols_all)

Traw = flows_by_year_raw[all_years[0]].shape[1]
print(f"N liens = {N} | Traw = {Traw}")

nan_counts = df[time_cols_all].isna().sum()
bad_cols = nan_counts[nan_counts > 0]

print("Colonnes avec NaN après conversion :")
print(bad_cols.sort_values(ascending=False))

train_stack = np.stack([flows_by_year_raw[y] for y in train_years], axis=0)  # (Ytrain, N, Traw)
col_sums = train_stack.sum(axis=(0, 1))   # somme sur années et liens -> (Traw,)
mask_nonzero = col_sums > 0.0

if mask_nonzero.sum() <= K_IN + H:
    raise ValueError("Trop peu de colonnes non nulles après filtrage.")

time_cols = [c for c, keep in zip(time_cols_all, mask_nonzero) if keep]

flows_by_year = {}
for y in all_years:
    flows_by_year[y] = flows_by_year_raw[y][:, mask_nonzero]   # (N, T)

T = flows_by_year[all_years[0]].shape[1]
print(f"T après filtrage = {T}")

train_all_values = np.concatenate(
    [flows_by_year[y].reshape(-1) for y in train_years],
    axis=0
)

global_mean = train_all_values.mean()
global_std = train_all_values.std() + 1e-6

flows_norm_by_year = {}
for y in all_years:
    flows_norm_by_year[y] = (flows_by_year[y] - global_mean) / global_std

print(f"mean = {global_mean:.4f}, std = {global_std:.4f}")

def build_windows_for_one_year(flows_norm_year, k_in, horizon):
    """
    flows_norm_year: (N, T)
    return:
      X_year: (S_year, K_IN, N)
      Y_year: (S_year, H, N)
    """
    Xs, Ys = [], []
    N_local, T_local = flows_norm_year.shape

    for t in range(k_in, T_local - horizon + 1):
        x_t = flows_norm_year[:, t - k_in : t]   # (N, K_IN)
        y_t = flows_norm_year[:, t : t + horizon]  # (N, H)

        Xs.append(x_t.T)   # (K_IN, N)
        Ys.append(y_t.T)   # (H, N)

    X_year = torch.tensor(np.stack(Xs, axis=0), dtype=torch.float32)
    Y_year = torch.tensor(np.stack(Ys, axis=0), dtype=torch.float32)
    return X_year, Y_year

X_train_list, Y_train_list = [], []
X_test_list, Y_test_list = [], []

train_year_datasets = {}
test_year_datasets = {}

for y in train_years:
    X_y, Y_y = build_windows_for_one_year(flows_norm_by_year[y], K_IN, H)
    train_year_datasets[y] = (X_y, Y_y)
    X_train_list.append(X_y)
    Y_train_list.append(Y_y)
    print(f"Train year {y}: X={tuple(X_y.shape)} Y={tuple(Y_y.shape)}")

for y in test_years:
    X_y, Y_y = build_windows_for_one_year(flows_norm_by_year[y], K_IN, H)
    test_year_datasets[y] = (X_y, Y_y)
    X_test_list.append(X_y)
    Y_test_list.append(Y_y)
    print(f"Test year {y}: X={tuple(X_y.shape)} Y={tuple(Y_y.shape)}")

X_train = torch.cat(X_train_list, dim=0)
Y_train = torch.cat(Y_train_list, dim=0)

X_test = torch.cat(X_test_list, dim=0)
Y_test = torch.cat(Y_test_list, dim=0)

S_train = X_train.shape[0]
val_ratio = 1/8
train_end = int((1 - val_ratio) * S_train)

X_val = X_train[train_end:]
Y_val = Y_train[train_end:]

X_train = X_train[:train_end]
Y_train = Y_train[:train_end]

print(f"\nTrain final: X={tuple(X_train.shape)} Y={tuple(Y_train.shape)}")
print(f"Val final  : X={tuple(X_val.shape)} Y={tuple(Y_val.shape)}")
print(f"Test final : X={tuple(X_test.shape)} Y={tuple(Y_test.shape)}")

class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(SeqDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SeqDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(SeqDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)

mean_t = torch.tensor(global_mean, dtype=torch.float32, device=device)
std_t = torch.tensor(global_std, dtype=torch.float32, device=device)


class DiffusionConv(nn.Module):
    """
    Diffusion convolution:
    concat( X, P X, P^2 X, ... ) (et optionnellement backward) puis Linear.
    X: (B, N, Cin)
    """
    def __init__(self, cin, cout, K, use_bidir=True):
        super().__init__()
        self.cin = cin
        self.cout = cout

        self.K = K
        self.use_bidir = use_bidir

        n_terms = (K + 1) * (2 if use_bidir else 1)
        self.lin = nn.Linear(n_terms * cin, cout)


    def forward(self, X, P_fwd, P_bwd):
        B, N, Cin = X.shape
        feats = []

        # forward terms
        Xk = X
        feats.append(Xk)  # P^0 X
        for k in range(1, self.K + 1):
            Xk = sparse_matmul(P_fwd, Xk)
            feats.append(Xk)

        # backward terms
        if self.use_bidir:
            Xk = X
            feats.append(Xk)  
            for k in range(1, self.K + 1):
                Xk = sparse_matmul(P_bwd, Xk)
                feats.append(Xk)

        Hcat = torch.cat(feats, dim=-1)  
        out = self.lin(Hcat)             
        return out

class DCRNNCell(nn.Module):
    """
    GRU-like cell avec diffusion conv pour les gates.
    """
    def __init__(self, input_dim, hidden_dim, K, use_bidir=True, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.conv_zr = DiffusionConv(input_dim + hidden_dim, 2 * hidden_dim, K, use_bidir)
        self.conv_h = DiffusionConv(input_dim + hidden_dim, hidden_dim, K, use_bidir)

        self.loss_train = []

        self.loss_val = []

    def forward(self, x_t, h_prev, P_fwd, P_bwd):
        inp = torch.cat([x_t, h_prev], dim=-1)

        zr = self.conv_zr(inp, P_fwd, P_bwd)
        z, r = torch.split(zr, self.hidden_dim, dim=-1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        inp_h = torch.cat([x_t, r * h_prev], dim=-1)
        h_tilde = torch.tanh(self.conv_h(inp_h, P_fwd, P_bwd))

        h = (1 - z) * h_prev + z * h_tilde
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

class DCRNN(nn.Module):
    """
    Encoder-Decoder DCRNN simple.
    - Encoder lit K_IN pas
    - Decoder génère H pas
    """
    def __init__(self, num_nodes, input_dim=1, hidden_dim=64, K=2, use_bidir=True, horizon=4, dropout=0.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        self.encoder_cell = DCRNNCell(input_dim, hidden_dim, K, use_bidir, dropout=dropout)
        self.decoder_cell = DCRNNCell(input_dim, hidden_dim, K, use_bidir, dropout=dropout)

        self.proj = nn.Linear(hidden_dim, 1)

        self.loss_train = []
        self.loss_val = []

    def forward(self, x_seq, y_seq=None, teacher_forcing_ratio=0.0, P_fwd=None, P_bwd=None):
        """
        x_seq: (B, K_IN, N)   (valeurs scalaires par nœud)
        y_seq: (B, H, N)      (optionnel, pour teacher forcing)
        returns: (B, H, N)
        """
        B, Tin, N = x_seq.shape
        assert N == self.num_nodes

        # encoder
        h = torch.zeros(B, N, self.hidden_dim, device=x_seq.device)
        for t in range(Tin):
            x_t = x_seq[:, t, :].unsqueeze(-1)  # (B,N,1)
            h = self.encoder_cell(x_t, h, P_fwd, P_bwd)

        outputs = []
        dec_in = x_seq[:, -1, :].unsqueeze(-1)  # (B,N,1)

        for t in range(self.horizon):
            h = self.decoder_cell(dec_in, h, P_fwd, P_bwd)
            y_hat = self.proj(h).squeeze(-1)  # (B,N)
            outputs.append(y_hat)

            if (y_seq is not None) and (torch.rand(1).item() < teacher_forcing_ratio):
                dec_in = y_seq[:, t, :].unsqueeze(-1)
            else:
                dec_in = y_hat.unsqueeze(-1)

        return torch.stack(outputs, dim=1)  
    


model = DCRNN(
    num_nodes=N,
    input_dim=1,
    hidden_dim=HIDDEN,
    K=DIFFUSION_STEPS,
    use_bidir=USE_BIDIR,
    horizon=H,
    dropout=DROPOUT,
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total = 0
    mse_norm_sum = 0.0
    mae_orig_sum = 0.0
    mse_orig_sum = 0.0

    mae_h = torch.zeros(H, device=device)
    mse_h = torch.zeros(H, device=device)

    for xb, yb in loader:
        xb = xb.to(device)  
        yb = yb.to(device)  

        pred = model(xb, y_seq=None, teacher_forcing_ratio=0.0, P_fwd=P_fwd, P_bwd=P_bwd)  
        mse_norm = loss_fn(pred, yb)

        y_true_orig = yb * std_t + mean_t
        y_pred_orig = pred * std_t + mean_t

        mae_orig = F.l1_loss(y_pred_orig, y_true_orig)
        mse_orig = F.mse_loss(y_pred_orig, y_true_orig)

        B = xb.shape[0]
        total += B
        mse_norm_sum += mse_norm.item() * B
        mae_orig_sum += mae_orig.item() * B
        mse_orig_sum += mse_orig.item() * B

        for h in range(H):
            mae_h[h] += F.l1_loss(y_pred_orig[:, h, :], y_true_orig[:, h, :]).item() * B
            mse_h[h] += F.mse_loss(y_pred_orig[:, h, :], y_true_orig[:, h, :]).item() * B

    mse_norm_avg = mse_norm_sum / total
    mae_orig_avg = mae_orig_sum / total
    mse_orig_avg = mse_orig_sum / total
    mae_h = (mae_h / total).detach().cpu().numpy()
    mse_h = (mse_h / total).detach().cpu().numpy()

    return mse_norm_avg, mae_orig_avg, mse_orig_avg, mae_h, mse_h

def train_one_epoch():
    model.train()
    total = 0
    loss_sum = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        opt.zero_grad()
        pred = model(xb, y_seq=yb, teacher_forcing_ratio=TEACHER_FORCING, P_fwd=P_fwd, P_bwd=P_bwd)
        loss = loss_fn(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        B = xb.shape[0]
        total += B
        loss_sum += loss.item() * B

    return loss_sum / total

for epoch in range(1, EPOCHS + 1):

    train_mse_norm = train_one_epoch()
    val_mse_norm, val_mae_orig, _, val_mae_h, _ = evaluate(val_loader)

    model.loss_train.append( train_mse_norm)
    model.loss_val.append( val_mse_norm )

    print(
        f"Epoch {epoch:03d} | "
        f"train MSE(norm)={train_mse_norm:.4f} | "
        f"val MSE(norm)={val_mse_norm:.4f} | "
        f"val MAE(orig)={val_mae_orig:.2f} | "
        f"val MAE h=[{', '.join(f'{x:.2f}' for x in val_mae_h)}]"
    )

test_mse_norm, test_mae_orig, test_mse_orig, test_mae_h, test_mse_h = evaluate(test_loader)
print("\n=== TEST ===")
print(f"MSE(norm): {test_mse_norm:.4f}")
print(f"MAE(orig): {test_mae_orig:.2f}")
print(f"MSE(orig): {test_mse_orig:.2f}")
print(f"MAE par horizon: {test_mae_h}")
print(f"MSE par horizon: {test_mse_h}")
