# src/app_link.py
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from build_table import build_or_load_long_table
from graph_link import build_link_index, attach_link_ids, build_line_graph_edge_index
from graph_station import edge_index_to_adj
from link_dataset_stgcn import build_link_tensor_day, load_global_norm_link
from model_stgcn import STGCN
from config import N_HISTORY, HORIZON, TEST_YEAR, CACHE_DIR

MODEL_PATH_LINK = CACHE_DIR / "stgcn_link_model.pt"

def order_links_geographically(links_df_line):
    links = links_df_line[["link_id","From Station","To Station"]].dropna().copy()

    next_map = {r["From Station"]: r["To Station"] for _,r in links.iterrows()}
    prev_map = {r["To Station"]: r["From Station"] for _,r in links.iterrows()}

    start_candidates = [s for s in next_map if s not in prev_map]

    if not start_candidates:
        return links.sort_values("link_id")["link_id"].tolist()

    start = start_candidates[0]

    ordered = []
    visited = set()
    cur = start

    while cur in next_map and cur not in visited:
        visited.add(cur)
        nxt = next_map[cur]

        link_id = links[
            (links["From Station"] == cur) &
            (links["To Station"] == nxt)
        ]["link_id"].iloc[0]

        ordered.append(link_id)
        cur = nxt

    return ordered

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# modèle 
@st.cache
def load_model(num_links, _A_norm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGCN(
        num_nodes=num_links,
        A_norm=_A_norm.to(device),
        in_channels=1,
        hidden_channels=32,
        kt=3,
        dropout=0.1,
        horizon=HORIZON,
    ).to(device)
    ckpt = torch.load(MODEL_PATH_LINK, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, device

# APP
def main():
    st.set_page_config(layout="wide")
    st.title("Prévision du flux passagers DANS LE MÉTRO (interstations)")

    # chargement données
    df = build_or_load_long_table(force_rebuild=False)
    TARGET_LINES = ["Bakerloo", "H&C and Circle"]
    df = df[df["Line"].isin(TARGET_LINES)].copy()
    links = build_link_index(df)
    df_l = attach_link_ids(df, links)

    # menus dynamiques 
    years = sorted(df_l["year"].unique())
    year = st.selectbox("Année", years, index=len(years)-1)
    df_y = df_l[df_l["year"] == year]

    lines = sorted(df_y["Line"].unique())
    line = st.selectbox("Ligne", lines)
    df_line = df_y[df_y["Line"] == line]

    dirs = sorted(df_line["Dir"].dropna().unique())
    direction = st.selectbox("Direction", dirs)
    df_ld = df_line[df_line["Dir"] == direction]

    day_codes = sorted(df_ld["day_code"].dropna().unique())
    day_code = st.selectbox("Jour type", day_codes)

    g = df_ld[df_ld["day_code"] == day_code].copy()
    if g.empty:
        st.error("Aucune donnée disponible pour cette combinaison.")
        return

    if "time_range" not in g.columns:
        g["time_range"] = g["time_index"].apply(lambda x: f"{(x*15)//60:02d}:{(x*15)%60:02d}")
    time_map = g[["time_index", "time_range"]].drop_duplicates().sort_values("time_index").reset_index(drop=True)

    selected_time = st.selectbox("Heure réelle", time_map["time_range"].tolist())
    idx_now = time_map.loc[time_map["time_range"] == selected_time, "time_index"].iloc[0]

    # ordre géographique 
    links_subset = links[(links["Line"] == line) & (links["Dir"] == direction)]
    ordered_link_ids = order_links_geographically(links_subset)

    # tenseur 
    X = build_link_tensor_day(g, len(links))
    if idx_now < N_HISTORY or idx_now + HORIZON >= X.shape[0]:
        st.error("Pas assez d'historique ou futur indisponible pour cette heure.")
        return

    # modèle 
    edge_index = build_line_graph_edge_index(links)
    A_norm = edge_index_to_adj(edge_index, num_nodes=len(links))
    model, device = load_model(len(links), A_norm)
    mean, std = load_global_norm_link()

    Xn = (X - mean) / std
    t0 = idx_now - (N_HISTORY - 1)
    xin = Xn[t0:t0+N_HISTORY, :]
    x_tensor = torch.tensor(xin.T, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        yhat_norm = model(x_tensor).cpu().numpy()[0]  
    yhat = yhat_norm * std + mean

    # affichage 
    segment_labels = [f"{links.loc[i,'From Station']} → {links.loc[i,'To Station']}" for i in ordered_link_ids]
    num_links_line = len(ordered_link_ids)
    st.subheader("Flux par horizon (1 figure par 15min)")
    metrics = {}
    
    fig = plt.figure(figsize=(18,6))
    x = np.arange(num_links_line)

    # Afficher uniquement +15min et +30min
    horizons_to_plot = [0, 1]  
    for h in horizons_to_plot:
        t_idx = idx_now + 1 + h
        g_h = g[g["time_index"] == t_idx]

        # flux réel
        y_true_h = np.zeros(num_links_line, dtype=np.float32)
        for i, link_id in enumerate(ordered_link_ids):
            flow = g_h[g_h["link_id"] == link_id]["flow"].values
            y_true_h[i] = flow[0] if len(flow) > 0 else 0.0

        # flux prédit
        y_pred_h = yhat[h, ordered_link_ids]

        mae, rmse = compute_metrics(y_true_h, y_pred_h)

        # stocker les metrics
        metrics[f"+{15*(h+1)}min"] = {"MAE": mae, "RMSE": rmse}

        # réel
        plt.plot(
            x,
            y_true_h,
            linestyle="--",
            alpha=0.4,
            label=f"Réel +{15*(h+1)}min"
        )

        # prédit
        plt.plot(
            x,
            y_pred_h,
            marker="o",
            label=f"Prédit +{15*(h+1)}min (MAE={mae:.1f})"
        )

    step = max(1, len(segment_labels)//25)

    plt.xticks(
        x[::step],
        [segment_labels[i] for i in x[::step]],
        rotation=45,
        ha="right"
    )

    plt.ylabel("Flux passagers")
    plt.xlabel("Segments interstations")

    plt.title(
        f"Prévision multi-horizon du flux passagers\n"
        f"Ligne {line} | Direction {direction} | {day_code} {year} | heure {selected_time}"
    )

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("Erreurs par horizon")
    st.write(metrics)

if __name__ == "__main__":
    main()



