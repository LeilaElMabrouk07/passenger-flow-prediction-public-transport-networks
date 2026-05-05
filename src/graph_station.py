import pandas as pd
import torch

def build_station_index(df_long: pd.DataFrame):
    stations = pd.Series(
        pd.concat([df_long["From Station"], df_long["To Station"]], ignore_index=True)
    ).dropna().drop_duplicates().reset_index(drop=True)

    stations_df = pd.DataFrame({"station": stations})
    stations_df["station_id"] = range(len(stations_df))
    return stations_df

def attach_station_ids(df_long: pd.DataFrame, stations_df: pd.DataFrame):
    df = df_long.merge(stations_df.rename(columns={"station": "From Station", "station_id": "from_id"}),
                       on="From Station", how="left")
    df = df.merge(stations_df.rename(columns={"station": "To Station", "station_id": "to_id"}),
                  on="To Station", how="left")
    return df

def build_edge_index_station(df_with_ids: pd.DataFrame):
    # edges from station -> to station (unique)
    e = df_with_ids[["from_id", "to_id"]].dropna().drop_duplicates()
    edge_index = torch.tensor(e.values.T, dtype=torch.long)
    return edge_index

def edge_index_to_adj(edge_index, num_nodes):
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        A[s, d] = 1.0

    # self-loops
    A.fill_diagonal_(1.0)

    # symmetrize
    A = torch.maximum(A, A.t())

    # normalize D^-1/2 A D^-1/2
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D = torch.diag(deg_inv_sqrt)
    return D @ A @ D
