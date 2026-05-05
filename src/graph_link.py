# src/graph_link.py
import pandas as pd
import torch

def build_link_index(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Crée un ID par lien dirigé (From Station -> To Station) + Line + Dir
    """
    links = (
        df_long[["Line", "Dir", "From Station", "To Station"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    links["link_id"] = range(len(links))
    return links

def attach_link_ids(df_long: pd.DataFrame, links_df: pd.DataFrame) -> pd.DataFrame:
    return df_long.merge(
        links_df,
        on=["Line", "Dir", "From Station", "To Station"],
        how="left",
    )

def build_line_graph_edge_index(links_df: pd.DataFrame) -> torch.Tensor:
    """
    Line graph : chaque lien = un noeud.
    On connecte e1 -> e2 si e1.To == e2.From (continuité le long de la ligne)
    """
    df = links_df[["link_id", "From Station", "To Station"]].dropna().copy()

    left = df.rename(columns={"link_id": "src", "To Station": "mid"})[["src", "mid"]]
    right = df.rename(columns={"link_id": "dst", "From Station": "mid"})[["dst", "mid"]]

    pairs = left.merge(right, on="mid", how="inner")[["src", "dst"]].drop_duplicates()

    edge_index = torch.tensor(pairs.values.T, dtype=torch.long)  
    return edge_index
