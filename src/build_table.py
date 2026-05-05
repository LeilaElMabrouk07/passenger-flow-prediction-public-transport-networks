import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import ROOT, YEAR_DIR_GLOB, SHEET_CANDIDATES, SKIPROWS, PROCESSED_PARQUET
from parsing import parse_year_and_daycode

ID_COLS = [
    "Link", "Line", "Dir", "Order",
    "From NLC", "From ASC", "From Station",
    "To NLC", "To ASC", "To Station",
    "Total", "Early", "AM Peak", "Midday", "PM Peak", "Evening", "Late"
]

def _read_link_loads_one_file(fp: Path):
    df = None
    last_err = None

    for sheet in SHEET_CANDIDATES:
        try:
            raw = pd.read_excel(fp, sheet_name=sheet, header=None)
            raw = raw.iloc[SKIPROWS:].reset_index(drop=True)
            raw.columns = raw.iloc[0]
            df = raw.iloc[1:].reset_index(drop=True)
            break
        except Exception as e:
            last_err = e

    if df is None:
        print(f" Impossible de lire {fp.name} (Link Loads). Dernière erreur: {last_err}")
        return None

    df.columns = df.columns.astype(str).str.strip()

    needed = ["Link", "Line", "Dir", "From Station", "To Station"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f" Colonnes manquantes dans {fp.name}: {missing}")
        return None

    flow_cols = [c for c in df.columns if isinstance(c, str) and len(c) == 9 and c[4] == "-" and c[:4].isdigit() and c[5:].isdigit()]
    if not flow_cols:
        flow_cols = [c for c in df.columns if isinstance(c, str) and "-" in c and c[:2].isdigit()]

    id_cols = [c for c in ID_COLS if c in df.columns]

    long_df = df.melt(
        id_vars=id_cols,
        value_vars=flow_cols,
        var_name="time_range",
        value_name="flow"
    )

    long_df["flow"] = (
        long_df["flow"]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .replace({"nan": None, "": None})
    )
    long_df["flow"] = pd.to_numeric(long_df["flow"], errors="coerce")

    return long_df

def build_or_load_long_table(force_rebuild=False):
    if PROCESSED_PARQUET.exists() and not force_rebuild:
        print(f" Chargement cache: {PROCESSED_PARQUET}")
        return pd.read_parquet(
    PROCESSED_PARQUET,
    columns=[
        "year", "day_code", "Line", "Dir",
        "time_index", "flow",
        "From Station", "To Station"
    ]
)

    print(" Construction table long format")

    xlsx_files = []
    for year_dir in sorted(ROOT.glob(YEAR_DIR_GLOB)):
        if year_dir.is_dir():
            xlsx_files.extend(sorted(year_dir.glob("*.xlsx")))

    rows = []
    for fp in tqdm(xlsx_files, desc="Reading XLSX"):
        year, day_code = parse_year_and_daycode(str(fp))
        if year is None:
            continue

        df = _read_link_loads_one_file(fp)
        if df is None or df.empty:
            continue

        df["year"] = year
        df["day_code"] = day_code
        df["source_file"] = fp.name
        rows.append(df)

    if not rows:
        raise ValueError("Aucun fichier exploitable n'a été lu.")

    out = pd.concat(rows, ignore_index=True)

    out["t_start"] = out["time_range"].str.slice(0, 4)
    out["t_min"] = out["t_start"].str.slice(0, 2).astype(int) * 60 + out["t_start"].str.slice(2, 4).astype(int)

    out.loc[out["t_min"] < 300, "t_min"] += 24 * 60

    out["time_index"] = out["t_min"].rank(method="dense").astype(int) - 1

    out.to_parquet(PROCESSED_PARQUET, index=False)
    print(f" Sauvegardé: {PROCESSED_PARQUET}")
    return out
