import pandas as pd

ID_COLS = [
    "Link", "Line", "Dir", "Order",
    "From NLC", "From ASC", "From Station",
    "To NLC", "To ASC", "To Station",
    "Total", "Early", "AM Peak", "Midday", "PM Peak", "Evening", "Late"
]

def preprocess(df, source=""):
    existing_id = [c for c in ID_COLS if c in df.columns]
    flux_cols = [c for c in df.columns if c not in existing_id]

    df_long = df.melt(
        id_vars=existing_id,
        value_vars=flux_cols,
        var_name="Time",
        value_name="Load"
    )

    df_long["Source_File"] = source
    return df_long
