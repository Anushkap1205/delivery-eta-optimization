from __future__ import annotations

import pandas as pd


def _normalize_input_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Map common alternative column names to project-standard names.
    aliases = {
        "source_center": "source_facility",
        "source_name": "source_facility",
        "destination_center": "dest_facility",
        "destination_name": "dest_facility",
        "od_start_time": "departure_ts",
        "od_end_time": "arrival_ts",
        "osrm_time": "osrm_eta_minutes",
        "actual_time": "actual_transit_minutes",
        "actual_distance_to_destination": "distance_km",
    }
    out = df.copy()
    for src, dst in aliases.items():
        if dst not in out.columns and src in out.columns:
            out[dst] = out[src]

    # For datasets with progressive cutoff snapshots, keep final snapshot per segment.
    if "is_cutoff" in out.columns:
        cutoff = out["is_cutoff"].astype(str).str.lower()
        final_rows = out[cutoff.isin(["false", "0"])].copy()
        if final_rows.empty:
            raise ValueError("All rows are cutoff snapshots — no final trip records found. Check is_cutoff column values.")
        out = final_rows

    return out


def _derive_actual_minutes(df: pd.DataFrame) -> pd.DataFrame:
    if "actual_transit_minutes" in df.columns:
        return df
    if {"departure_ts", "arrival_ts"}.issubset(df.columns):
        dep = pd.to_datetime(df["departure_ts"], errors="coerce")
        arr = pd.to_datetime(df["arrival_ts"], errors="coerce")
        df["actual_transit_minutes"] = (arr - dep).dt.total_seconds() / 60.0
        return df
    raise ValueError(
        "Need either `actual_transit_minutes` or both `departure_ts` and `arrival_ts`."
    )


def load_and_prepare(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = _normalize_input_schema(df)
    required = {"source_facility", "dest_facility", "route_type", "osrm_eta_minutes"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = _derive_actual_minutes(df)
    df = df.dropna(
        subset=["source_facility", "dest_facility", "route_type", "osrm_eta_minutes", "actual_transit_minutes"]
    ).copy()
    df["osrm_eta_minutes"] = pd.to_numeric(df["osrm_eta_minutes"], errors="coerce")
    df["actual_transit_minutes"] = pd.to_numeric(df["actual_transit_minutes"], errors="coerce")
    df = df[(df["osrm_eta_minutes"] > 0) & (df["actual_transit_minutes"] > 0)].copy()

    if "departure_ts" in df.columns:
        dep = pd.to_datetime(df["departure_ts"], errors="coerce")
        bad_ts_count = dep.isna().sum()
        if bad_ts_count > 0:
            print(f"[pipeline] WARNING: {bad_ts_count} rows have unparseable departure_ts and will be dropped.")
        df = df[dep.notna()].copy()
        dep = dep[dep.notna()]
        hour = dep.dt.hour
        df["hour_of_day"] = hour
        df["day_of_week"] = dep.dt.dayofweek.astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6)).astype(int)
        df["month"] = dep.dt.month.astype(int)
    else:
        hour = pd.Series(12, index=df.index)
        df["hour_of_day"] = 12
        df["day_of_week"] = 0
        df["is_weekend"] = 0
        df["is_night"] = 0
        df["month"] = 1

    df["time_bucket"] = pd.cut(
        hour,
        bins=[-1, 6, 11, 17, 21, 24],
        labels=["night", "morning", "afternoon", "evening", "night_late"],
    ).astype(str)
    df["delay_ratio"] = df["actual_transit_minutes"] / df["osrm_eta_minutes"]
    # Cap at 5x: trips exceeding 5× OSRM ETA are likely data entry errors
    df["delay_ratio"] = df["delay_ratio"].clip(upper=5.0)
    df["delay_pct"] = (df["actual_transit_minutes"] - df["osrm_eta_minutes"]) / df["osrm_eta_minutes"]
    df["delay_pct"] = df["delay_pct"].clip(upper=4.0)  # equivalent upper bound for delay_pct
    df["corridor"] = df["source_facility"].astype(str) + "->" + df["dest_facility"].astype(str)

    if "promised_eta_minutes" in df.columns:
        df["sla_breach"] = (df["actual_transit_minutes"] > df["promised_eta_minutes"]).astype(int)
    else:
        # Fallback business proxy when promised ETA is unavailable.
        df["sla_breach"] = (df["delay_pct"] > 0.20).astype(int)

    return df

