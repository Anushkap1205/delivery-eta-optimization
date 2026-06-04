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
        if not final_rows.empty:
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
    df = df[(df["osrm_eta_minutes"] > 0) & (df["actual_transit_minutes"] > 0)].copy()

    if "departure_ts" in df.columns:
        dep = pd.to_datetime(df["departure_ts"], errors="coerce")
        hour = dep.dt.hour.fillna(12)
        df["hour_of_day"] = hour
        df["day_of_week"] = dep.dt.dayofweek.fillna(0).astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6)).astype(int)
        df["month"] = dep.dt.month.fillna(1).astype(int)
    else:
        hour = pd.Series(12, index=df.index)
        df["hour_of_day"] = 12
        df["day_of_week"] = 0
        df["is_weekend"] = 0
        df["is_night"] = 0
        df["month"] = 1

    df["time_bucket"] = pd.cut(
        hour,
        bins=[-1, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"],
    ).astype(str)
    df["delay_ratio"] = df["actual_transit_minutes"] / df["osrm_eta_minutes"]
    df["delay_pct"] = (df["actual_transit_minutes"] - df["osrm_eta_minutes"]) / df["osrm_eta_minutes"]
    df["corridor"] = df["source_facility"].astype(str) + "->" + df["dest_facility"].astype(str)

    if "promised_eta_minutes" in df.columns:
        df["sla_breach"] = (df["actual_transit_minutes"] > df["promised_eta_minutes"]).astype(int)
    else:
        # Fallback business proxy when promised ETA is unavailable.
        df["sla_breach"] = (df["delay_pct"] > 0.20).astype(int)

    return df

