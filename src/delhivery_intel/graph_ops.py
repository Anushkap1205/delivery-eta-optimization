from __future__ import annotations

import networkx as nx
import pandas as pd


def build_corridor_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    grp_cols = ["source_facility", "dest_facility", "route_type", "time_bucket"]
    agg = (
        df.groupby(grp_cols, dropna=False)
        .agg(
            trips=("corridor", "count"),
            median_delay_ratio=("delay_ratio", "median"),
            avg_delay_pct=("delay_pct", "mean"),
            breach_rate=("sla_breach", "mean"),
        )
        .reset_index()
    )
    agg["sla_breach_count"] = (agg["breach_rate"] * agg["trips"]).round().astype(int)
    total_breaches = agg["sla_breach_count"].sum()
    agg["sla_breach_share_pct"] = (
        (agg["sla_breach_count"] / total_breaches * 100).round(2)
        if total_breaches > 0 else 0.0
    )
    agg["chronic_delay"] = agg["avg_delay_pct"] > 0.20
    return agg


def build_network_graph(edge_df: pd.DataFrame) -> nx.DiGraph:
    required = {"source_facility", "dest_facility", "trips", "median_delay_ratio", "avg_delay_pct", "breach_rate"}
    missing = required - set(edge_df.columns)
    if missing:
        raise ValueError(f"build_network_graph expects pre-aggregated edge_df. Missing columns: {sorted(missing)}. Run build_corridor_aggregates() first.")
    # Pre-aggregation is required to prevent DiGraph edge overwriting, keeping original edge_df available separately for FTL/Carting logic
    graph_edges = (
        edge_df.groupby(["source_facility", "dest_facility"])
        .agg(
            trips=("trips", "sum"),
            median_delay_ratio=("median_delay_ratio", "mean"),
            avg_delay_pct=("avg_delay_pct", "mean"),
            breach_rate=("breach_rate", "mean"),
        )
        .reset_index()
    )
    g = nx.DiGraph()
    for _, row in graph_edges.iterrows():
        src = row["source_facility"]
        dst = row["dest_facility"]
        trips_val = int(row["trips"])
        g.add_edge(
            src,
            dst,
            trips=trips_val,
            inv_trips=1.0 / trips_val if trips_val > 0 else 1.0,
            median_delay_ratio=float(row["median_delay_ratio"]),
            avg_delay_pct=float(row["avg_delay_pct"]),
            breach_rate=float(row["breach_rate"]),
        )
    return g


def compute_hub_metrics(g: nx.DiGraph) -> pd.DataFrame:
    if g.number_of_nodes() == 0:
        return pd.DataFrame(columns=["hub", "betweenness", "in_degree", "out_degree", "clustering"])

    k_sample = g.number_of_nodes() if g.number_of_nodes() <= 50 else min(100, g.number_of_nodes())
    betweenness = nx.betweenness_centrality(g, normalized=True, weight="inv_trips", k=None if g.number_of_nodes() <= 50 else k_sample)
    # weight='inv_trips' correctly biases shortest paths toward high-volume corridors (lower inverse weight). k=min(100,N) approximates betweenness for large graphs.
    clustering = nx.clustering(g.to_undirected())
    rows = []
    for n in g.nodes():
        rows.append(
            {
                "hub": n,
                "betweenness": betweenness.get(n, 0.0),
                "in_degree": g.in_degree(n),
                "out_degree": g.out_degree(n),
                "in_degree_weighted": g.in_degree(n, weight="trips"),
                "out_degree_weighted": g.out_degree(n, weight="trips"),
                "clustering": clustering.get(n, 0.0),
            }
        )
    df_metrics = pd.DataFrame(rows)
    if not df_metrics.empty and df_metrics["betweenness"].max() > 0:
        idw_max = df_metrics["in_degree_weighted"].max()
        df_metrics["risk_score"] = (
            0.5 * df_metrics["betweenness"] / df_metrics["betweenness"].max()
            + (0.3 * df_metrics["in_degree_weighted"] / idw_max if idw_max > 0 else 0.0)
            + 0.2 * df_metrics["clustering"]
        )
    else:
        df_metrics["risk_score"] = 0.0
    # betweenness (0.5) captures systemic network dependency; in_degree_weighted (0.3) captures operational inflow load; clustering (0.2) captures redundancy risk.
    return df_metrics


def get_top_delay_corridors(corridor_agg: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Returns the top-N chronically delayed corridors where actual transit
    exceeds OSRM ETA by more than 20%, ranked by SLA breach share.
    Deliverable 2: corridor audit with breach contribution ranking.
    """
    chronic = corridor_agg[corridor_agg["chronic_delay"]].copy()
    chronic = chronic.sort_values("sla_breach_share_pct", ascending=False).head(top_n)
    return chronic[["source_facility", "dest_facility", "route_type", "time_bucket",
                     "trips", "avg_delay_pct", "breach_rate", "sla_breach_count", "sla_breach_share_pct"]]
