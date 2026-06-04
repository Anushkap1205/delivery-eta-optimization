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
    agg["chronic_delay"] = agg["avg_delay_pct"] > 0.20
    return agg


def build_network_graph(edge_df: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for _, row in edge_df.iterrows():
        src = row["source_facility"]
        dst = row["dest_facility"]
        g.add_edge(
            src,
            dst,
            trips=int(row["trips"]),
            median_delay_ratio=float(row["median_delay_ratio"]),
            avg_delay_pct=float(row["avg_delay_pct"]),
            breach_rate=float(row["breach_rate"]),
            route_type=row["route_type"],
            time_bucket=row["time_bucket"],
        )
    return g


def compute_hub_metrics(g: nx.DiGraph) -> pd.DataFrame:
    if g.number_of_nodes() == 0:
        return pd.DataFrame(columns=["hub", "betweenness", "in_degree", "out_degree", "clustering"])

    betweenness = nx.betweenness_centrality(g, normalized=True)
    clustering = nx.clustering(g.to_undirected())
    rows = []
    for n in g.nodes():
        rows.append(
            {
                "hub": n,
                "betweenness": betweenness.get(n, 0.0),
                "in_degree": g.in_degree(n),
                "out_degree": g.out_degree(n),
                "clustering": clustering.get(n, 0.0),
            }
        )
    return pd.DataFrame(rows)

