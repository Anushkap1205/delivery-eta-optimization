from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def plot_top_hubs(top_hubs: pd.DataFrame, out_path: str) -> None:
    if top_hubs.empty:
        return
    plt.figure(figsize=(8, 4))
    df = top_hubs.sort_values("hub_risk_score", ascending=True)
    plt.barh(df["hub"], df["hub_risk_score"])
    plt.title("Top Bottleneck Hubs by Risk Score")
    plt.xlabel("Hub Risk Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_delay_corridors(corridors: pd.DataFrame, out_path: str) -> None:
    if corridors.empty:
        return
    plt.figure(figsize=(10, 4))
    df = corridors.head(15).copy()
    names = df["source_facility"].astype(str) + "->" + df["dest_facility"].astype(str)
    plt.bar(range(len(df)), df["avg_delay_pct"] * 100)
    plt.xticks(range(len(df)), names, rotation=60, ha="right")
    plt.ylabel("Avg Delay %")
    plt.title("Chronic Delay Corridors")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_network_topology(g: nx.DiGraph, top_hubs: pd.DataFrame, top_corridors: pd.DataFrame, out_path: str) -> None:
    if g.number_of_nodes() == 0:
        return
        
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(g, seed=42, k=0.15)
    
    top_hub_names = set(top_hubs["hub"].tolist()) if not top_hubs.empty else set()
    top_corridor_edges = set(zip(top_corridors["source_facility"], top_corridors["dest_facility"])) if not top_corridors.empty else set()
    
    # Draw background nodes
    background_nodes = [n for n in g.nodes() if n not in top_hub_names]
    nx.draw_networkx_nodes(g, pos, nodelist=background_nodes, node_size=20, node_color="#AAAAAA", alpha=0.6)
    
    # Draw top bottleneck hubs
    if top_hub_names:
        nx.draw_networkx_nodes(g, pos, nodelist=list(top_hub_names), node_size=150, node_color="#D62728", edgecolors="black")
        
    # Draw background edges
    background_edges = [e for e in g.edges() if e not in top_corridor_edges]
    nx.draw_networkx_edges(g, pos, edgelist=background_edges, edge_color="#DDDDDD", alpha=0.3, arrows=False)
    
    # Draw top delay corridors
    if top_corridor_edges:
        nx.draw_networkx_edges(g, pos, edgelist=list(top_corridor_edges), edge_color="#D62728", width=2.0, alpha=0.9, arrows=True, arrowsize=15)
        
    plt.title("Logistics Network Topology\n(Red: Top Bottleneck Hubs & Delay Corridors)", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
