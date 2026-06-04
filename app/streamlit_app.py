from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
METRICS = ROOT / "outputs" / "metrics"
FIGURES = ROOT / "outputs" / "figures"
REPORT = ROOT / "reports" / "network_operations_strategy_memo.md"

st.set_page_config(page_title="Delhivery Network Intelligence", layout="wide")
st.title("Delhivery Graph-Based ETA Intelligence")

# --- Dashboard Guide ---
st.markdown("""
Welcome to the **Network Intelligence Dashboard**. This tool models Delhivery's logistics network as a directed graph to surface critical chokepoints, track chronically delayed corridors, and produce smarter ETA predictions using graph neural networks.

**How to use this dashboard:**
- **Metrics Overview:** See how integrating graph-based features improves our ETA predictions over traditional baseline models.
- **Network Topology:** Visually identify the most critical structural risks in our network layout.
- **Data Tables:** Dive into the specific hubs, corridors, and ML-backed route recommendations.
- **Strategy Memo:** Read a synthesized, actionable summary tailored for network operations leaders.
---
""")

metrics_file = METRICS / "model_comparison.json"
if metrics_file.exists():
    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)
    
    st.subheader("1. ETA Model Performance")
    st.markdown("By representing facilities as nodes and computing their graph embeddings (Node2Vec), the **Graph-enhanced model** produces much smarter ETA predictions. Below is the performance uplift compared to a standard baseline regression.")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline MAE", f'{m["baseline_mae"]:.2f}', help="Mean Absolute Error of the baseline model (lower is better)")
    c2.metric("Graph MAE", f'{m["graph_mae"]:.2f}', delta=f'{m["baseline_mae"] - m["graph_mae"]:.2f} mins', delta_color="inverse")
    c3.metric("Graph Acc@15%", f'{m["graph_acc15"]:.2%}', delta=f'{m["graph_acc15"] - m["baseline_acc15"]:.2%}', help="Percentage of predicted ETAs within 15% of actual delivery time")

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_rmse1, col_rmse2 = st.columns(2)
    col_rmse1.metric("Baseline RMSE", f'{m.get("baseline_rmse", 0):.2f}', help="Root Mean Squared Error of the baseline model (lower is better)")
    col_rmse2.metric("Graph RMSE", f'{m.get("graph_rmse", 0):.2f}', delta=f'{m.get("baseline_rmse", 0) - m.get("graph_rmse", 0):.2f} mins', delta_color="inverse")
    
    if "embedding_stability" in m:
        stability = m["embedding_stability"]
        if stability < 0.85:
            st.warning(f"**Warning:** Node2Vec embedding stability is low ({stability:.2f}). Model predictions may not be fully reliable.")
        else:
            st.info(f"Node2Vec embedding stability is good: **{stability:.2f}**")
else:
    st.info("Run `python scripts/run_full_pipeline.py` to generate outputs.")

st.divider()

st.subheader("2. Network Topology")
st.markdown("A structural map of the logistics network. **Large red nodes** indicate top bottleneck hubs. **Thick red arrows** indicate chronically delayed corridors.")
topology_file = FIGURES / "network_topology.png"
if topology_file.exists():
    st.image(str(topology_file), use_container_width=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("3. Top Bottleneck Hubs")
    st.markdown("Facilities with the highest structural risk. Delays at these highly-central hubs cascade across the network and cause massive SLA breaches downstream.")
    hubs_file = TABLES / "top_5_bottleneck_hubs.csv"
    if hubs_file.exists():
        st.dataframe(pd.read_csv(hubs_file), width="stretch")

with col2:
    st.subheader("4. Top Delay Corridors")
    st.markdown("The most chronically delayed corridors, ranked by their contribution to SLA breaches. These are prime targets for parallel route activations.")
    corridors_file = TABLES / "top_delay_corridors.csv"
    if corridors_file.exists():
        st.dataframe(pd.read_csv(corridors_file).head(20), width="stretch")

st.divider()

st.subheader("5. FTL vs Carting Recommendations")
st.markdown("An **ML-backed framework** that recommends the optimal route type (Full Truck Load vs. Carting). It evaluates historical SLA breach probabilities alongside transit costs for each specific corridor profile, minimizing total expected cost.")
policy_file = TABLES / "ftl_vs_carting_policy.csv"
if policy_file.exists():
    st.dataframe(pd.read_csv(policy_file).head(50), width="stretch")

st.divider()

if REPORT.exists():
    st.subheader("6. Strategy Memo")
    st.markdown("A synthesized strategy memo for operations leaders, outlining exactly what interventions to make and estimating the recovered revenue impact.")
    with st.container(border=True):
        st.markdown(REPORT.read_text(encoding="utf-8"))

