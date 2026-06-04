import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
METRICS = ROOT / "outputs" / "metrics"
FIGURES = ROOT / "outputs" / "figures"
REPORT = ROOT / "reports" / "network_operations_strategy_memo.md"

st.set_page_config(page_title="Delhivery Network Intelligence", layout="wide", initial_sidebar_state="collapsed")

# --- Premium CSS Injection ---
st.markdown("""
<style>
    /* Global Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4FACFE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Hide Streamlit Cruft */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 30, 36, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 107, 107, 0.3);
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        font-weight: 600;
    }
    
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 4px;
    }
    
    .metric-delta {
        font-size: 1rem;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 8px;
        display: inline-block;
    }
    
    .delta-positive { background: rgba(72, 187, 120, 0.2); color: #48BB78; }
    .delta-negative { background: rgba(245, 101, 101, 0.2); color: #F56565; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Delhivery Graph-Based Intelligence</h1>", unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <p style="font-size: 1.1rem; color: #E2E8F0; margin:0;">
        Welcome to the <b>Network Intelligence Dashboard</b>. This platform models Delhivery's logistics network as a directed graph to surface critical bottlenecks, track chronically delayed corridors, and generate highly accurate ETA predictions using Graph Neural Networks (Node2Vec).
    </p>
</div>
""", unsafe_allow_html=True)

metrics_file = METRICS / "model_comparison.json"
if metrics_file.exists():
    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)
    
    st.markdown("<h3>1. ML Model Performance</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#A0AEC0; margin-bottom: 20px;'>By computing Node2Vec graph embeddings, the Enhanced Model significantly outperforms standard baselines.</p>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-title">Baseline MAE</div>
            <div class="metric-value">{m['baseline_mae']:.2f}</div>
            <div class="metric-delta delta-negative" style="background: rgba(160, 174, 192, 0.2); color: #A0AEC0;">Standard Regression</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        mae_diff = m['baseline_mae'] - m['graph_mae']
        st.markdown(f"""
        <div class="glass-card" style="border: 1px solid rgba(79, 172, 254, 0.3);">
            <div class="metric-title">Graph-Enhanced MAE</div>
            <div class="metric-value">{m['graph_mae']:.2f}</div>
            <div class="metric-delta delta-positive">↓ {mae_diff:.2f} mins</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        acc_diff = (m['graph_acc15'] - m['baseline_acc15']) * 100
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-title">SLA Accuracy (±15%)</div>
            <div class="metric-value">{m['graph_acc15']:.1%}</div>
            <div class="metric-delta delta-positive">↑ {acc_diff:.1f}% uplift</div>
        </div>
        """, unsafe_allow_html=True)
        
    if "embedding_stability" in m:
        stability = m["embedding_stability"]
        if stability < 0.85:
            st.warning(f"**Network Topology Alert:** Node2Vec embedding stability is {stability:.3f}. The network graph topology is experiencing some structural variance.")
        else:
            st.success(f"**Network Topology Stable:** Node2Vec embedding stability is {stability:.3f}.")
else:
    st.info("Run `python scripts/run_full_pipeline.py` to generate outputs.")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h3>2. Network Topology & Bottlenecks</h3>", unsafe_allow_html=True)

col_top, col_bot = st.columns([1.2, 1])

with col_top:
    st.markdown("""
    <div class="glass-card" style="padding: 12px; height: 100%;">
    """, unsafe_allow_html=True)
    topology_file = FIGURES / "network_topology.png"
    if topology_file.exists():
        st.image(str(topology_file), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_bot:
    st.markdown("<p style='color:#A0AEC0; font-weight: 600;'>Top 5 Structural Bottlenecks</p>", unsafe_allow_html=True)
    hubs_file = TABLES / "top_5_bottleneck_hubs.csv"
    if hubs_file.exists():
        st.dataframe(pd.read_csv(hubs_file), use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

c_corr, c_policy = st.columns(2)

with c_corr:
    st.markdown("<h3>3. Chronic Delay Corridors</h3>", unsafe_allow_html=True)
    corridors_file = TABLES / "top_delay_corridors.csv"
    if corridors_file.exists():
        st.dataframe(pd.read_csv(corridors_file).head(10), use_container_width=True, hide_index=True)

with c_policy:
    st.markdown("<h3>4. Route Strategy (FTL vs Carting)</h3>", unsafe_allow_html=True)
    policy_file = TABLES / "ftl_vs_carting_policy.csv"
    if policy_file.exists():
        st.dataframe(pd.read_csv(policy_file).head(10), use_container_width=True, hide_index=True)

if REPORT.exists():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>5. Executive Strategy Memo</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="background: rgba(20, 20, 25, 0.8);">
    """, unsafe_allow_html=True)
    st.markdown(REPORT.read_text(encoding="utf-8"))
    st.markdown("</div>", unsafe_allow_html=True)
