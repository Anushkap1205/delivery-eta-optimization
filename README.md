# Delhivery Graph-Based ETA Intelligence

End-to-end machine learning + consulting project to improve delivery ETAs by modeling a logistics network as a graph of hubs and corridors, instead of isolated point-to-point routes.

## Problem Statement

OSRM-style ETA estimation often underestimates real transit time because it misses network effects like:

- hub dwell and processing delays
- congestion spillover across dependent corridors
- route-type dynamics (FTL vs Carting)
- time-of-day structural risk

This project builds a graph-aware ETA system that both predicts better and explains where network operations should intervene.

## Outcomes (Latest Run)

Using real trip data (`delivery_data.csv`) and the final benchmark setup:

- Baseline MAE: `44.43`
- Graph-Enhanced MAE: `33.02`
- MAE improvement: `11.41`
- Baseline Accuracy @ +/-15%: `43.03%`
- Graph Accuracy @ +/-15%: `55.34%`
- Accuracy gain: `+12.31 percentage points`

## What This Project Delivers

- Directed weighted graph construction from shipment legs
- Bottleneck hub detection using network centrality metrics
- Chronic delay corridor audit and SLA breach contribution ranking
- Baseline vs graph-enhanced ETA benchmark (measured graph advantage)
- FTL vs Carting decision framework with expected time-cost tradeoff
- Auto-generated Network Operations strategy memo
- Streamlit dashboard for stakeholder-facing review

## Architecture

1. **Ingestion and schema normalization**
   - Accepts standard trip schema and maps alternative schemas (including Delhivery-style columns)
2. **Feature engineering**
   - Creates `delay_ratio`, `delay_pct`, `time_bucket`, `sla_breach`, corridor identifiers
3. **Graph intelligence layer**
   - Nodes: facilities, Edges: directed corridors with delay + breach + volume attributes
4. **Modeling layer**
   - Baseline (trip-level features)
   - Graph-enhanced model (node2vec source/destination embeddings + trip features)
5. **Decision and reporting layer**
   - Route-type recommendation (FTL vs Carting)
   - Top bottleneck hubs, top delay corridors, impact memo

## Repository Structure

- `src/delhivery_intel/pipeline.py` data loading and feature prep
- `src/delhivery_intel/graph_ops.py` graph building and network metrics
- `src/delhivery_intel/modeling.py` baseline and graph-enhanced ETA models
- `src/delhivery_intel/decision_framework.py` FTL vs Carting recommendation logic
- `src/delhivery_intel/reporting.py` memo generation and impact estimation
- `src/delhivery_intel/visualization.py` charts for hubs and delay corridors
- `scripts/run_full_pipeline.py` one-command end-to-end execution
- `app/streamlit_app.py` optional dashboard
- `reports/network_operations_strategy_memo.md` generated strategy deliverable

## Supported Input Columns

The pipeline works with either canonical names or mapped alternatives.

Canonical expected columns:

- `source_facility`, `dest_facility`, `route_type`
- `osrm_eta_minutes`, `actual_transit_minutes`
- `departure_ts`/`arrival_ts` (if actual minutes not directly provided)
- optional: `distance_km`, `promised_eta_minutes`

Also supports Delhivery-style columns via auto-mapping, e.g.:

- `source_name` or `source_center` -> `source_facility`
- `destination_name` or `destination_center` -> `dest_facility`
- `actual_time` -> `actual_transit_minutes`
- `osrm_time` -> `osrm_eta_minutes`
- `actual_distance_to_destination` -> `distance_km`
- `od_start_time` / `od_end_time` -> timestamps

## Quick Start

```bash
cd /Users/anushkapatil/delhivery-network-intelligence
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run full pipeline on your own data:

```bash
python scripts/run_full_pipeline.py --input "/path/to/your/data.csv"
```

Run on default file location:

```bash
python scripts/run_full_pipeline.py
```

Launch dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Generated Outputs

- `outputs/metrics/model_comparison.json`
- `outputs/tables/top_5_bottleneck_hubs.csv`
- `outputs/tables/top_delay_corridors.csv`
- `outputs/tables/ftl_vs_carting_policy.csv`
- `outputs/figures/top_bottleneck_hubs.png`
- `outputs/figures/top_delay_corridors.png`
- `reports/network_operations_strategy_memo.md`

## Business Metrics Tracked

- MAE (ETA prediction error)
- Accuracy @ +/-15% of actual ETA
- Corridor-level SLA breach contribution
- Hub-level structural risk score
- Scenario impact from top-3 bottleneck hub upgrades

