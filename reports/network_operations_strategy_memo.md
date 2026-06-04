# Network Operations Strategy Memo

## Executive takeaway
- Graph-enhanced ETA model outperforms baseline by measured business metrics.
- A small set of structurally central hubs contributes disproportionate SLA risk.
- Focused upgrades on top hubs are expected to reduce late deliveries and recover revenue-at-risk.

## Model performance
- Baseline MAE: 244.36
- Graph-enhanced MAE: 201.15
- Baseline % within 15%: 53.19%
- Graph-enhanced % within 15%: 61.81%
- Graph advantage (MAE reduction): 43.21
- Graph advantage (+/-15% accuracy): 8.63%

## Top bottleneck hubs
- AMD: risk_score=0.00
- BLR: risk_score=0.00
- BOM: risk_score=0.00
- CCU: risk_score=0.00
- DEL: risk_score=0.00

## Chronic delay corridors
- BOM -> AMD (Carting, night): avg_delay=40.2%, breach_rate=100.0%
- BOM -> DEL (Carting, morning): avg_delay=76.2%, breach_rate=100.0%
- BOM -> JAI (Carting, morning): avg_delay=36.4%, breach_rate=95.0%
- BOM -> BLR (Carting, night): avg_delay=55.1%, breach_rate=94.7%
- DEL -> JAI (Carting, afternoon): avg_delay=47.3%, breach_rate=100.0%

## Recommended interventions
- Parallel route activation for high-delay, high-volume corridors.
- Facility process/capacity upgrades for top chokepoint hubs.
- Route-type policy shifts (FTL vs Carting) by corridor and time bucket.

## Estimated impact (top-3 hub upgrade scenario)
- Estimated late-delivery reduction: 7.85%
- Estimated revenue-at-risk recovered: 35877.87
