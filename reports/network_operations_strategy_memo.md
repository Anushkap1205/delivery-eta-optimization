# Network Operations Strategy Memo

## Executive takeaway
- Graph-enhanced ETA model outperforms baseline by measured business metrics.
- A small set of structurally central hubs contributes disproportionate SLA risk.
- Focused upgrades on top hubs are expected to reduce late deliveries and recover revenue-at-risk.

## Model performance
- Baseline MAE: 46.84
- Graph-enhanced MAE: 36.15
- Baseline % within 15%: 37.02%
- Graph-enhanced % within 15%: 48.12%
- Graph advantage (MAE reduction): 10.69
- Graph advantage (+/-15% accuracy): 11.10%

## Top bottleneck hubs
- IND000000ACB: risk_score=61.51
- IND562132AAA: risk_score=40.94
- IND501359AAE: risk_score=31.36
- IND421302AAG: risk_score=29.10
- IND160002AAC: risk_score=28.53

## Chronic delay corridors
- IND562132AAA -> IND560300AAA (Carting, morning): avg_delay=95.9%, breach_rate=100.0%
- IND400011AAA -> IND400072AAD (Carting, afternoon): avg_delay=194.8%, breach_rate=100.0%
- IND411033AAA -> IND421302AAG (FTL, night): avg_delay=127.6%, breach_rate=100.0%
- IND562132AAA -> IND560300AAA (Carting, evening): avg_delay=52.5%, breach_rate=89.4%
- IND400072AAD -> IND421302AAG (Carting, evening): avg_delay=184.6%, breach_rate=100.0%

## Recommended interventions
- Parallel route activation for high-delay, high-volume corridors.
- Facility process/capacity upgrades for top chokepoint hubs.
- Route-type policy shifts (FTL vs Carting) by corridor and time bucket.

## Estimated impact (top-3 hub upgrade scenario)
- Estimated late-delivery reduction: 2.80%
- Estimated revenue-at-risk recovered: 69590.45
