from __future__ import annotations

import random
from datetime import datetime, timedelta
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from delhivery_intel.config import DATA_RAW, ensure_dirs


def main(n: int = 8000) -> None:
    ensure_dirs()
    random.seed(42)
    np.random.seed(42)

    hubs = ["BLR", "DEL", "BOM", "HYD", "PNQ", "CCU", "LKO", "JAI", "AMD", "NAG"]
    route_types = ["FTL", "Carting"]
    start = datetime(2025, 1, 1)
    rows = []
    for i in range(n):
        src = random.choice(hubs)
        dst = random.choice([h for h in hubs if h != src])
        dep = start + timedelta(minutes=random.randint(0, 365 * 24 * 60))
        distance = random.uniform(40, 1800)
        osrm = max(distance * random.uniform(1.0, 1.8), 30)
        rtype = random.choice(route_types)
        corridor_risk = 1.0 + (0.25 if src in {"DEL", "BOM"} else 0.0) + (0.20 if dst in {"BLR", "DEL"} else 0.0)
        rt_factor = 0.95 if rtype == "FTL" else 1.08
        rush = 1.12 if dep.hour in [9, 10, 11, 17, 18, 19] else 1.0
        noise = np.random.lognormal(mean=0.0, sigma=0.15)
        actual = osrm * corridor_risk * rt_factor * rush * noise
        promised = osrm * random.uniform(1.02, 1.15)
        arr = dep + timedelta(minutes=float(actual))

        rows.append(
            {
                "shipment_id": f"SHP{i:06d}",
                "source_facility": src,
                "dest_facility": dst,
                "departure_ts": dep.isoformat(),
                "arrival_ts": arr.isoformat(),
                "route_type": rtype,
                "osrm_eta_minutes": round(osrm, 2),
                "actual_transit_minutes": round(actual, 2),
                "promised_eta_minutes": round(promised, 2),
                "distance_km": round(distance, 2),
            }
        )

    df = pd.DataFrame(rows)
    out = DATA_RAW / "trips.csv"
    df.to_csv(out, index=False)
    print(f"Sample data written: {out}")


if __name__ == "__main__":
    main()

