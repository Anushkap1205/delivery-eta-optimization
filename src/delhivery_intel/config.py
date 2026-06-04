from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
METRICS = OUTPUTS / "metrics"
REPORTS = PROJECT_ROOT / "reports"


def ensure_dirs() -> None:
    for path in [DATA_RAW, DATA_PROCESSED, FIGURES, TABLES, METRICS, REPORTS]:
        path.mkdir(parents=True, exist_ok=True)

