from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATA_DIR = DATA_DIR / "final"
TMP_DATA_DIR = DATA_DIR / "tmp"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_AUDIT_REPORTS_DIR = REPORTS_DIR / "data_audit"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
