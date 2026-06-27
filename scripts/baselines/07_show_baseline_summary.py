from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "outputs" / "baselines" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "metrics_baselines.csv"

def main():
    df = pd.read_csv(METRICS_PATH)
    cols = [
        "method",
        "val_roc_auc",
        "test_roc_auc",
        "test_pr_auc",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "threshold",
    ]
    df = df[cols].sort_values("test_roc_auc", ascending=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()