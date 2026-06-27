from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = PROJECT_ROOT / "outputs" / "baselines" / "metrics_baselines.csv"
LLM_PATH = PROJECT_ROOT / "outputs" / "llm_baselines" / "llm_metrics.csv"

COMMON_COLS = [
    "group",
    "method",
    "model",
    "eval_type",
    "val_roc_auc",
    "test_roc_auc",
    "test_pr_auc",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "threshold",
]

def prepare_baselines():
    df = pd.read_csv(BASELINE_PATH)
    df["group"] = "non_llm"
    df["model"] = ""
    df["eval_type"] = "full_test"
    return df

def prepare_llm():
    if not LLM_PATH.exists():
        return pd.DataFrame(columns=COMMON_COLS)
    df = pd.read_csv(LLM_PATH)
    if "model" not in df.columns:
        df["model"] = ""
    if "eval_type" not in df.columns:
        df["eval_type"] = ""
    df["group"] = "llm"
    return df

def main():
    baseline = prepare_baselines()
    llm = prepare_llm()
    df = pd.concat([baseline, llm], ignore_index=True)
    for col in COMMON_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[COMMON_COLS].sort_values("test_roc_auc", ascending=False)
    out_path = PROJECT_ROOT / "outputs" / "llm_baselines" / "llm_vs_baselines_summary.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print("\nSaved:")
    print(out_path)

if __name__ == "__main__":
    main()