from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "final" / "tabular_temporal"
OUT_DIR = PROJECT_ROOT / "outputs" / "baselines" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label_strict"
SCORE_COL = "target_jaccard_diff"

def read_split(name):
    return pd.read_parquet(DATA_DIR / f"{name}_tabular.parquet")

def calc_metrics(y_true, score, threshold):
    pred = score >= threshold
    return {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }

def choose_threshold(y_true, score):
    finite_score = score[np.isfinite(score)]
    candidates = np.quantile(finite_score, np.linspace(0.0, 1.0, 1001))
    candidates = np.unique(np.concatenate([
        candidates,
        [finite_score.min() - 1e-12, finite_score.max() + 1e-12, 0.0]
    ]))
    best_threshold = 0.0
    best_f1 = -1.0
    for threshold in candidates:
        pred = score >= threshold
        cur_f1 = f1_score(y_true, pred, zero_division=0)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = threshold
    return float(best_threshold)

def append_metrics(row):
    path = OUT_DIR / "metrics_baselines.csv"
    df_new = pd.DataFrame([row])
    if path.exists():
        df_old = pd.read_csv(path)
        df_old = df_old[df_old["method"] != row["method"]]
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(path, index=False)

def save_threshold(method, threshold):
    path = OUT_DIR / "thresholds.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[method] = threshold
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def get_score(df):
    return df[SCORE_COL].fillna(0.0).astype(float).values

def main():
    train = read_split("train")
    val = read_split("val")
    test = read_split("test")

    y_val = val[LABEL_COL].astype(int).values
    y_test = test[LABEL_COL].astype(int).values

    val_score = get_score(val)
    test_score = get_score(test)

    threshold = choose_threshold(y_val, val_score)
    val_metrics = calc_metrics(y_val, val_score, threshold)
    test_metrics = calc_metrics(y_test, test_score, threshold)

    row = {
        "method": "content_jaccard_diff",
        "val_roc_auc": val_metrics["roc_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "threshold": threshold,
    }

    append_metrics(row)
    save_threshold(row["method"], threshold)

    print("Method: content_jaccard_diff")
    print(f"Score column: {SCORE_COL}")
    print(f"Selected threshold on val: {threshold:.6f}")
    print(f"Train score mean: {get_score(train).mean():.6f}")
    print(f"Val score mean: {val_score.mean():.6f}")
    print(f"Test score mean: {test_score.mean():.6f}")
    print("\nVal metrics:")
    print(val_metrics)
    print("\nTest metrics:")
    print(test_metrics)
    print("\nSaved:")
    print(OUT_DIR / "metrics_baselines.csv")
    print(OUT_DIR / "thresholds.json")

if __name__ == "__main__":
    main()