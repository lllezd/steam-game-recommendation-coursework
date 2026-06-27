from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs" / "llm_baselines"
VAL_PATH = OUT_DIR / "zero_shot_pilot_predictions_val.csv"
TEST_PATH = OUT_DIR / "zero_shot_pilot_predictions_test.csv"

def calc_metrics(y_true, score, pred):
    return {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }

def choose_threshold(y_true, score):
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        pred = score >= threshold
        cur_f1 = f1_score(y_true, pred, zero_division=0)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = threshold
    return float(best_threshold), float(best_f1)

def describe_scores(df, name):
    print(f"\n{name} score by label:")
    print(df.groupby("label")["score_yes"].describe())
    print(f"\n{name} generated answer counts:")
    print(df["generated_answer"].value_counts(dropna=False))

def main():
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)

    y_val = val["label"].values
    y_test = test["label"].values
    val_score = val["score_yes"].values
    test_score = test["score_yes"].values

    threshold, val_best_f1 = choose_threshold(y_val, val_score)

    rows = []
    for name, df, y, score in [
        ("val", val, y_val, val_score),
        ("test", test, y_test, test_score),
    ]:
        rows.append({
            "split": name,
            "mode": "threshold_0_5",
            **calc_metrics(y, score, (score >= 0.5).astype(int)),
        })
        rows.append({
            "split": name,
            "mode": "generated_answer",
            **calc_metrics(y, score, df["parsed_answer"].astype(int).values),
        })
        rows.append({
            "split": name,
            "mode": f"val_best_threshold_{threshold:.3f}",
            **calc_metrics(y, score, (score >= threshold).astype(int)),
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "zero_shot_pilot_analysis.csv", index=False)

    print(f"Best threshold on val: {threshold:.6f}")
    print(f"Best val F1: {val_best_f1:.6f}")
    print("\nMetrics:")
    print(result.to_string(index=False))

    describe_scores(val, "Val")
    describe_scores(test, "Test")

    print("\nSaved:")
    print(OUT_DIR / "zero_shot_pilot_analysis.csv")

if __name__ == "__main__":
    main()