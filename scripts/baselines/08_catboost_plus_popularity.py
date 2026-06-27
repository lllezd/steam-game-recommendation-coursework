from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "final" / "tabular_temporal"
OUT_DIR = PROJECT_ROOT / "outputs" / "baselines" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label_strict"
ITEM_COL = "target_app_id"
METHOD = "safe_catboost_plus_popularity"
SEED = 42
USE_GPU = False
ALPHA = 10.0

BASE_FEATURES = [
    "history_len",
    "history_positive_count",
    "history_negative_count",
    "history_positive_share",
    "history_total_hours",
    "history_mean_hours",
    "history_median_hours",
    "history_max_hours",
    "history_min_hours",
    "history_liked_mean_hours",
    "history_disliked_mean_hours",
    "target_token_count",
    "liked_token_count",
    "disliked_token_count",
    "target_liked_overlap_count",
    "target_disliked_overlap_count",
    "target_liked_jaccard",
    "target_disliked_jaccard",
    "target_jaccard_diff",
    "target_description_len",
    "target_title_len",
    "price",
    "required_age",
    "dlc_count",
    "release_year",
]

POP_FEATURES = [
    "train_item_popularity_score",
    "train_item_log_count",
]

FEATURES = BASE_FEATURES + POP_FEATURES

def read_split(name):
    return pd.read_parquet(DATA_DIR / f"{name}_tabular.parquet")

def build_item_stats(df, global_rate):
    stats = df.groupby(ITEM_COL)[LABEL_COL].agg(["sum", "count"]).reset_index()
    stats["train_item_popularity_score"] = (stats["sum"] + ALPHA * global_rate) / (stats["count"] + ALPHA)
    stats["train_item_log_count"] = np.log1p(stats["count"])
    return stats[[ITEM_COL, "train_item_popularity_score", "train_item_log_count"]]

def add_full_train_popularity(df, train):
    global_rate = float(train[LABEL_COL].mean())
    stats = build_item_stats(train, global_rate)
    result = df.merge(stats, on=ITEM_COL, how="left")
    result["train_item_popularity_score"] = result["train_item_popularity_score"].fillna(global_rate)
    result["train_item_log_count"] = result["train_item_log_count"].fillna(0.0)
    return result

def add_oof_train_popularity(train):
    result = train.copy()
    result["train_item_popularity_score"] = np.nan
    result["train_item_log_count"] = np.nan
    y = train[LABEL_COL].astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fit_idx, hold_idx in skf.split(train, y):
        fit_part = train.iloc[fit_idx]
        hold_part = train.iloc[hold_idx]
        global_rate = float(fit_part[LABEL_COL].mean())
        stats = build_item_stats(fit_part, global_rate)
        encoded = hold_part[[ITEM_COL]].merge(stats, on=ITEM_COL, how="left")
        result.iloc[hold_idx, result.columns.get_loc("train_item_popularity_score")] = encoded["train_item_popularity_score"].fillna(global_rate).values
        result.iloc[hold_idx, result.columns.get_loc("train_item_log_count")] = encoded["train_item_log_count"].fillna(0.0).values
    return result

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
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
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

def main():
    train_raw = read_split("train")
    val_raw = read_split("val")
    test_raw = read_split("test")

    missing = [col for col in BASE_FEATURES if col not in train_raw.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    train = add_oof_train_popularity(train_raw)
    val = add_full_train_popularity(val_raw, train_raw)
    test = add_full_train_popularity(test_raw, train_raw)

    X_train = train[FEATURES]
    X_val = val[FEATURES]
    X_test = test[FEATURES]

    y_train = train[LABEL_COL].astype(int).values
    y_val = val[LABEL_COL].astype(int).values
    y_test = test[LABEL_COL].astype(int).values

    params = {
        "iterations": 1000,
        "learning_rate": 0.04,
        "depth": 6,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": SEED,
        "verbose": 100,
        "allow_writing_files": False,
        "early_stopping_rounds": 80,
    }
    if USE_GPU:
        params["task_type"] = "GPU"
        params["devices"] = "0"

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]

    threshold = choose_threshold(y_val, val_score)
    val_metrics = calc_metrics(y_val, val_score, threshold)
    test_metrics = calc_metrics(y_test, test_score, threshold)

    row = {
        "method": METHOD,
        "val_roc_auc": val_metrics["roc_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "threshold": threshold,
    }

    feature_importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.get_feature_importance()
    }).sort_values("importance", ascending=False)

    append_metrics(row)
    save_threshold(METHOD, threshold)
    feature_importance.to_csv(OUT_DIR / "catboost_plus_popularity_feature_importance.csv", index=False)

    print(f"Method: {METHOD}")
    print(f"Features: {len(FEATURES)}")
    print(f"Best iteration: {model.get_best_iteration()}")
    print(f"Selected threshold on val: {threshold:.6f}")
    print("\nPopularity feature means:")
    print(f"train popularity mean: {train['train_item_popularity_score'].mean():.6f}")
    print(f"val popularity mean: {val['train_item_popularity_score'].mean():.6f}")
    print(f"test popularity mean: {test['train_item_popularity_score'].mean():.6f}")
    print("\nVal metrics:")
    print(val_metrics)
    print("\nTest metrics:")
    print(test_metrics)
    print("\nTop feature importance:")
    print(feature_importance.head(15))
    print("\nSaved:")
    print(OUT_DIR / "metrics_baselines.csv")
    print(OUT_DIR / "thresholds.json")
    print(OUT_DIR / "catboost_plus_popularity_feature_importance.csv")

if __name__ == "__main__":
    main()