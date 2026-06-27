from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "user_histories_mvp_v4_temporal.parquet"
OUT_DIR = PROJECT_ROOT / "outputs" / "baselines" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

USER_COL = "user_id"
ITEM_COL = "target_app_id"
LABEL_COL = "label_strict"
METHOD = "itemknn_cf_sample_history"
TOP_K = 30

def as_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return []

def split_history_items(app_ids, labels):
    app_ids = as_list(app_ids)
    labels = as_list(labels)
    liked, disliked = [], []
    for item, label in zip(app_ids, labels):
        try:
            item = int(item)
        except Exception:
            continue
        if bool(label):
            liked.append(item)
        else:
            disliked.append(item)
    return liked, disliked

def read_data():
    df = pd.read_parquet(DATA_PATH)
    return {
        "train": df[df["split"] == "train"].copy(),
        "val": df[df["split"] == "val"].copy(),
        "test": df[df["split"] == "test"].copy(),
    }

def build_positive_events(train):
    events = []
    cols = [USER_COL, ITEM_COL, LABEL_COL, "history_app_ids", "history_labels"]
    for row in train[cols].itertuples(index=False):
        liked, _ = split_history_items(row.history_app_ids, row.history_labels)
        for item in liked:
            events.append((int(row.user_id), int(item)))
        if bool(row.label_strict):
            events.append((int(row.user_id), int(row.target_app_id)))
    events = pd.DataFrame(events, columns=[USER_COL, ITEM_COL]).drop_duplicates()
    return events

def build_item_user_matrix(train):
    events = build_positive_events(train)
    users = np.sort(events[USER_COL].unique())
    items = np.sort(events[ITEM_COL].unique())
    user_to_idx = {user: idx for idx, user in enumerate(users)}
    item_to_idx = {item: idx for idx, item in enumerate(items)}
    rows = events[ITEM_COL].map(item_to_idx).values
    cols = events[USER_COL].map(user_to_idx).values
    data = np.ones(len(events), dtype=np.float32)
    matrix = csr_matrix((data, (rows, cols)), shape=(len(items), len(users)), dtype=np.float32)
    matrix = normalize(matrix, norm="l2", axis=1)
    return matrix, item_to_idx, len(events)

def mean_top_similarity(target_idx, hist_items, matrix, item_to_idx):
    hist_idx = [item_to_idx[item] for item in hist_items if item in item_to_idx and item_to_idx[item] != target_idx]
    if not hist_idx:
        return 0.0
    sims = (matrix[target_idx] @ matrix[hist_idx].T).toarray().ravel()
    sims = sims[sims > 0]
    if sims.size == 0:
        return 0.0
    sims = np.sort(sims)[::-1][:TOP_K]
    return float(sims.mean())

def score_row(row, matrix, item_to_idx):
    item_id = int(row.target_app_id)
    if item_id not in item_to_idx:
        return 0.0
    liked, disliked = split_history_items(row.history_app_ids, row.history_labels)
    target_idx = item_to_idx[item_id]
    liked_score = mean_top_similarity(target_idx, liked, matrix, item_to_idx)
    disliked_score = mean_top_similarity(target_idx, disliked, matrix, item_to_idx)
    return liked_score - disliked_score

def score_split(df, matrix, item_to_idx):
    cols = [ITEM_COL, "history_app_ids", "history_labels"]
    return np.array([score_row(row, matrix, item_to_idx) for row in df[cols].itertuples(index=False)], dtype=float)

def choose_threshold(y_true, score):
    finite_score = score[np.isfinite(score)]
    candidates = np.quantile(finite_score, np.linspace(0.0, 1.0, 1001))
    candidates = np.unique(np.concatenate([candidates, [finite_score.min() - 1e-12, finite_score.max() + 1e-12, 0.0]]))
    best_threshold = 0.0
    best_f1 = -1.0
    for threshold in candidates:
        pred = score >= threshold
        cur_f1 = f1_score(y_true, pred, zero_division=0)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = threshold
    return float(best_threshold)

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

def append_metrics(row):
    path = OUT_DIR / "metrics_baselines.csv"
    df_new = pd.DataFrame([row])
    if path.exists():
        df_old = pd.read_csv(path)
        df_old = df_old[~df_old["method"].isin([row["method"], "itemknn_cf_pos_minus_neg"])]
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(path, index=False)

def save_threshold(method, threshold):
    path = OUT_DIR / "thresholds.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.pop("itemknn_cf_pos_minus_neg", None)
    data[method] = threshold
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def coverage_report(df, item_to_idx):
    known_targets = int(df[ITEM_COL].isin(set(item_to_idx)).sum())
    liked_known, disliked_known = 0, 0
    for row in df[["history_app_ids", "history_labels"]].itertuples(index=False):
        liked, disliked = split_history_items(row.history_app_ids, row.history_labels)
        if any(item in item_to_idx for item in liked):
            liked_known += 1
        if any(item in item_to_idx for item in disliked):
            disliked_known += 1
    return known_targets, liked_known, disliked_known

def main():
    data = read_data()
    train, val, test = data["train"], data["val"], data["test"]
    matrix, item_to_idx, event_count = build_item_user_matrix(train)

    y_val = val[LABEL_COL].astype(int).values
    y_test = test[LABEL_COL].astype(int).values
    val_score = score_split(val, matrix, item_to_idx)
    test_score = score_split(test, matrix, item_to_idx)

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

    val_cov = coverage_report(val, item_to_idx)
    test_cov = coverage_report(test, item_to_idx)
    append_metrics(row)
    save_threshold(METHOD, threshold)

    print(f"Method: {METHOD}")
    print(f"Train shape: {train.shape}")
    print(f"Val shape: {val.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Positive item-user events: {event_count}")
    print(f"Item vectors: {matrix.shape[0]}")
    print(f"User dimension: {matrix.shape[1]}")
    print(f"TOP_K: {TOP_K}")
    print(f"Val known target items: {val_cov[0]} / {len(val)}")
    print(f"Val samples with known liked history: {val_cov[1]} / {len(val)}")
    print(f"Val samples with known disliked history: {val_cov[2]} / {len(val)}")
    print(f"Test known target items: {test_cov[0]} / {len(test)}")
    print(f"Test samples with known liked history: {test_cov[1]} / {len(test)}")
    print(f"Test samples with known disliked history: {test_cov[2]} / {len(test)}")
    print(f"Selected threshold on val: {threshold:.6f}")
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