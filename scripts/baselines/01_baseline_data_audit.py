from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "final" / "tabular_temporal"
OUT_DIR = PROJECT_ROOT / "outputs" / "baselines" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data" / "full_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": DATA_DIR / "train_tabular.parquet",
    "val": DATA_DIR / "val_tabular.parquet",
    "test": DATA_DIR / "test_tabular.parquet",
}

LABEL_COL = "label_strict"

RISKY_FEATURES = [
    "positive_ratio",
    "user_reviews",
    "positive",
    "negative",
    "recommendations",
    "peak_ccu",
    "average_playtime_forever",
    "median_playtime_forever",
    "pct_pos_total",
    "num_reviews_total",
    "pct_pos_recent",
    "num_reviews_recent",
]

EXTRA_LABELS = [
    "label_hybrid",
    "label_hours_relative",
]

ID_TEXT_COLS = [
    "user_id",
    "steamid",
    "app_id",
    "target_app_id",
    "target_name",
    "target_title",
    "target_description",
    "target_short_description",
    "target_about_the_game",
    "history",
    "liked_history",
    "disliked_history",
    "prompt",
    "instruction",
    "input",
    "output",
]

PREFERRED_SAFE_FEATURES = [
    "history_len",
    "history_positive_count",
    "history_negative_count",
    "history_positive_share",
    "history_total_hours",
    "history_mean_hours",
    "history_median_hours",
    "history_liked_mean_hours",
    "history_disliked_mean_hours",
    "target_liked_jaccard",
    "target_disliked_jaccard",
    "target_jaccard_diff",
    "target_liked_overlap_count",
    "target_disliked_overlap_count",
    "target_text_len",
    "target_description_len",
    "target_short_description_len",
    "price",
    "release_year",
    "windows",
    "mac",
    "linux",
]

def read_data():
    data = {}
    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        data[name] = pd.read_parquet(path)
    return data

def label_balance(df):
    counts = df[LABEL_COL].value_counts(dropna=False).sort_index()
    rates = df[LABEL_COL].value_counts(normalize=True, dropna=False).sort_index()
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "rates": {str(k): float(v) for k, v in rates.items()},
    }

def null_report(df):
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    return {col: int(val) for col, val in nulls.items()}

def dtype_report(df):
    return {col: str(dtype) for col, dtype in df.dtypes.items()}

def numeric_columns(df):
    return df.select_dtypes(include=["number", "bool"]).columns.tolist()

def existing(cols, df):
    return [col for col in cols if col in df.columns]

def main():
    data = read_data()
    train = data["train"]
    all_columns = train.columns.tolist()
    missing_label = [name for name, df in data.items() if LABEL_COL not in df.columns]
    if missing_label:
        raise ValueError(f"{LABEL_COL} is missing in: {missing_label}")

    risky_existing = existing(RISKY_FEATURES, train)
    extra_labels_existing = existing(EXTRA_LABELS, train)
    id_text_existing = existing(ID_TEXT_COLS, train)
    preferred_safe_existing = existing(PREFERRED_SAFE_FEATURES, train)

    blocked = set([LABEL_COL] + risky_existing + extra_labels_existing + id_text_existing)
    safe_numeric_auto = [col for col in numeric_columns(train) if col not in blocked]
    safe_preferred_numeric = [col for col in preferred_safe_existing if col in safe_numeric_auto]

    audit = {
        "paths": {name: str(path) for name, path in FILES.items()},
        "shapes": {name: list(df.shape) for name, df in data.items()},
        "label": LABEL_COL,
        "label_balance": {name: label_balance(df) for name, df in data.items()},
        "columns": all_columns,
        "dtypes": dtype_report(train),
        "nulls": {name: null_report(df) for name, df in data.items()},
        "risky_existing": risky_existing,
        "extra_labels_existing": extra_labels_existing,
        "id_text_existing": id_text_existing,
        "safe_numeric_auto": safe_numeric_auto,
        "safe_preferred_numeric": safe_preferred_numeric,
        "safe_missing_from_preferred": [col for col in PREFERRED_SAFE_FEATURES if col not in train.columns],
    }

    feature_lists = {
        "label": LABEL_COL,
        "safe_features_logreg": safe_preferred_numeric,
        "safe_features_catboost": safe_numeric_auto,
        "risky_features_analysis_only": risky_existing,
        "excluded_extra_labels": extra_labels_existing,
        "excluded_id_text_cols": id_text_existing,
    }

    (OUT_DIR / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "feature_lists.json").write_text(json.dumps(feature_lists, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Shapes:")
    for name, shape in audit["shapes"].items():
        print(f"{name}: {shape}")

    print("\nLabel balance:")
    for name, balance in audit["label_balance"].items():
        print(f"{name}: {balance}")

    print("\nRisky features found:")
    print(risky_existing)

    print("\nSafe features for Logistic Regression:")
    print(safe_preferred_numeric)

    print("\nSafe numeric features for CatBoost/LightGBM:")
    print(safe_numeric_auto)

    print("\nSaved:")
    print(OUT_DIR / "data_audit.json")
    print(OUT_DIR / "feature_lists.json")

if __name__ == "__main__":
    main()