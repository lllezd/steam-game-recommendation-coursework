import ast
import re
import numpy as np
import pandas as pd
from data_script_paths import DATA_AUDIT_REPORTS_DIR, FINAL_DATA_DIR, PROCESSED_DATA_DIR

IN_PATH = PROCESSED_DATA_DIR / "user_histories_mvp_v3.parquet"
ITEMS_PATH = PROCESSED_DATA_DIR / "item_metadata.parquet"
OUT_DIR = FINAL_DATA_DIR / "tabular"
REPORT_PATH = DATA_AUDIT_REPORTS_DIR / "10_tabular_baseline_report.txt"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

JUNK_TOKENS = {
    "steam achievements","steam trading cards","steam cloud","controller","partial controller support",
    "full controller support","remote play","family sharing","stats","leaderboards","steam workshop"
}

lines = []
def log(x=""):
    lines.append(str(x))

def clean_text(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(x).replace("\n", " ").replace("\r", " ").split())

def to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if x is None:
        return []
    try:
        if pd.isna(x):
            return []
    except (TypeError, ValueError):
        pass
    return list(x)

def parse_tokens(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        raw = []
        for v in x:
            raw.extend(parse_tokens(v))
        return raw
    s = clean_text(x)
    if not s:
        return []
    raw = None
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                raw = parsed
        except Exception:
            raw = None
    if raw is None:
        raw = re.split(r"[,;|]", s)
    out, seen = [], set()
    for v in raw:
        t = clean_text(v).strip(" '\"[]").lower()
        if t and t not in JUNK_TOKENS and t not in seen:
            out.append(t)
            seen.add(t)
    return out

def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def safe_float_list(x):
    out = []
    for v in to_list(x):
        try:
            if not pd.isna(v):
                out.append(float(v))
        except Exception:
            pass
    return out

def parse_estimated_owners(x):
    s = clean_text(x).replace(",", "")
    nums = [int(v) for v in re.findall(r"\d+", s)]
    if not nums:
        return pd.Series([np.nan, np.nan, np.nan])
    if len(nums) == 1:
        return pd.Series([nums[0], nums[0], nums[0]])
    return pd.Series([nums[0], nums[1], (nums[0] + nums[1]) / 2])

def build_features(row):
    labels = [bool(v) for v in to_list(row["history_labels"])]
    hours = safe_float_list(row["history_hours"])
    h = np.array(hours, dtype=float) if hours else np.array([], dtype=float)
    liked_count = sum(labels)
    disliked_count = len(labels) - liked_count
    liked_hours = np.array([hours[i] for i, v in enumerate(labels) if v and i < len(hours)], dtype=float)
    disliked_hours = np.array([hours[i] for i, v in enumerate(labels) if (not v) and i < len(hours)], dtype=float)
    liked_tokens, disliked_tokens = set(), set()
    history_tags = to_list(row["history_tags"])
    history_genres = to_list(row["history_genres"])
    for i in range(len(labels)):
        toks = set(parse_tokens(history_tags[i] if i < len(history_tags) else "") + parse_tokens(history_genres[i] if i < len(history_genres) else ""))
        if labels[i]:
            liked_tokens |= toks
        else:
            disliked_tokens |= toks
    target_tokens = set(parse_tokens(row["target_tags"]) + parse_tokens(row["target_genres"]) + parse_tokens(row["target_categories"]))
    return pd.Series({
        "history_positive_count": liked_count,
        "history_negative_count": disliked_count,
        "history_positive_share": liked_count / len(labels) if labels else 0.0,
        "history_total_hours": float(h.sum()) if h.size else 0.0,
        "history_mean_hours": float(h.mean()) if h.size else 0.0,
        "history_median_hours": float(np.median(h)) if h.size else 0.0,
        "history_max_hours": float(h.max()) if h.size else 0.0,
        "history_min_hours": float(h.min()) if h.size else 0.0,
        "history_liked_mean_hours": float(liked_hours.mean()) if liked_hours.size else 0.0,
        "history_disliked_mean_hours": float(disliked_hours.mean()) if disliked_hours.size else 0.0,
        "target_token_count": len(target_tokens),
        "liked_token_count": len(liked_tokens),
        "disliked_token_count": len(disliked_tokens),
        "target_liked_overlap_count": len(target_tokens & liked_tokens),
        "target_disliked_overlap_count": len(target_tokens & disliked_tokens),
        "target_liked_jaccard": jaccard(target_tokens, liked_tokens),
        "target_disliked_jaccard": jaccard(target_tokens, disliked_tokens),
        "target_jaccard_diff": jaccard(target_tokens, liked_tokens) - jaccard(target_tokens, disliked_tokens),
        "target_description_len": len(clean_text(row["target_description"])),
        "target_title_len": len(clean_text(row["target_title"]))
    })

log("Loading datasets")
df = pd.read_parquet(IN_PATH)
items = pd.read_parquet(ITEMS_PATH)
log(f"user_histories_rows: {len(df)}")
log(f"user_histories_columns: {df.shape[1]}")
log(f"item_metadata_rows: {len(items)}")

log("Building history/content features")
features = df.apply(build_features, axis=1)
df = pd.concat([df, features], axis=1)

item_cols = [
    "app_id","has_artermiloff_metadata","release_date","price","rating","positive_ratio","user_reviews",
    "price_final","price_original","discount","steam_deck","required_age","dlc_count","estimated_owners",
    "average_playtime_forever","average_playtime_2weeks","median_playtime_forever","median_playtime_2weeks",
    "peak_ccu","positive","negative","recommendations","pct_pos_total","num_reviews_total","pct_pos_recent",
    "num_reviews_recent"
]
items = items[[c for c in item_cols if c in items.columns]].copy()
items = items.rename(columns={"app_id":"target_app_id"})
owners = items["estimated_owners"].apply(parse_estimated_owners) if "estimated_owners" in items.columns else pd.DataFrame([[np.nan, np.nan, np.nan]] * len(items))
owners.columns = ["estimated_owners_min","estimated_owners_max","estimated_owners_mid"]
items = pd.concat([items, owners], axis=1)
if "release_date" in items.columns:
    items["release_year"] = pd.to_datetime(items["release_date"], errors="coerce").dt.year
df = df.merge(items, on="target_app_id", how="left")

id_cols = ["sample_id","user_id","target_app_id","split"]
label_cols = [
    "label_strict","output_text_strict","label_hybrid","output_text_hybrid",
    "label_hours_relative","output_text_hours_relative"
]
text_cols = [
    "target_title","target_tags","target_genres","target_categories","target_description","target_item_text_prompt"
]
feature_cols = [
    "history_len","history_positive_count","history_negative_count","history_positive_share",
    "history_total_hours","history_mean_hours","history_median_hours","history_max_hours","history_min_hours",
    "history_liked_mean_hours","history_disliked_mean_hours","target_token_count","liked_token_count",
    "disliked_token_count","target_liked_overlap_count","target_disliked_overlap_count",
    "target_liked_jaccard","target_disliked_jaccard","target_jaccard_diff","target_description_len","target_title_len",
    "has_artermiloff_metadata","price","positive_ratio","user_reviews","price_final","price_original","discount",
    "steam_deck","required_age","dlc_count","average_playtime_forever","average_playtime_2weeks",
    "median_playtime_forever","median_playtime_2weeks","peak_ccu","positive","negative","recommendations",
    "pct_pos_total","num_reviews_total","pct_pos_recent","num_reviews_recent",
    "estimated_owners_min","estimated_owners_max","estimated_owners_mid","release_year"
]
select_cols = [c for c in id_cols + label_cols + text_cols + feature_cols if c in df.columns]
tab = df[select_cols].copy()

log("Saving splits")
for split in ["train","val","test"]:
    part = tab[tab["split"] == split].copy()
    out_path = OUT_DIR / f"{split}_tabular.parquet"
    part.to_parquet(out_path, index=False, compression="zstd")
    log(f"{split}: rows={len(part)}, path={out_path}, size_mb={out_path.stat().st_size / 1024**2:.2f}")

log("\nDataset shape")
log(f"rows: {len(tab)}")
log(f"columns: {tab.shape[1]}")

log("\nSplit counts")
log(tab["split"].value_counts().sort_index().to_string())

log("\nStrict label balance")
log(pd.crosstab(tab["split"], tab["output_text_strict"]).to_string())

log("\nHybrid label balance")
log(pd.crosstab(tab["split"], tab["output_text_hybrid"]).to_string())

log("\nHours-relative label balance")
log(pd.crosstab(tab["split"], tab["output_text_hours_relative"]).to_string())

log("\nMissing share top")
log(tab.isna().mean().sort_values(ascending=False).head(30).to_string())

safe_feature_note = [
    "history_* features",
    "target token overlap features",
    "target text length features",
    "price/release_year/platform flags"
]
risky_feature_note = [
    "positive_ratio/user_reviews/positive/negative/recommendations",
    "peak_ccu/global playtime fields",
    "pct_pos_total/pct_pos_recent/num_reviews fields"
]
log("\nRecommended safe feature groups")
log("\n".join(safe_feature_note))
log("\nRisky popularity/global aggregate feature groups")
log("\n".join(risky_feature_note))

log("\nColumns")
log("\n".join(tab.columns))

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved tabular files to: {OUT_DIR}")
print(f"Saved report: {REPORT_PATH}")
