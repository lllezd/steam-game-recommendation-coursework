import ast
import numpy as np
import pandas as pd
import duckdb
from data_script_paths import DATA_AUDIT_REPORTS_DIR, PROCESSED_DATA_DIR

IN_PATH = PROCESSED_DATA_DIR / "user_histories_mvp.parquet"
INTERACTIONS_PATH = PROCESSED_DATA_DIR / "cleaned_interactions.parquet"
OUT_PATH = PROCESSED_DATA_DIR / "user_histories_mvp_v2.parquet"
REPORT_PATH = DATA_AUDIT_REPORTS_DIR / "07_user_histories_v2_report.txt"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

HYBRID_MIN_HOURS = 2.0
MAX_HISTORY_TAGS = 6
MAX_TARGET_TAGS = 12
MAX_TARGET_GENRES = 8
MAX_TARGET_CATEGORIES = 8
MAX_DESCRIPTION_CHARS = 500

lines = []
def log(x=""):
    lines.append(str(x))

def clean_text(x, max_len=None):
    if x is None:
        return ""
    if isinstance(x, (list, tuple, np.ndarray)):
        x = ", ".join(str(v) for v in x)
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(x).replace("\n", " ").replace("\r", " ").strip()
    s = " ".join(s.split())
    if s in {"[]", "{}", "nan", "None", "<NA>"}:
        return ""
    if max_len and len(s) > max_len:
        s = s[:max_len].rsplit(" ", 1)[0] + "..."
    return s

def parse_tokens(x, max_items=None):
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        raw = list(x)
    else:
        s = clean_text(x)
        if not s:
            return []
        raw = None
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    raw = list(parsed)
            except Exception:
                raw = None
        if raw is None:
            raw = s.split(",") if "," in s else s.split(";") if ";" in s else [s]
    out = []
    seen = set()
    for v in raw:
        t = clean_text(v).strip(" '\"[]")
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out[:max_items] if max_items else out

def field_text(x, max_items=None, max_chars=None):
    s = ", ".join(parse_tokens(x, max_items=max_items))
    if max_chars and len(s) > max_chars:
        s = s[:max_chars].rsplit(" ", 1)[0] + "..."
    return s

def as_list(x):
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

def yes_no(v):
    return "Yes" if bool(v) else "No"

def hours_bucket(h):
    if pd.isna(h):
        return "unknown"
    h = float(h)
    if h <= 0:
        return "zero"
    if h < 1:
        return "very_low_0_1"
    if h < 5:
        return "low_1_5"
    if h < 20:
        return "medium_5_20"
    if h < 100:
        return "high_20_100"
    return "very_high_100_plus"

def build_prompt(row):
    titles = as_list(row["history_titles"])
    hours = as_list(row["history_hours"])
    labels = as_list(row["history_labels"])
    tags = as_list(row["history_tags"])
    genres = as_list(row["history_genres"])
    liked, disliked = [], []
    for i, title in enumerate(titles):
        title = clean_text(title, 120)
        tag_text = field_text(tags[i] if i < len(tags) else "", max_items=MAX_HISTORY_TAGS)
        genre_text = field_text(genres[i] if i < len(genres) else "", max_items=MAX_HISTORY_TAGS)
        meta = tag_text or genre_text
        h = hours[i] if i < len(hours) else None
        item = title
        if meta:
            item += f" - tags: {meta}"
        if h is not None and not pd.isna(h):
            item += f"; {float(h):.1f} hours"
        if i < len(labels) and bool(labels[i]):
            liked.append(item)
        else:
            disliked.append(item)
    liked_text = "\n".join(f"{i+1}. {x}" for i, x in enumerate(liked)) if liked else "None"
    disliked_text = "\n".join(f"{i+1}. {x}" for i, x in enumerate(disliked)) if disliked else "None"
    target_title = clean_text(row["target_title"], 150)
    target_genres = field_text(row["target_genres"], max_items=MAX_TARGET_GENRES)
    target_categories = field_text(row["target_categories"], max_items=MAX_TARGET_CATEGORIES)
    target_tags = field_text(row["target_tags"], max_items=MAX_TARGET_TAGS)
    target_desc = clean_text(row["target_description"], MAX_DESCRIPTION_CHARS)
    target_lines = [f"Title: {target_title}"]
    if target_genres:
        target_lines.append(f"Genres: {target_genres}")
    if target_categories:
        target_lines.append(f"Categories: {target_categories}")
    if target_tags:
        target_lines.append(f"Tags: {target_tags}")
    if target_desc:
        target_lines.append(f"Description: {target_desc}")
    return (
        "Instruction:\n"
        "Given the user's Steam game history, determine whether the user will recommend the target game. Answer only Yes or No.\n\n"
        "Input:\n"
        "User's liked games:\n"
        f"{liked_text}\n\n"
        "User's disliked games:\n"
        f"{disliked_text}\n\n"
        "Target game:\n"
        f"{chr(10).join(target_lines)}"
    )

log("Loading user histories")
df = pd.read_parquet(IN_PATH)
log(f"input_rows: {len(df)}")
log(f"input_columns: {df.shape[1]}")

log("Computing per-game hours references")
con = duckdb.connect()
game_hours = con.execute(f"""
SELECT
    app_id AS target_app_id,
    COUNT(*) AS target_game_interactions,
    approx_quantile(hours, 0.5) AS target_game_median_hours,
    approx_quantile(hours, 0.75) AS target_game_p75_hours
FROM read_parquet('{INTERACTIONS_PATH.as_posix()}')
GROUP BY app_id
""").fetchdf()

df = df.merge(game_hours, on="target_app_id", how="left")
df["target_hours_reference"] = df["target_game_median_hours"]
df.loc[df["target_hours_reference"].isna() | (df["target_hours_reference"] <= 0), "target_hours_reference"] = df["target_game_p75_hours"]
df.loc[df["target_hours_reference"].isna() | (df["target_hours_reference"] <= 0), "target_hours_reference"] = 1.0
df["target_hours_rel_to_game"] = df["target_hours_aux"] / df["target_hours_reference"]
df["target_hours_bucket"] = df["target_hours_aux"].apply(hours_bucket)

df["label_strict"] = df["label"].astype(bool)
df["output_text_strict"] = df["label_strict"].map(yes_no)
df["label_hybrid"] = df["label_strict"] & (df["target_hours_aux"] >= HYBRID_MIN_HOURS)
df["output_text_hybrid"] = df["label_hybrid"].map(yes_no)
df["label_hours_relative"] = df["target_hours_rel_to_game"] >= 1.0
df["output_text_hours_relative"] = df["label_hours_relative"].map(yes_no)
df["is_low_hours_positive"] = df["label_strict"] & (df["target_hours_aux"] < HYBRID_MIN_HOURS)
df["is_high_hours_negative"] = (~df["label_strict"]) & (df["target_hours_rel_to_game"] >= 1.0)

if "prompt_text" in df.columns:
    df = df.rename(columns={"prompt_text":"prompt_text_v1"})
df["prompt_text"] = df.apply(build_prompt, axis=1)

df.to_parquet(OUT_PATH, index=False, compression="zstd")

log("\nSaved dataset")
log(f"output_path: {OUT_PATH}")
log(f"rows: {len(df)}")
log(f"columns: {df.shape[1]}")

log("\nSplit counts")
log(df["split"].value_counts().sort_index().to_string())

log("\nStrict label counts")
log(df["output_text_strict"].value_counts().to_string())

log("\nHybrid label counts")
log(df["output_text_hybrid"].value_counts().to_string())

log("\nHours-relative label counts")
log(df["output_text_hours_relative"].value_counts().to_string())

log("\nTarget hours bucket")
log(df["target_hours_bucket"].value_counts().to_string())

log("\nLow-hours positive / high-hours negative")
log(f"low_hours_positive_count: {int(df['is_low_hours_positive'].sum())}")
log(f"high_hours_negative_count: {int(df['is_high_hours_negative'].sum())}")

log("\nSplit x strict label")
log(pd.crosstab(df["split"], df["output_text_strict"]).to_string())

log("\nSplit x hybrid label")
log(pd.crosstab(df["split"], df["output_text_hybrid"]).to_string())

log("\nSplit x hours-relative label")
log(pd.crosstab(df["split"], df["output_text_hours_relative"]).to_string())

log("\nTarget hours stats")
log(df["target_hours_aux"].describe().to_string())

log("\nTarget hours relative stats")
log(df["target_hours_rel_to_game"].describe().to_string())

log("\nMissing share")
check_cols = ["target_title","target_tags","target_genres","target_categories","target_description","prompt_text","target_game_median_hours"]
log(df[check_cols].isna().mean().sort_values(ascending=False).to_string())

log("\nExample prompt v2")
log(df.iloc[0]["prompt_text"])

log("\nExample labels")
example_cols = ["output_text_strict","output_text_hybrid","output_text_hours_relative","target_hours_aux","target_hours_reference","target_hours_rel_to_game","target_hours_bucket"]
log(df.iloc[0][example_cols].to_string())

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: {OUT_PATH}")
print(f"Saved report: {REPORT_PATH}")
