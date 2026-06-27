import pandas as pd
import duckdb
from data_script_paths import DATA_AUDIT_REPORTS_DIR, PROCESSED_DATA_DIR, TMP_DATA_DIR

INTERACTIONS_PATH = PROCESSED_DATA_DIR / "cleaned_interactions.parquet"
ITEMS_PATH = PROCESSED_DATA_DIR / "item_metadata.parquet"
OUT_PATH = PROCESSED_DATA_DIR / "user_histories_mvp.parquet"
REPORT_PATH = DATA_AUDIT_REPORTS_DIR / "06_user_histories_report.txt"
TMP_PATH = TMP_DATA_DIR / "user_histories_candidates.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
TMP_PATH.parent.mkdir(parents=True, exist_ok=True)

MIN_HISTORY_LEN = 4
MAX_HISTORY_LEN = 10
N_PER_CLASS = 50000
SEED_SALT = 42

lines = []
def log(x=""):
    lines.append(str(x))

def clean_text(x, max_len=None):
    if x is None or pd.isna(x):
        return ""
    x = str(x).replace("\n", " ").replace("\r", " ")
    x = " ".join(x.split())
    if max_len and len(x) > max_len:
        x = x[:max_len].rsplit(" ", 1)[0] + "..."
    return x

def as_list(x):
    if isinstance(x, list):
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    if x is None or pd.isna(x):
        return []
    return list(x)

def yes_no(v):
    return "Yes" if bool(v) else "No"

def build_prompt(row):
    titles = as_list(row["history_titles"])
    hours = as_list(row["history_hours"])
    labels = as_list(row["history_labels"])
    tags = as_list(row["history_tags"])
    liked, disliked = [], []
    for i, title in enumerate(titles):
        title = clean_text(title, 120)
        tag = clean_text(tags[i] if i < len(tags) else "", 180)
        h = hours[i] if i < len(hours) else None
        item = f"{title}"
        if tag:
            item += f" - {tag}"
        if h is not None and not pd.isna(h):
            item += f", {float(h):.1f} hours"
        if i < len(labels) and bool(labels[i]):
            liked.append(item)
        else:
            disliked.append(item)
    liked_text = "\n".join(f"{i+1}. {x}" for i, x in enumerate(liked)) if liked else "None"
    disliked_text = "\n".join(f"{i+1}. {x}" for i, x in enumerate(disliked)) if disliked else "None"
    target = clean_text(row["target_title"], 120)
    target_tags = clean_text(row["target_tags"], 220)
    target_genres = clean_text(row["target_genres"], 160)
    target_desc = clean_text(row["target_description"], 450)
    target_parts = [target]
    if target_genres:
        target_parts.append(f"Genres: {target_genres}")
    if target_tags:
        target_parts.append(f"Tags: {target_tags}")
    if target_desc:
        target_parts.append(f"Description: {target_desc}")
    target_text = "\n".join(target_parts)
    return (
        "Instruction:\n"
        "Given the user's Steam game history, determine whether the user will recommend the target game. Answer only Yes or No.\n\n"
        "Input:\n"
        "User's liked games:\n"
        f"{liked_text}\n\n"
        "User's disliked games:\n"
        f"{disliked_text}\n\n"
        "Target game:\n"
        f"{target_text}"
    )

con = duckdb.connect()
con.execute("PRAGMA threads=4")
log("Building user history candidates")
log(f"min_history_len: {MIN_HISTORY_LEN}")
log(f"max_history_len: {MAX_HISTORY_LEN}")
log(f"n_per_class: {N_PER_CLASS}")

con.execute(f"""
COPY (
WITH ranked AS (
    SELECT
        user_id,
        app_id,
        date,
        is_recommended,
        hours,
        review_id,
        helpful,
        funny,
        user_interaction_count,
        user_interaction_idx,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY date DESC, review_id DESC
        ) AS reverse_idx
    FROM read_parquet('{INTERACTIONS_PATH.as_posix()}')
),
targets AS (
    SELECT *
    FROM ranked
    WHERE reverse_idx = 1 AND user_interaction_idx > {MIN_HISTORY_LEN}
),
hist AS (
    SELECT
        t.user_id,
        t.app_id AS target_app_id,
        t.date AS target_date,
        t.is_recommended AS label,
        t.hours AS target_hours_aux,
        t.review_id AS target_review_id,
        h.app_id AS history_app_id,
        h.date AS history_date,
        h.is_recommended AS history_label,
        h.hours AS history_hour,
        h.user_interaction_idx AS history_idx
    FROM targets t
    JOIN ranked h ON t.user_id = h.user_id
    WHERE h.user_interaction_idx < t.user_interaction_idx
      AND h.user_interaction_idx >= t.user_interaction_idx - {MAX_HISTORY_LEN}
),
grouped AS (
    SELECT
        user_id,
        target_app_id,
        target_date,
        label,
        target_hours_aux,
        target_review_id,
        COUNT(*) AS history_len,
        list(history_app_id ORDER BY history_idx) AS history_app_ids,
        list(history_date ORDER BY history_idx) AS history_dates,
        list(history_label ORDER BY history_idx) AS history_labels,
        list(history_hour ORDER BY history_idx) AS history_hours
    FROM hist
    GROUP BY user_id, target_app_id, target_date, label, target_hours_aux, target_review_id
    HAVING COUNT(*) >= {MIN_HISTORY_LEN}
),
sampled AS (
    SELECT *
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY label
                ORDER BY hash(user_id + {SEED_SALT})
            ) AS label_sample_rank
        FROM grouped
    )
    WHERE label_sample_rank <= {N_PER_CLASS}
),
with_split AS (
    SELECT
        *,
        CASE
            WHEN ROW_NUMBER() OVER (PARTITION BY label ORDER BY target_date, user_id) <= 0.8 * COUNT(*) OVER (PARTITION BY label) THEN 'train'
            WHEN ROW_NUMBER() OVER (PARTITION BY label ORDER BY target_date, user_id) <= 0.9 * COUNT(*) OVER (PARTITION BY label) THEN 'val'
            ELSE 'test'
        END AS split
    FROM sampled
)
SELECT
    s.user_id,
    s.target_app_id,
    s.target_date,
    s.label,
    CASE WHEN s.label THEN 'Yes' ELSE 'No' END AS output_text,
    s.target_hours_aux,
    s.target_review_id,
    s.history_len,
    s.history_app_ids,
    s.history_dates,
    s.history_labels,
    s.history_hours,
    t.game_title AS target_title,
    t.tags AS target_tags,
    t.genres AS target_genres,
    t.categories AS target_categories,
    t.description AS target_description,
    t.item_text_prompt AS target_item_text_prompt,
    list(hm.game_title ORDER BY pos) AS history_titles,
    list(hm.tags ORDER BY pos) AS history_tags,
    list(hm.genres ORDER BY pos) AS history_genres,
    list(hm.description ORDER BY pos) AS history_descriptions,
    s.split
FROM with_split s
LEFT JOIN read_parquet('{ITEMS_PATH.as_posix()}') t ON s.target_app_id = t.app_id
LEFT JOIN UNNEST(s.history_app_ids) WITH ORDINALITY AS u(hist_app_id, pos) ON TRUE
LEFT JOIN read_parquet('{ITEMS_PATH.as_posix()}') hm ON u.hist_app_id = hm.app_id
GROUP BY
    s.user_id, s.target_app_id, s.target_date, s.label, s.target_hours_aux,
    s.target_review_id, s.history_len, s.history_app_ids, s.history_dates,
    s.history_labels, s.history_hours, t.game_title, t.tags, t.genres,
    t.categories, t.description, t.item_text_prompt, s.split
ORDER BY split, label, user_id
) TO '{TMP_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

log("Adding prompt_text in pandas")
df = pd.read_parquet(TMP_PATH)
df.insert(0, "sample_id", [f"steam_mvp_{i:06d}" for i in range(len(df))])
df["prompt_text"] = df.apply(build_prompt, axis=1)
df.to_parquet(OUT_PATH, index=False, compression="zstd")

log("\nSaved dataset")
log(f"output_path: {OUT_PATH}")
log(f"rows: {len(df)}")
log(f"columns: {df.shape[1]}")

log("\nSplit counts")
log(df["split"].value_counts().sort_index().to_string())

log("\nLabel counts")
log(df["output_text"].value_counts().to_string())

log("\nSplit x label")
log(pd.crosstab(df["split"], df["output_text"]).to_string())

log("\nHistory length stats")
log(df["history_len"].describe().to_string())

log("\nTarget date range by split")
log(df.groupby("split")["target_date"].agg(["min","max"]).to_string())

log("\nMissing share")
cols = ["target_title","target_tags","target_genres","target_description","history_titles","history_tags","prompt_text"]
log(df[cols].isna().mean().sort_values(ascending=False).to_string())

log("\nExample prompt")
log(df.iloc[0]["prompt_text"])
log("\nExample output")
log(df.iloc[0]["output_text"])

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: {OUT_PATH}")
print(f"Saved report: {REPORT_PATH}")
