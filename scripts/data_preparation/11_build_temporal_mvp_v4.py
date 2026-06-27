import ast, json, re
import numpy as np
import pandas as pd
import duckdb
from data_script_paths import DATA_AUDIT_REPORTS_DIR, FINAL_DATA_DIR, PROCESSED_DATA_DIR, TMP_DATA_DIR

INTERACTIONS_PATH = PROCESSED_DATA_DIR / "cleaned_interactions.parquet"
ITEMS_PATH = PROCESSED_DATA_DIR / "item_metadata.parquet"
OUT_PARQUET = PROCESSED_DATA_DIR / "user_histories_mvp_v4_temporal.parquet"
OUT_INSTR = FINAL_DATA_DIR / "instruction_temporal"
OUT_TAB = FINAL_DATA_DIR / "tabular_temporal"
REPORT_PATH = DATA_AUDIT_REPORTS_DIR / "11_temporal_mvp_v4_report.txt"
TMP_PATH = TMP_DATA_DIR / "user_histories_temporal_candidates.parquet"
for p in [OUT_PARQUET.parent, OUT_INSTR, OUT_TAB, REPORT_PATH.parent, TMP_PATH.parent]:
    p.mkdir(parents=True, exist_ok=True)

MIN_HISTORY_LEN = 4
MAX_HISTORY_LEN = 10
QUOTAS = {"train": 40000, "val": 5000, "test": 5000}
HYBRID_MIN_HOURS = 2.0
SEED_SALT = 777
MAX_HISTORY_TAGS = 8
MAX_TARGET_TAGS = 14
MAX_TARGET_GENRES = 8
MAX_TARGET_CATEGORIES = 8
MAX_DESCRIPTION_CHARS = 500

SEMANTIC_KEYWORDS = ["horror","survival","rpg","strategy","souls-like","soulslike","open world","co-op","coop","multiplayer","simulation","racing","sports","puzzle","fps","shooter","roguelike","rogue-like","story rich","sandbox","management","turn-based","tactical","city builder","platformer","metroidvania","mmo","mmorpg","stealth","fighting","rhythm","deckbuilding","card game","visual novel","crafting","base building","exploration","psychological","detective","mystery","tower defense","battle royale","pvp","pve","online co-op"]
GENERIC_KEYWORDS = ["action","adventure","indie","early access","singleplayer","single-player","free to play","casual","2d","3d","first-person","third person","third-person","realistic","stylized"]
JUNK_KEYWORDS = ["steam achievements","steam trading cards","steam cloud","steam leaderboard","leaderboards","controller","partial controller support","full controller support","remote play","family sharing","captions available","commentary available","stats","includes level editor","steam workshop","in-app purchases"]
JUNK_TOKENS = set(JUNK_KEYWORDS)

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
        t = clean_text(v).strip(" '\"[]")
        key = t.lower()
        if t and key not in seen:
            out.append(t)
            seen.add(key)
    return out

def token_score(token):
    t = token.lower()
    if any(j in t for j in JUNK_KEYWORDS):
        return (3, 999)
    for i, k in enumerate(SEMANTIC_KEYWORDS):
        if k in t:
            return (0, i)
    for i, k in enumerate(GENERIC_KEYWORDS):
        if k in t:
            return (1, i)
    return (2, 0)

def select_tags(*fields, max_items=8):
    tokens, seen = [], set()
    for field in fields:
        for t in parse_tokens(field):
            key = t.lower()
            if key not in seen:
                tokens.append(t)
                seen.add(key)
    useful = [t for t in tokens if token_score(t)[0] < 3]
    useful = sorted(enumerate(useful), key=lambda x: (token_score(x[1]), x[0]))
    return [t for _, t in useful[:max_items]]

def tags_text(*fields, max_items=8):
    return ", ".join(select_tags(*fields, max_items=max_items))

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
        tag_text = tags_text(tags[i] if i < len(tags) else "", genres[i] if i < len(genres) else "", max_items=MAX_HISTORY_TAGS)
        h = hours[i] if i < len(hours) else None
        item = title
        if tag_text:
            item += f" - tags: {tag_text}"
        if h is not None and not pd.isna(h):
            item += f"; {float(h):.1f} hours"
        if i < len(labels) and bool(labels[i]):
            liked.append(item)
        else:
            disliked.append(item)
    liked_text = "\n".join(f"{i+1}. {x}" for i, x in enumerate(liked)) if liked else "None"
    disliked_text = "\n".join(f"{i+1}. {x}" for i, x in enumerate(disliked)) if disliked else "None"
    target_title = clean_text(row["target_title"], 150)
    target_genres = tags_text(row["target_genres"], max_items=MAX_TARGET_GENRES)
    target_categories = tags_text(row["target_categories"], max_items=MAX_TARGET_CATEGORIES)
    target_tags = tags_text(row["target_tags"], row["target_genres"], max_items=MAX_TARGET_TAGS)
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
    return "Instruction:\nGiven the user's Steam game history, determine whether the user will recommend the target game. Answer only Yes or No.\n\nInput:\nUser's liked games:\n" + liked_text + "\n\nUser's disliked games:\n" + disliked_text + "\n\nTarget game:\n" + "\n".join(target_lines)

def split_prompt(prompt_text):
    instruction_marker = "Instruction:\n"
    input_marker = "\n\nInput:\n"
    if prompt_text.startswith(instruction_marker) and input_marker in prompt_text:
        body = prompt_text[len(instruction_marker):]
        instruction, input_text = body.split(input_marker, 1)
        return instruction.strip(), input_text.strip()
    return "Given the user's Steam game history, determine whether the user will recommend the target game. Answer only Yes or No.", str(prompt_text).strip()

def write_jsonl(df, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            instruction, input_text = split_prompt(row["prompt_text"])
            rec = {
                "sample_id": row["sample_id"],
                "instruction": instruction,
                "input": input_text,
                "output": row["output_text_strict"],
                "label_strict": bool(row["label_strict"]),
                "output_text_hybrid": row["output_text_hybrid"],
                "output_text_hours_relative": row["output_text_hours_relative"],
                "target_app_id": int(row["target_app_id"]),
                "target_title": row["target_title"],
                "split": row["split"]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def safe_float_list(x):
    out = []
    for v in as_list(x):
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

def build_tabular_features(row):
    labels = [bool(v) for v in as_list(row["history_labels"])]
    hours = safe_float_list(row["history_hours"])
    h = np.array(hours, dtype=float) if hours else np.array([], dtype=float)
    liked_count = sum(labels)
    disliked_count = len(labels) - liked_count
    liked_hours = np.array([hours[i] for i, v in enumerate(labels) if v and i < len(hours)], dtype=float)
    disliked_hours = np.array([hours[i] for i, v in enumerate(labels) if (not v) and i < len(hours)], dtype=float)
    liked_tokens, disliked_tokens = set(), set()
    history_tags = as_list(row["history_tags"])
    history_genres = as_list(row["history_genres"])
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

con = duckdb.connect()
con.execute("PRAGMA threads=4")
log("Building full candidate pool")
log(f"min_history_len: {MIN_HISTORY_LEN}")
log(f"max_history_len: {MAX_HISTORY_LEN}")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE candidate_pool AS
WITH ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY date DESC, review_id DESC) AS reverse_idx
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
)
SELECT * FROM grouped
""")

candidate_stats = con.execute("""
SELECT COUNT(*) AS candidates, SUM(CASE WHEN label THEN 1 ELSE 0 END) AS yes, SUM(CASE WHEN NOT label THEN 1 ELSE 0 END) AS no, MIN(target_date) AS min_date, MAX(target_date) AS max_date
FROM candidate_pool
""").fetchdf()
log("\nCandidate pool stats")
log(candidate_stats.to_string(index=False))

bounds = con.execute("""
WITH date_counts AS (
    SELECT target_date, COUNT(*) AS cnt
    FROM candidate_pool
    GROUP BY target_date
),
cum AS (
    SELECT target_date, cnt, SUM(cnt) OVER (ORDER BY target_date) AS cum_cnt, SUM(cnt) OVER () AS total_cnt
    FROM date_counts
)
SELECT
    MIN(CASE WHEN cum_cnt >= 0.8 * total_cnt THEN target_date END) AS val_start,
    MIN(CASE WHEN cum_cnt >= 0.9 * total_cnt THEN target_date END) AS test_start
FROM cum
""").fetchdf()
val_start = str(bounds.loc[0, "val_start"])
test_start = str(bounds.loc[0, "test_start"])
log("\nTemporal boundaries")
log(f"val_start: {val_start}")
log(f"test_start: {test_start}")

availability = con.execute(f"""
WITH temporal AS (
    SELECT *,
        CASE
            WHEN target_date < DATE '{val_start}' THEN 'train'
            WHEN target_date < DATE '{test_start}' THEN 'val'
            ELSE 'test'
        END AS split
    FROM candidate_pool
)
SELECT split, label, COUNT(*) AS cnt
FROM temporal
GROUP BY split, label
ORDER BY split, label
""").fetchdf()
log("\nCandidate availability by temporal split and label")
log(availability.to_string(index=False))

for split, quota in QUOTAS.items():
    for label_value in [False, True]:
        available = availability[(availability["split"] == split) & (availability["label"] == label_value)]["cnt"]
        available = int(available.iloc[0]) if len(available) else 0
        if available < quota:
            raise ValueError(f"Not enough candidates for split={split}, label={label_value}: available={available}, quota={quota}")

log("\nWriting sampled enriched dataset")
con.execute(f"""
COPY (
WITH temporal AS (
    SELECT *,
        CASE
            WHEN target_date < DATE '{val_start}' THEN 'train'
            WHEN target_date < DATE '{test_start}' THEN 'val'
            ELSE 'test'
        END AS split
    FROM candidate_pool
),
ranked_sample AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY split, label ORDER BY hash(user_id + {SEED_SALT})) AS sample_rank
    FROM temporal
),
sampled AS (
    SELECT *
    FROM ranked_sample
    WHERE
        (split = 'train' AND sample_rank <= {QUOTAS['train']})
        OR (split = 'val' AND sample_rank <= {QUOTAS['val']})
        OR (split = 'test' AND sample_rank <= {QUOTAS['test']})
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
FROM sampled s
LEFT JOIN read_parquet('{ITEMS_PATH.as_posix()}') t ON s.target_app_id = t.app_id
LEFT JOIN UNNEST(s.history_app_ids) WITH ORDINALITY AS u(hist_app_id, pos) ON TRUE
LEFT JOIN read_parquet('{ITEMS_PATH.as_posix()}') hm ON u.hist_app_id = hm.app_id
GROUP BY
    s.user_id, s.target_app_id, s.target_date, s.label, s.target_hours_aux,
    s.target_review_id, s.history_len, s.history_app_ids, s.history_dates,
    s.history_labels, s.history_hours, t.game_title, t.tags, t.genres,
    t.categories, t.description, t.item_text_prompt, s.split
ORDER BY split, label, target_date, user_id
) TO '{TMP_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

log("Loading sampled dataset")
df = pd.read_parquet(TMP_PATH)
df.insert(0, "sample_id", [f"steam_v4_{i:06d}" for i in range(len(df))])

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
df["prompt_text"] = df.apply(build_prompt, axis=1)
df.to_parquet(OUT_PARQUET, index=False, compression="zstd")

log("\nSaved v4 parquet")
log(f"path: {OUT_PARQUET}")
log(f"rows: {len(df)}")
log(f"columns: {df.shape[1]}")

log("\nSplit counts")
log(df["split"].value_counts().sort_index().to_string())
log("\nSplit x strict label")
log(pd.crosstab(df["split"], df["output_text_strict"]).to_string())
log("\nTarget date range by split")
log(df.groupby("split")["target_date"].agg(["min","max"]).to_string())
log("\nHybrid label balance")
log(pd.crosstab(df["split"], df["output_text_hybrid"]).to_string())
log("\nHours-relative label balance")
log(pd.crosstab(df["split"], df["output_text_hours_relative"]).to_string())
log("\nLow-hours positive / high-hours negative")
log(f"low_hours_positive_count: {int(df['is_low_hours_positive'].sum())}")
log(f"high_hours_negative_count: {int(df['is_high_hours_negative'].sum())}")
log("\nExample prompt")
log(df.iloc[0]["prompt_text"])

log("\nExporting instruction jsonl")
for split in ["train", "val", "test"]:
    part = df[df["split"] == split].copy()
    out_path = OUT_INSTR / f"{split}.jsonl"
    write_jsonl(part, out_path)
    log(f"{split}.jsonl: rows={len(part)}, size_mb={out_path.stat().st_size / 1024**2:.2f}")

log("\nBuilding tabular temporal dataset")
features = df.apply(build_tabular_features, axis=1)
df_tab = pd.concat([df, features], axis=1)
items = pd.read_parquet(ITEMS_PATH)
item_cols = ["app_id","has_artermiloff_metadata","release_date","price","rating","positive_ratio","user_reviews","price_final","price_original","discount","steam_deck","required_age","dlc_count","estimated_owners","average_playtime_forever","average_playtime_2weeks","median_playtime_forever","median_playtime_2weeks","peak_ccu","positive","negative","recommendations","pct_pos_total","num_reviews_total","pct_pos_recent","num_reviews_recent"]
items = items[[c for c in item_cols if c in items.columns]].copy().rename(columns={"app_id":"target_app_id"})
if "estimated_owners" in items.columns:
    owners = items["estimated_owners"].apply(parse_estimated_owners)
else:
    owners = pd.DataFrame([[np.nan, np.nan, np.nan]] * len(items))
owners.columns = ["estimated_owners_min","estimated_owners_max","estimated_owners_mid"]
items = pd.concat([items, owners], axis=1)
if "release_date" in items.columns:
    items["release_year"] = pd.to_datetime(items["release_date"], errors="coerce").dt.year
df_tab = df_tab.merge(items, on="target_app_id", how="left")

id_cols = ["sample_id","user_id","target_app_id","split"]
label_cols = ["label_strict","output_text_strict","label_hybrid","output_text_hybrid","label_hours_relative","output_text_hours_relative"]
text_cols = ["target_title","target_tags","target_genres","target_categories","target_description","target_item_text_prompt"]
feature_cols = ["history_len","history_positive_count","history_negative_count","history_positive_share","history_total_hours","history_mean_hours","history_median_hours","history_max_hours","history_min_hours","history_liked_mean_hours","history_disliked_mean_hours","target_token_count","liked_token_count","disliked_token_count","target_liked_overlap_count","target_disliked_overlap_count","target_liked_jaccard","target_disliked_jaccard","target_jaccard_diff","target_description_len","target_title_len","has_artermiloff_metadata","price","positive_ratio","user_reviews","price_final","price_original","discount","steam_deck","required_age","dlc_count","average_playtime_forever","average_playtime_2weeks","median_playtime_forever","median_playtime_2weeks","peak_ccu","positive","negative","recommendations","pct_pos_total","num_reviews_total","pct_pos_recent","num_reviews_recent","estimated_owners_min","estimated_owners_max","estimated_owners_mid","release_year"]
select_cols = [c for c in id_cols + label_cols + text_cols + feature_cols if c in df_tab.columns]
tab = df_tab[select_cols].copy()
for split in ["train", "val", "test"]:
    part = tab[tab["split"] == split].copy()
    out_path = OUT_TAB / f"{split}_tabular.parquet"
    part.to_parquet(out_path, index=False, compression="zstd")
    log(f"{split}_tabular.parquet: rows={len(part)}, size_mb={out_path.stat().st_size / 1024**2:.2f}")

log("\nTabular columns")
log("\n".join(tab.columns))
REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved v4 parquet: {OUT_PARQUET}")
print(f"Saved instruction files: {OUT_INSTR}")
print(f"Saved tabular files: {OUT_TAB}")
print(f"Saved report: {REPORT_PATH}")
