import ast
import numpy as np
import pandas as pd
from data_script_paths import DATA_AUDIT_REPORTS_DIR, PROCESSED_DATA_DIR

IN_PATH = PROCESSED_DATA_DIR / "user_histories_mvp_v2.parquet"
OUT_PATH = PROCESSED_DATA_DIR / "user_histories_mvp_v3.parquet"
REPORT_PATH = DATA_AUDIT_REPORTS_DIR / "08_user_histories_v3_report.txt"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_HISTORY_TAGS = 8
MAX_TARGET_TAGS = 14
MAX_TARGET_GENRES = 8
MAX_TARGET_CATEGORIES = 8
MAX_DESCRIPTION_CHARS = 500

SEMANTIC_KEYWORDS = [
    "horror","survival","rpg","strategy","souls-like","soulslike","open world","co-op","coop","multiplayer",
    "simulation","racing","sports","puzzle","fps","shooter","roguelike","rogue-like","story rich","sandbox",
    "management","turn-based","tactical","city builder","platformer","metroidvania","mmo","mmorpg","stealth",
    "fighting","rhythm","deckbuilding","card game","visual novel","crafting","base building","exploration",
    "psychological","detective","mystery","tower defense","battle royale","pvp","pve","online co-op"
]
GENERIC_KEYWORDS = [
    "action","adventure","indie","early access","singleplayer","single-player","free to play","casual",
    "2d","3d","first-person","third person","third-person","realistic","stylized"
]
JUNK_KEYWORDS = [
    "steam achievements","steam trading cards","steam cloud","steam leaderboard","leaderboards","controller",
    "partial controller support","full controller support","remote play","family sharing","captions available",
    "commentary available","stats","includes level editor","steam workshop","in-app purchases"
]

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

log("Loading v2")
df = pd.read_parquet(IN_PATH)
log(f"input_rows: {len(df)}")
log(f"input_columns: {df.shape[1]}")

if "prompt_text_v2" not in df.columns:
    df = df.rename(columns={"prompt_text":"prompt_text_v2"})
else:
    df = df.drop(columns=["prompt_text"], errors="ignore")

log("Building v3 prompt_text")
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

log("\nExample prompt v3")
log(df.iloc[0]["prompt_text"])

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: {OUT_PATH}")
print(f"Saved report: {REPORT_PATH}")
