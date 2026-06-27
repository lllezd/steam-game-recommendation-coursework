import pandas as pd
import duckdb
from data_script_paths import DATA_AUDIT_REPORTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

BASE_GAMES_PATH = RAW_DATA_DIR / "game_recommendations" / "games.csv"
BASE_META_PATH = RAW_DATA_DIR / "game_recommendations" / "games_metadata.json"
REC_PATH = RAW_DATA_DIR / "game_recommendations" / "recommendations.csv"
ARTER_PATH = RAW_DATA_DIR / "artermiloff_games" / "games_march2025_cleaned.csv"
OUT_PATH = PROCESSED_DATA_DIR / "item_metadata.parquet"
REPORT_PATH = DATA_AUDIT_REPORTS_DIR / "04_item_metadata_report.txt"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

lines = []
def log(x=""):
    lines.append(str(x))

def normalize_text(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).replace("\n", " ").replace("\r", " ").strip()
    return x if x else pd.NA

def list_to_text(x):
    if isinstance(x, list):
        return ", ".join(str(v).strip() for v in x if str(v).strip())
    return normalize_text(x)

log("Loading recommendation app_id counts...")
con = duckdb.connect()
rec_counts = con.execute(f"""
SELECT app_id, COUNT(*) AS interaction_count, COUNT(DISTINCT user_id) AS user_count
FROM read_csv_auto('{REC_PATH.as_posix()}', header=true, sample_size=100000)
GROUP BY app_id
""").fetchdf()
log(f"recommendation_games: {rec_counts['app_id'].nunique()}")
log(f"recommendation_interactions: {int(rec_counts['interaction_count'].sum())}")

log("\nLoading base metadata...")
base_games = pd.read_csv(BASE_GAMES_PATH)
base_meta = pd.read_json(BASE_META_PATH, lines=True)
base_games = base_games.drop_duplicates("app_id", keep="first")
base_meta = base_meta.drop_duplicates("app_id", keep="first")
base_meta["base_tags"] = base_meta["tags"].apply(list_to_text)
base_meta["base_description"] = base_meta["description"].apply(normalize_text)
base_games = base_games.rename(columns={"title":"base_title"})
base_games["base_title"] = base_games["base_title"].apply(normalize_text)

log("\nLoading artermiloff metadata...")
arter_cols = [
    "appid","name","release_date","price","required_age","dlc_count",
    "detailed_description","about_the_game","short_description",
    "developers","publishers","categories","genres","tags",
    "estimated_owners","average_playtime_forever","average_playtime_2weeks",
    "median_playtime_forever","median_playtime_2weeks","peak_ccu",
    "positive","negative","recommendations","pct_pos_total",
    "num_reviews_total","pct_pos_recent","num_reviews_recent"
]
arter = pd.read_csv(ARTER_PATH, usecols=arter_cols, low_memory=False)
arter = arter.rename(columns={
    "appid":"app_id",
    "name":"arter_title",
    "release_date":"arter_release_date",
    "price":"arter_price",
    "tags":"arter_tags",
    "detailed_description":"arter_detailed_description",
    "about_the_game":"arter_about_the_game",
    "short_description":"arter_short_description"
})
arter["app_id"] = pd.to_numeric(arter["app_id"], errors="coerce")
arter = arter.dropna(subset=["app_id"]).copy()
arter["app_id"] = arter["app_id"].astype("int64")
arter = arter.drop_duplicates("app_id", keep="first")
for col in ["arter_title","arter_tags","arter_detailed_description","arter_about_the_game","arter_short_description","developers","publishers","categories","genres","estimated_owners"]:
    arter[col] = arter[col].apply(normalize_text)

log(f"base_games_unique: {base_games['app_id'].nunique()}")
log(f"base_meta_unique: {base_meta['app_id'].nunique()}")
log(f"artermiloff_unique: {arter['app_id'].nunique()}")

log("\nBuilding item metadata...")
items = rec_counts.merge(base_games, on="app_id", how="left")
items = items.merge(base_meta[["app_id","base_description","base_tags"]], on="app_id", how="left")
items = items.merge(arter, on="app_id", how="left")
items["has_artermiloff_metadata"] = items["arter_title"].notna()
items["game_title"] = items["base_title"].fillna(items["arter_title"])
items["description"] = items["base_description"].fillna(items["arter_short_description"]).fillna(items["arter_about_the_game"]).fillna(items["arter_detailed_description"])
items["tags"] = items["base_tags"].fillna(items["arter_tags"])
items["release_date"] = items["date_release"].fillna(items["arter_release_date"])
items["price"] = items["price_final"].fillna(items["arter_price"])
items["item_text_prompt"] = (
    items["game_title"].fillna("") + " " +
    items["tags"].fillna("") + " " +
    items["genres"].fillna("") + " " +
    items["categories"].fillna("") + " " +
    items["description"].fillna("")
).str.replace(r"\s+", " ", regex=True).str.strip()
items["item_text_retrieval"] = (
    items["game_title"].fillna("") + " " +
    items["developers"].fillna("") + " " +
    items["publishers"].fillna("") + " " +
    items["tags"].fillna("") + " " +
    items["genres"].fillna("") + " " +
    items["categories"].fillna("") + " " +
    items["description"].fillna("")
).str.replace(r"\s+", " ", regex=True).str.strip()

final_cols = [
    "app_id","interaction_count","user_count","has_artermiloff_metadata",
    "game_title","release_date","price","rating","positive_ratio","user_reviews",
    "price_final","price_original","discount","steam_deck",
    "description","tags","genres","categories","developers","publishers",
    "required_age","dlc_count","estimated_owners","average_playtime_forever",
    "average_playtime_2weeks","median_playtime_forever","median_playtime_2weeks",
    "peak_ccu","positive","negative","recommendations","pct_pos_total",
    "num_reviews_total","pct_pos_recent","num_reviews_recent",
    "item_text_prompt","item_text_retrieval"
]
items = items[final_cols]

log("\nCoverage")
total_games = len(items)
total_interactions = items["interaction_count"].sum()
arter_games = items["has_artermiloff_metadata"].sum()
arter_interactions = items.loc[items["has_artermiloff_metadata"], "interaction_count"].sum()
log(f"items_total: {total_games}")
log(f"interactions_total: {int(total_interactions)}")
log(f"artermiloff_games_covered: {arter_games}/{total_games} ({100 * arter_games / total_games:.2f}%)")
log(f"artermiloff_interactions_covered: {int(arter_interactions)}/{int(total_interactions)} ({100 * arter_interactions / total_interactions:.2f}%)")

log("\nMissing share by games")
log(items.isna().mean().sort_values(ascending=False).to_string())

log("\nInteraction-weighted missing share")
for col in items.columns:
    missing_interactions = items.loc[items[col].isna(), "interaction_count"].sum()
    log(f"{col}: {100 * missing_interactions / total_interactions:.2f}%")

log("\nTop games without artermiloff metadata")
log(items.loc[~items["has_artermiloff_metadata"], ["app_id","game_title","interaction_count","user_count"]].sort_values("interaction_count", ascending=False).head(20).to_string(index=False))

items.to_parquet(OUT_PATH, index=False)
REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: {OUT_PATH}")
print(f"Saved report: {REPORT_PATH}")
