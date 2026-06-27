import pandas as pd
import duckdb
import sys
import numpy as np
from data_script_paths import DATA_AUDIT_REPORTS_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR

LOG_PATH = DATA_AUDIT_REPORTS_DIR / "02_metadata_join_audit_output.txt"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
log_file = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file

raw = RAW_DATA_DIR
rec_path = raw / "game_recommendations" / "recommendations.csv"
base_games_path = raw / "game_recommendations" / "games.csv"
base_meta_path = raw / "game_recommendations" / "games_metadata.json"
fronkon_path = raw / "fronkon_games" / "games.csv"
INTERIM_AUDIT_PATH = INTERIM_DATA_DIR / "item_metadata_join_audit.parquet"

def pct(x):
    return round(100 * x, 2)

print("Loading app_id counts from recommendations.csv...")
con = duckdb.connect()
rec_counts = con.execute(f"""
SELECT app_id, COUNT(*) AS interaction_count, COUNT(DISTINCT user_id) AS user_count
FROM read_csv_auto('{rec_path.as_posix()}', header=true, sample_size=100000)
GROUP BY app_id
""").fetchdf()
total_games = rec_counts["app_id"].nunique()
total_interactions = rec_counts["interaction_count"].sum()
print(f"recommendation_games: {total_games}")
print(f"recommendation_interactions: {total_interactions}")

print("\nLoading metadata files...")
base_games = pd.read_csv(base_games_path)
base_meta = pd.read_json(base_meta_path, lines=True)
fronkon_cols = ["AppID","Name","Release date","Price","About the game","Developers","Publishers","Categories","Genres","Tags","Positive","Negative","Recommendations"]
fronkon = pd.read_csv(fronkon_path, usecols=fronkon_cols, low_memory=False)
fronkon["app_id"] = pd.to_numeric(fronkon["AppID"], errors="coerce")
bad_app_id = fronkon["app_id"].isna() | ~np.isfinite(fronkon["app_id"])
print(f"fronkon bad AppID rows removed: {bad_app_id.sum()}")
fronkon = fronkon.loc[~bad_app_id].copy()
fronkon["app_id"] = fronkon["app_id"].astype("int64")

def dedup(df, name):
    before = len(df)
    df = df.drop_duplicates("app_id", keep="first")
    print(f"{name}: rows={before}, unique_app_id={df['app_id'].nunique()}, duplicates_removed={before-len(df)}")
    return df

print("\nMetadata app_id counts")
base_games = dedup(base_games, "base_games")
base_meta = dedup(base_meta, "base_meta")
fronkon = dedup(fronkon, "fronkon")

def coverage(df, name):
    ids = set(df["app_id"])
    mask = rec_counts["app_id"].isin(ids)
    games_covered = mask.sum()
    interactions_covered = rec_counts.loc[mask, "interaction_count"].sum()
    print(f"\n{name}")
    print(f"covered_games: {games_covered}/{total_games} ({pct(games_covered/total_games)}%)")
    print(f"covered_interactions: {interactions_covered}/{total_interactions} ({pct(interactions_covered/total_interactions)}%)")

coverage(base_games, "coverage: game_recommendations/games.csv")
coverage(base_meta, "coverage: game_recommendations/games_metadata.json")
coverage(fronkon, "coverage: fronkon_games/games.csv")

print("\nBuilding combined item metadata...")
base_games_small = base_games[["app_id","title","date_release","rating","positive_ratio","user_reviews","price_final","price_original","discount","steam_deck"]]
base_meta_small = base_meta[["app_id","description","tags"]].rename(columns={"description":"base_description","tags":"base_tags"})
fronkon_small = fronkon.rename(columns={
    "Name":"fronkon_title",
    "Release date":"fronkon_release_date",
    "Price":"fronkon_price",
    "About the game":"fronkon_description",
    "Developers":"developers",
    "Publishers":"publishers",
    "Categories":"categories",
    "Genres":"genres",
    "Tags":"fronkon_tags",
    "Positive":"fronkon_positive",
    "Negative":"fronkon_negative",
    "Recommendations":"fronkon_recommendations"
})[["app_id","fronkon_title","fronkon_release_date","fronkon_price","fronkon_description","developers","publishers","categories","genres","fronkon_tags","fronkon_positive","fronkon_negative","fronkon_recommendations"]]

items = rec_counts[["app_id","interaction_count","user_count"]].merge(base_games_small, on="app_id", how="left")
items = items.merge(base_meta_small, on="app_id", how="left")
items = items.merge(fronkon_small, on="app_id", how="left")
items["final_title"] = items["title"].fillna(items["fronkon_title"])
items["final_description"] = items["base_description"].fillna(items["fronkon_description"])
items["final_tags"] = items["base_tags"].astype("string").fillna(items["fronkon_tags"].astype("string"))
items["item_text"] = (
    items["final_title"].fillna("") + " " +
    items["genres"].fillna("") + " " +
    items["final_tags"].fillna("") + " " +
    items["final_description"].fillna("")
).str.strip()

print("\nCombined metadata missing share for games from recommendations.csv")
cols = ["final_title","final_description","final_tags","genres","categories","developers","publishers","price_final","date_release","item_text"]
print(items[cols].isna().mean().sort_values(ascending=False))

print("\nInteraction-weighted missing share")
for col in cols:
    missing_interactions = items.loc[items[col].isna(), "interaction_count"].sum()
    print(f"{col}: {pct(missing_interactions/total_interactions)}% interactions missing")

print("\nTop games without final_title")
print(items[items["final_title"].isna()].sort_values("interaction_count", ascending=False).head(10)[["app_id","interaction_count","user_count"]])

print("\nTop games without final_description")
print(items[items["final_description"].isna()].sort_values("interaction_count", ascending=False).head(10)[["app_id","interaction_count","user_count","final_title"]])

INTERIM_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
items.to_parquet(INTERIM_AUDIT_PATH, index=False)
print(f"\nSaved: {INTERIM_AUDIT_PATH}")


sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()
print(f"\nSaved output to: {LOG_PATH}")
