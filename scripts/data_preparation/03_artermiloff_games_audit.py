import sys
import pandas as pd
import duckdb
from data_script_paths import DATA_AUDIT_REPORTS_DIR, RAW_DATA_DIR

RAW_DIR = RAW_DATA_DIR / "artermiloff_games"
REC_PATH = RAW_DATA_DIR / "game_recommendations" / "recommendations.csv"
REPORT_PATH = DATA_AUDIT_REPORTS_DIR / "03_artermiloff_games_audit_output.txt"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

log_file = open(REPORT_PATH, "w", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file

print("ARTERMILOFF FILES")
files = [p for p in RAW_DIR.rglob("*") if p.is_file()]
for p in files:
    print(f"{p} | {p.stat().st_size / 1024**2:.2f} MB")

csv_files = [p for p in files if p.suffix.lower() == ".csv"]
json_files = [p for p in files if p.suffix.lower() == ".json"]
parquet_files = [p for p in files if p.suffix.lower() == ".parquet"]

print("\nSAMPLE AUDIT")
for path in csv_files:
    print("\n" + "=" * 100)
    print(path)
    df = pd.read_csv(path, nrows=10000, low_memory=False)
    print("sample_shape:", df.shape)
    print("columns:", df.columns.tolist())
    print("dtypes:")
    print(df.dtypes)
    print("missing_share_top:")
    print(df.isna().mean().sort_values(ascending=False).head(30))
    print("head:")
    print(df.head(5))

for path in json_files:
    print("\n" + "=" * 100)
    print(path)
    df = pd.read_json(path, lines=True, nrows=10000)
    print("sample_shape:", df.shape)
    print("columns:", df.columns.tolist())
    print("dtypes:")
    print(df.dtypes)
    print("missing_share_top:")
    print(df.isna().mean().sort_values(ascending=False).head(30))
    print("head:")
    print(df.head(5))

for path in parquet_files:
    print("\n" + "=" * 100)
    print(path)
    df = pd.read_parquet(path)
    print("shape:", df.shape)
    print("columns:", df.columns.tolist())
    print("dtypes:")
    print(df.dtypes)
    print("missing_share_top:")
    print(df.isna().mean().sort_values(ascending=False).head(30))
    print("head:")
    print(df.head(5))

print("\nJOIN COVERAGE AUDIT")
main_csv = max(csv_files, key=lambda p: p.stat().st_size)
print("main_metadata_file:", main_csv)

sample = pd.read_csv(main_csv, nrows=1000, low_memory=False)
cols = sample.columns.tolist()
print("detected_columns:", cols)

app_id_candidates = ["app_id", "AppID", "appid", "steam_appid", "id"]
app_col = next((c for c in app_id_candidates if c in cols), None)
print("detected_app_id_column:", app_col)

if app_col is None:
    print("No app_id-like column found. Need manual inspection.")
else:
    con = duckdb.connect()
    rec_counts = con.execute(f"""
        SELECT app_id, COUNT(*) AS interaction_count
        FROM read_csv_auto('{REC_PATH.as_posix()}', header=true, sample_size=100000)
        GROUP BY app_id
    """).fetchdf()
    total_games = rec_counts["app_id"].nunique()
    total_interactions = rec_counts["interaction_count"].sum()

    meta_ids = con.execute(f"""
        SELECT DISTINCT TRY_CAST("{app_col}" AS BIGINT) AS app_id
        FROM read_csv_auto('{main_csv.as_posix()}', header=true, sample_size=100000)
        WHERE TRY_CAST("{app_col}" AS BIGINT) IS NOT NULL
    """).fetchdf()

    covered = rec_counts["app_id"].isin(set(meta_ids["app_id"]))
    covered_games = int(covered.sum())
    covered_interactions = int(rec_counts.loc[covered, "interaction_count"].sum())

    print("recommendation_games:", total_games)
    print("recommendation_interactions:", total_interactions)
    print("artermiloff_unique_games:", meta_ids["app_id"].nunique())
    print("covered_games:", covered_games, "/", total_games, round(100 * covered_games / total_games, 2), "%")
    print("covered_interactions:", covered_interactions, "/", total_interactions, round(100 * covered_interactions / total_interactions, 2), "%")

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()
print(f"\nSaved output to: {REPORT_PATH}")
