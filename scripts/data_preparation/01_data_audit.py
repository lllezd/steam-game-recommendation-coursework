import pandas as pd
import duckdb
import sys
from data_script_paths import DATA_AUDIT_REPORTS_DIR, RAW_DATA_DIR

LOG_PATH = DATA_AUDIT_REPORTS_DIR / "01_raw_audit_output.txt"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
log_file = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file

raw = RAW_DATA_DIR
rec_path = raw / "game_recommendations" / "recommendations.csv"
games_path = raw / "game_recommendations" / "games.csv"
meta_path = raw / "game_recommendations" / "games_metadata.json"
fronkon_csv_path = raw / "fronkon_games" / "games.csv"

def show_file(path):
    print("\n" + "=" * 100)
    print(path)
    print(f"size_mb: {path.stat().st_size / 1024 ** 2:.2f}")

def audit_csv_sample(path, nrows=10000):
    show_file(path)
    df = pd.read_csv(path, nrows=nrows)
    print("sample_shape:", df.shape)
    print("columns:", df.columns.tolist())
    print("dtypes:")
    print(df.dtypes)
    print("missing_share_top:")
    print(df.isna().mean().sort_values(ascending=False).head(20))
    print("head:")
    print(df.head(5))

def audit_jsonl_sample(path, nrows=10000):
    show_file(path)
    df = pd.read_json(path, lines=True, nrows=nrows)
    print("sample_shape:", df.shape)
    print("columns:", df.columns.tolist())
    print("dtypes:")
    print(df.dtypes)
    print("missing_share_top:")
    print(df.isna().mean().sort_values(ascending=False).head(20))
    print("head:")
    print(df.head(5))

print("RAW FILES")
for path in raw.rglob("*"):
    if path.is_file():
        print(f"{path} | {path.stat().st_size / 1024 ** 2:.2f} MB")

print("\nSAMPLE AUDIT")
audit_csv_sample(rec_path)
audit_csv_sample(games_path)
audit_jsonl_sample(meta_path)
audit_csv_sample(fronkon_csv_path)

print("\nFULL RECOMMENDATIONS AUDIT")
con = duckdb.connect()
con.execute(f"""
CREATE VIEW recommendations AS
SELECT * FROM read_csv_auto('{rec_path.as_posix()}', header=true, sample_size=100000)
""")
queries = {
    "rows": "SELECT COUNT(*) AS rows FROM recommendations",
    "columns": "DESCRIBE recommendations",
    "unique_users": "SELECT COUNT(DISTINCT user_id) AS unique_users FROM recommendations",
    "unique_games": "SELECT COUNT(DISTINCT app_id) AS unique_games FROM recommendations",
    "label_balance": "SELECT is_recommended, COUNT(*) AS cnt FROM recommendations GROUP BY is_recommended ORDER BY is_recommended",
    "date_range": "SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM recommendations",
    "hours_stats": "SELECT MIN(hours) AS min_hours, approx_quantile(hours, 0.5) AS median_hours, approx_quantile(hours, 0.95) AS p95_hours, approx_quantile(hours, 0.99) AS p99_hours, MAX(hours) AS max_hours FROM recommendations",
    "interactions_per_user": "SELECT approx_quantile(cnt, 0.5) AS median_cnt, approx_quantile(cnt, 0.9) AS p90_cnt, approx_quantile(cnt, 0.99) AS p99_cnt, MAX(cnt) AS max_cnt FROM (SELECT user_id, COUNT(*) AS cnt FROM recommendations GROUP BY user_id)",
    "duplicate_user_game": "SELECT COUNT(*) AS duplicated_rows FROM (SELECT user_id, app_id, COUNT(*) AS cnt FROM recommendations GROUP BY user_id, app_id HAVING COUNT(*) > 1)"
}
for name, query in queries.items():
    print("\n" + "-" * 100)
    print(name)
    print(con.execute(query).fetchdf())

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()
print(f"\nSaved output to: {LOG_PATH}")
