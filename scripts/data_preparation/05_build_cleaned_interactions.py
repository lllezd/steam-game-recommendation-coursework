import duckdb
from data_script_paths import DATA_AUDIT_REPORTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, TMP_DATA_DIR

REC_PATH = RAW_DATA_DIR / "game_recommendations" / "recommendations.csv"
OUT_PATH = PROCESSED_DATA_DIR / "cleaned_interactions.parquet"
REPORT_PATH = DATA_AUDIT_REPORTS_DIR / "05_cleaned_interactions_report.txt"
TMP_DIR = TMP_DATA_DIR / "duckdb"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

MIN_USER_INTERACTIONS = 5
lines = []
def log(x=""):
    lines.append(str(x))

con = duckdb.connect()
con.execute(f"PRAGMA temp_directory='{TMP_DIR.as_posix()}'")
con.execute("PRAGMA threads=4")

log("Building cleaned interactions...")
log(f"min_user_interactions: {MIN_USER_INTERACTIONS}")

con.execute(f"""
CREATE OR REPLACE TEMP VIEW raw_rec AS
SELECT
    CAST(user_id AS BIGINT) AS user_id,
    CAST(app_id AS BIGINT) AS app_id,
    CAST(date AS DATE) AS date,
    CAST(is_recommended AS BOOLEAN) AS is_recommended,
    CAST(hours AS DOUBLE) AS hours,
    CAST(review_id AS BIGINT) AS review_id,
    CAST(helpful AS BIGINT) AS helpful,
    CAST(funny AS BIGINT) AS funny
FROM read_csv_auto('{REC_PATH.as_posix()}', header=true, sample_size=100000)
WHERE user_id IS NOT NULL
  AND app_id IS NOT NULL
  AND date IS NOT NULL
  AND is_recommended IS NOT NULL
  AND hours IS NOT NULL
  AND hours >= 0
""")

raw_stats = con.execute("""
SELECT
    COUNT(*) AS rows,
    COUNT(DISTINCT user_id) AS users,
    COUNT(DISTINCT app_id) AS games,
    SUM(CASE WHEN is_recommended THEN 1 ELSE 0 END) AS positive,
    SUM(CASE WHEN NOT is_recommended THEN 1 ELSE 0 END) AS negative,
    MIN(date) AS min_date,
    MAX(date) AS max_date,
    MIN(hours) AS min_hours,
    approx_quantile(hours, 0.5) AS median_hours,
    approx_quantile(hours, 0.95) AS p95_hours,
    approx_quantile(hours, 0.99) AS p99_hours,
    MAX(hours) AS max_hours
FROM raw_rec
""").fetchdf()
log("\nRaw valid stats")
log(raw_stats.to_string(index=False))

duplicate_stats = con.execute("""
SELECT COUNT(*) AS duplicate_user_game_pairs
FROM (
    SELECT user_id, app_id, COUNT(*) AS cnt
    FROM raw_rec
    GROUP BY user_id, app_id
    HAVING COUNT(*) > 1
)
""").fetchdf()
log("\nDuplicate user-app pairs before dedup")
log(duplicate_stats.to_string(index=False))

con.execute("""
CREATE OR REPLACE TEMP VIEW dedup_rec AS
SELECT user_id, app_id, date, is_recommended, hours, review_id, helpful, funny
FROM (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY user_id, app_id
               ORDER BY date DESC, review_id DESC
           ) AS rn
    FROM raw_rec
)
WHERE rn = 1
""")

con.execute(f"""
CREATE OR REPLACE TEMP VIEW eligible_users AS
SELECT user_id, COUNT(*) AS user_interaction_count
FROM dedup_rec
GROUP BY user_id
HAVING COUNT(*) >= {MIN_USER_INTERACTIONS}
""")

log("\nWriting parquet...")
con.execute(f"""
COPY (
    SELECT
        r.user_id,
        r.app_id,
        r.date,
        r.is_recommended,
        r.hours,
        r.review_id,
        r.helpful,
        r.funny,
        u.user_interaction_count,
        ROW_NUMBER() OVER (
            PARTITION BY r.user_id
            ORDER BY r.date, r.review_id
        ) AS user_interaction_idx
    FROM dedup_rec r
    JOIN eligible_users u USING (user_id)
    ORDER BY r.user_id, r.date, r.review_id
) TO '{OUT_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

clean_stats = con.execute(f"""
SELECT
    COUNT(*) AS rows,
    COUNT(DISTINCT user_id) AS users,
    COUNT(DISTINCT app_id) AS games,
    SUM(CASE WHEN is_recommended THEN 1 ELSE 0 END) AS positive,
    SUM(CASE WHEN NOT is_recommended THEN 1 ELSE 0 END) AS negative,
    MIN(date) AS min_date,
    MAX(date) AS max_date,
    approx_quantile(user_interaction_count, 0.5) AS median_user_interactions,
    approx_quantile(user_interaction_count, 0.9) AS p90_user_interactions,
    approx_quantile(user_interaction_count, 0.99) AS p99_user_interactions,
    MAX(user_interaction_count) AS max_user_interactions,
    approx_quantile(hours, 0.5) AS median_hours,
    approx_quantile(hours, 0.95) AS p95_hours,
    approx_quantile(hours, 0.99) AS p99_hours
FROM read_parquet('{OUT_PATH.as_posix()}')
""").fetchdf()
log("\nCleaned stats")
log(clean_stats.to_string(index=False))

label_balance = con.execute(f"""
SELECT is_recommended, COUNT(*) AS cnt, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM read_parquet('{OUT_PATH.as_posix()}')
GROUP BY is_recommended
ORDER BY is_recommended
""").fetchdf()
log("\nLabel balance")
log(label_balance.to_string(index=False))

top_users = con.execute(f"""
SELECT user_interaction_count, COUNT(*) AS rows
FROM read_parquet('{OUT_PATH.as_posix()}')
GROUP BY user_interaction_count
ORDER BY user_interaction_count
LIMIT 20
""").fetchdf()
log("\nSmallest retained user_interaction_count values")
log(top_users.to_string(index=False))

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: {OUT_PATH}")
print(f"Saved report: {REPORT_PATH}")
