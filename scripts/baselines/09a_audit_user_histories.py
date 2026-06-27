from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH = PROJECT_ROOT / "data" / "processed" / "user_histories_mvp_v4_temporal.parquet"

def main():
    df = pd.read_parquet(PATH)
    print("Shape:")
    print(df.shape)
    print("\nColumns:")
    for col in df.columns:
        print(col)
    print("\nDtypes:")
    print(df.dtypes)
    print("\nFirst row:")
    row = df.iloc[0]
    for col in df.columns:
        value = row[col]
        text = str(value)
        if len(text) > 500:
            text = text[:500] + "..."
        print(f"\n{col}:")
        print(text)

if __name__ == "__main__":
    main()