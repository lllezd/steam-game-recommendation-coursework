import json
import pandas as pd
from data_script_paths import DATA_AUDIT_REPORTS_DIR, FINAL_DATA_DIR, PROCESSED_DATA_DIR

IN_PATH = PROCESSED_DATA_DIR / "user_histories_mvp_v3.parquet"
OUT_DIR = FINAL_DATA_DIR / "instruction"
REPORT_PATH = DATA_AUDIT_REPORTS_DIR / "09_instruction_jsonl_report.txt"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

lines = []
def log(x=""):
    lines.append(str(x))

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

log("Loading dataset")
df = pd.read_parquet(IN_PATH)
log(f"input_path: {IN_PATH}")
log(f"rows: {len(df)}")
log(f"columns: {df.shape[1]}")

required_cols = ["sample_id","prompt_text","output_text_strict","label_strict","output_text_hybrid","output_text_hours_relative","target_app_id","target_title","split"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

log("\nSplit counts")
log(df["split"].value_counts().sort_index().to_string())

log("\nStrict label counts")
log(df["output_text_strict"].value_counts().to_string())

log("\nSplit x strict label")
log(pd.crosstab(df["split"], df["output_text_strict"]).to_string())

for split in ["train","val","test"]:
    part = df[df["split"] == split].copy()
    out_path = OUT_DIR / f"{split}.jsonl"
    write_jsonl(part, out_path)
    log(f"\nSaved {split}: {out_path}")
    log(f"rows: {len(part)}")
    log(f"size_mb: {out_path.stat().st_size / 1024**2:.2f}")

example_row = df.iloc[0]
instruction, input_text = split_prompt(example_row["prompt_text"])
log("\nExample json record")
example = {
    "sample_id": example_row["sample_id"],
    "instruction": instruction,
    "input": input_text,
    "output": example_row["output_text_strict"],
    "label_strict": bool(example_row["label_strict"]),
    "output_text_hybrid": example_row["output_text_hybrid"],
    "output_text_hours_relative": example_row["output_text_hours_relative"],
    "target_app_id": int(example_row["target_app_id"]),
    "target_title": example_row["target_title"],
    "split": example_row["split"]
}
log(json.dumps(example, ensure_ascii=False, indent=2))

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved jsonl files to: {OUT_DIR}")
print(f"Saved report: {REPORT_PATH}")
