from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "user_histories_mvp_v4_temporal.parquet"
OUT_DIR = PROJECT_ROOT / "outputs" / "llm_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
METHOD = "qwen2_5_3b_zero_shot_pilot"
MAX_INPUT_TOKENS = 2048
LABEL_COL = "label_strict"
PROMPT_COL = "prompt_text"
N_PER_CLASS = 250
SEED = 42

def sample_split(df, split):
    part = df[df["split"] == split].copy()
    rows = []
    for label in [False, True]:
        g = part[part[LABEL_COL] == label]
        rows.append(g.sample(n=min(N_PER_CLASS, len(g)), random_state=SEED))
    return pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

def parse_yes_no(text):
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    matches = re.findall(r"\b(yes|no)\b", text)
    if not matches:
        return None
    if len(set(matches)) > 1:
        return None
    return 1 if matches[0] == "yes" else 0

def build_prompt(tokenizer, prompt_text):
    messages = [
        {"role": "system", "content": "You are a recommendation model. Answer with exactly one word: Yes or No."},
        {"role": "user", "content": prompt_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def one_token_score(model, tokenizer, prompt, device, yes_id, no_id):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[:, -1, :][0]
    pair_logits = torch.stack([logits[no_id], logits[yes_id]])
    probs = torch.softmax(pair_logits, dim=0)
    return float(probs[1].detach().cpu()), int(enc["input_ids"].shape[1])

def generate_answer(model, tokenizer, prompt, device):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def run_split(df, split, model, tokenizer, device, yes_id, no_id):
    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=split):
        prompt = build_prompt(tokenizer, getattr(row, PROMPT_COL))
        score, prompt_tokens = one_token_score(model, tokenizer, prompt, device, yes_id, no_id)
        answer = generate_answer(model, tokenizer, prompt, device)
        parsed = parse_yes_no(answer)
        rows.append({
            "sample_id": row.sample_id,
            "split": row.split,
            "label": int(getattr(row, LABEL_COL)),
            "score_yes": score,
            "generated_answer": answer,
            "parsed_answer": parsed,
            "parse_ok": parsed is not None,
            "prompt_tokens": prompt_tokens,
        })
    return pd.DataFrame(rows)

def choose_threshold(y_true, score):
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        pred = score >= threshold
        cur_f1 = f1_score(y_true, pred, zero_division=0)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = threshold
    return float(best_threshold)

def calc_metrics(y_true, score, threshold):
    pred = score >= threshold
    return {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }

def append_metrics(row):
    path = OUT_DIR / "llm_metrics.csv"
    df_new = pd.DataFrame([row])
    if path.exists():
        df_old = pd.read_csv(path)
        df_old = df_old[df_old["method"] != row["method"]]
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(path, index=False)

def save_threshold(method, threshold):
    path = OUT_DIR / "llm_thresholds.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[method] = threshold
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is not available.")

    df = pd.read_parquet(DATA_PATH)
    val = sample_split(df, "val")
    test = sample_split(df, "test")

    sample_ids = {
        "val": val["sample_id"].tolist(),
        "test": test["sample_id"].tolist(),
    }
    (OUT_DIR / "zero_shot_3b_pilot_sample_ids.json").write_text(json.dumps(sample_ids, ensure_ascii=False, indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.truncation_side = "left"
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(f"Yes ids: {yes_ids}, No ids: {no_ids}")
    yes_id, no_id = yes_ids[0], no_ids[0]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    val_pred = run_split(val, "val", model, tokenizer, device, yes_id, no_id)
    test_pred = run_split(test, "test", model, tokenizer, device, yes_id, no_id)

    val_path = OUT_DIR / "zero_shot_3b_pilot_predictions_val.csv"
    test_path = OUT_DIR / "zero_shot_3b_pilot_predictions_test.csv"
    val_pred.to_csv(val_path, index=False)
    test_pred.to_csv(test_path, index=False)

    y_val = val_pred["label"].values
    y_test = test_pred["label"].values
    val_score = val_pred["score_yes"].values
    test_score = test_pred["score_yes"].values

    threshold = choose_threshold(y_val, val_score)
    val_metrics = calc_metrics(y_val, val_score, threshold)
    test_metrics = calc_metrics(y_test, test_score, threshold)

    row = {
        "method": METHOD,
        "model": MODEL_NAME,
        "eval_type": "pilot_stratified",
        "val_size": len(val_pred),
        "test_size": len(test_pred),
        "val_parse_rate": float(val_pred["parse_ok"].mean()),
        "test_parse_rate": float(test_pred["parse_ok"].mean()),
        "threshold": threshold,
        "val_roc_auc": val_metrics["roc_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
    }

    append_metrics(row)
    save_threshold(METHOD, threshold)

    config = {
        "method": METHOD,
        "model_name": MODEL_NAME,
        "n_per_class": N_PER_CLASS,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "score_type": "P(Yes)/(P(Yes)+P(No)) from next-token logits",
        "threshold_selection": "maximize F1 on val",
        "threshold": threshold,
    }
    (OUT_DIR / "zero_shot_3b_pilot_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Method: {METHOD}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Val size: {len(val_pred)}")
    print(f"Test size: {len(test_pred)}")
    print(f"Val parse rate: {row['val_parse_rate']:.4f}")
    print(f"Test parse rate: {row['test_parse_rate']:.4f}")
    print(f"Selected threshold on val: {threshold:.6f}")
    print("\nVal metrics:")
    print(val_metrics)
    print("\nTest metrics:")
    print(test_metrics)
    print("\nVal score summary:")
    print(val_pred["score_yes"].describe())
    print("\nTest score summary:")
    print(test_pred["score_yes"].describe())
    print("\nTest preview:")
    print(test_pred[["sample_id", "label", "score_yes", "generated_answer", "parsed_answer", "prompt_tokens"]].head(20).to_string(index=False))
    print("\nSaved:")
    print(val_path)
    print(test_path)
    print(OUT_DIR / "llm_metrics.csv")
    print(OUT_DIR / "llm_thresholds.json")

if __name__ == "__main__":
    main()
