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

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LABEL_COL = "label_strict"
PROMPT_COL = "prompt_text"
SPLIT = "val"
N_PER_CLASS = 10
SEED = 42
MAX_INPUT_TOKENS = 4096

def sample_split(df):
    part = df[df["split"] == SPLIT].copy()
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

def one_token_score(model, tokenizer, prompt, device):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[:, -1, :][0]
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(f"Yes ids: {yes_ids}, No ids: {no_ids}")
    pair_logits = torch.stack([logits[no_ids[0]], logits[yes_ids[0]]])
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

def calc_metrics(y_true, score, pred):
    return {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is not available. Stop here and fix torch/CUDA first.")

    df = pd.read_parquet(DATA_PATH)
    sample = sample_split(df)
    sample_ids = sample["sample_id"].tolist()
    (OUT_DIR / "zero_shot_smoke_sample_ids.json").write_text(json.dumps(sample_ids, ensure_ascii=False, indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    rows = []
    for row in tqdm(sample.itertuples(index=False), total=len(sample)):
        prompt = build_prompt(tokenizer, getattr(row, PROMPT_COL))
        score, prompt_tokens = one_token_score(model, tokenizer, prompt, device)
        answer = generate_answer(model, tokenizer, prompt, device)
        parsed = parse_yes_no(answer)
        rows.append({
            "sample_id": row.sample_id,
            "split": row.split,
            "label": int(getattr(row, LABEL_COL)),
            "score_yes": score,
            "pred_05": int(score >= 0.5),
            "generated_answer": answer,
            "parsed_answer": parsed,
            "parse_ok": parsed is not None,
            "prompt_tokens": prompt_tokens,
        })

    pred = pd.DataFrame(rows)
    pred_path = OUT_DIR / "zero_shot_smoke_predictions.csv"
    pred.to_csv(pred_path, index=False)

    y_true = pred["label"].values
    score = pred["score_yes"].values
    pred_05 = pred["pred_05"].values
    metrics = calc_metrics(y_true, score, pred_05)
    parse_rate = float(pred["parse_ok"].mean())

    config = {
        "model_name": MODEL_NAME,
        "split": SPLIT,
        "n_examples": int(len(pred)),
        "n_per_class": N_PER_CLASS,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "score_type": "P(Yes)/(P(Yes)+P(No)) from next-token logits",
        "threshold_debug": 0.5,
        "parse_rate": parse_rate,
        "metrics_debug_threshold_05": metrics,
    }
    (OUT_DIR / "zero_shot_smoke_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Examples: {len(pred)}")
    print(f"Parse rate: {parse_rate:.4f}")
    print("\nMetrics with threshold=0.5:")
    print(metrics)
    print("\nPredictions preview:")
    print(pred[["sample_id", "label", "score_yes", "pred_05", "generated_answer", "parsed_answer", "prompt_tokens"]].head(20).to_string(index=False))
    print("\nSaved:")
    print(pred_path)
    print(OUT_DIR / "zero_shot_smoke_config.json")

if __name__ == "__main__":
    main()