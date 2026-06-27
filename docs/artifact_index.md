# Artifact Index

This file lists the small artifacts included in the clean GitHub repository and the large artifacts intentionally omitted.

## Final Report

- `reports/final/coursework_final_note.pdf` - final coursework note.

## Final Metrics

- `outputs/metrics/classical_baselines_metrics.csv` - classical baseline metrics.
- `outputs/metrics/prompt_llm_baselines_metrics.csv` - prompt-only LLM baseline metrics.
- `outputs/metrics/semantic_features_comparison_metrics.csv` - quick semantic feature comparison.
- `outputs/metrics/catboost_semantic_metrics.csv` - CatBoost with semantic features.
- `outputs/metrics/catboost_semantic_with_qlora_reference_metrics.csv` - CatBoost semantic metrics with QLoRA reference.
- `outputs/metrics/same_budget_catboost_metrics.csv` - same-budget CatBoost comparison.
- `outputs/metrics/qlora_qwen3_4b_metrics.csv` - QLoRA evaluation metrics.
- `outputs/metrics/rella_lite_10k_metrics.csv` - ReLLa-lite 10k evaluation metrics.
- `outputs/metrics/rella_lite_20k_metrics.csv` - ReLLa-lite 20k evaluation metrics.
- `outputs/metrics/rella_lite_30k_metrics.csv` - ReLLa-lite 30k evaluation metrics.
- `outputs/metrics/group_analysis_all_metrics.csv` - long-tail and cold-start group metrics.
- `outputs/metrics/group_analysis_catboost_metrics.csv` - CatBoost group metrics.
- `outputs/metrics/group_analysis_qlora_metrics.csv` - QLoRA group metrics.
- `outputs/metrics/group_analysis_overall_metrics.csv` - overall group analysis metrics.
- `outputs/metrics/group_analysis_fair_subset_metrics.csv` - fair-subset group metrics.

## Summaries

Small summaries are stored under `outputs/summaries/`. They include baseline summaries, semantic feature summaries, QLoRA evaluation summaries, ReLLa-lite training/evaluation summaries, and group analysis notes.

## Omitted Large Artifacts

The following artifact classes are not included in this repository:

- raw source data files such as `recommendations.csv`, `games.csv`, and `games_metadata.json`;
- processed `.parquet` datasets;
- instruction `.jsonl` splits;
- large prediction CSV files;
- CatBoost binary model files (`.cbm`);
- QLoRA checkpoints and optimizer states;
- adapter weights (`.safetensors`, `.bin`, `.pt`, `.pth`);
- semantic embedding arrays (`.npy`);
- HuggingFace cache and local Python environment folders;
- zipped model/data exports.

## Files Needed for Full Local Reproduction

For full reproduction, restore the local data paths described in `docs/data_files.md`, install `requirements.txt`, and then run the relevant scripts/notebooks in this order:

1. `scripts/data_preparation/`
2. `scripts/baselines/`
3. `scripts/llm_baselines/`
4. `notebooks/semantic_features/`
5. `notebooks/qlora/`
6. `notebooks/rella_lite/`
7. `notebooks/analysis/`