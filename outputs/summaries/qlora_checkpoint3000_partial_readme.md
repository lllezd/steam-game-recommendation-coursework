# Ordinary QLoRA checkpoint-3000 partial evaluation

Stage: checkpoint3000_partial

This folder preserves the checkpoint artifacts for an ordinary QLoRA run that
reached approximately 80 percent of the planned 30k training. It is not the main
official result. The main ordinary QLoRA result is checkpoint-2500 under
`outputs/qwen_qlora/ordinary/checkpoint2500_eval/`.

Standalone checkpoint-3000 CSV/JSON eval outputs were not present during cleanup;
the executed eval notebook is kept at
`notebooks/04_qlora/evaluation/qwen_checkpoint3000_partial_eval_only.ipynb`.
