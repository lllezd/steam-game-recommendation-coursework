# Qwen QLoRA checkpoint-2500 summary

Model: `Qwen/Qwen3-4B-Instruct-2507`

Data: `data/final/instruction_temporal/`

Training setup:
- 20k train examples
- 1 epoch
- checkpoint-2500
- clean instruction prompt
- answer-only loss

Evaluation setup:
- val: 2000 balanced examples
- test: 2000 balanced examples
- eval per class: 1000
- max sequence length: 1024
- selected threshold: 0.44

Artifact paths:
- checkpoint: `outputs/qwen_qlora/ordinary/checkpoint2500_eval/checkpoint/`
- eval outputs: `outputs/qwen_qlora/ordinary/checkpoint2500_eval/eval_outputs/`
- metrics: `outputs/qwen_qlora/ordinary/checkpoint2500_eval/eval_outputs/qwen_checkpoint_eval_metrics.csv`
- summary: `outputs/qwen_qlora/ordinary/checkpoint2500_eval/eval_outputs/qwen_checkpoint_eval_summary.json`

Main metrics:

| split | ROC-AUC | PR-AUC | accuracy | precision | recall | F1 | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| val_pilot | 0.7962935 | 0.7923404137570969 | 0.718 | 0.6831932773109244 | 0.813 | 0.7424657534246575 | 0.44 |
| test_pilot | 0.795959 | 0.7900838560066099 | 0.7145 | 0.6771263418662262 | 0.82 | 0.741745816372682 | 0.44 |

Interpretation:
This is the main current QLoRA / TALLRec-style result. It slightly exceeds the best classical baseline by ROC-AUC on the balanced pilot test set, while the classical baseline remains the stronger full-test tabular reference.
