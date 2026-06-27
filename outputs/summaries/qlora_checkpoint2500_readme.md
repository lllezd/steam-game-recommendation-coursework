# QLoRA checkpoint-2500 evaluation

Model: Qwen/Qwen3-4B-Instruct-2507

Dataset: instruction_temporal

Training setup:
- 20k train examples
- 1 epoch
- checkpoint-2500
- clean instruction prompt
- answer-only loss

Evaluation:
- val: 2000 examples, balanced
- test: 2000 examples, balanced
- threshold selected on validation

Main result:
- val ROC-AUC: 0.7962935
- val PR-AUC: 0.7923404137570969
- val F1: 0.7424657534246575
- test ROC-AUC: 0.795959
- test PR-AUC: 0.7900838560066099
- test F1: 0.741745816372682
- threshold: 0.44

Artifacts:
- checkpoint: `checkpoint/`
- evaluation outputs: `eval_outputs/`
- metrics source: `eval_outputs/qwen_checkpoint_eval_metrics.csv`
- summary source: `eval_outputs/qwen_checkpoint_eval_summary.json`

Interpretation:
This is the main current QLoRA / TALLRec-style result.
