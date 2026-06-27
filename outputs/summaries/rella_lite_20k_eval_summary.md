# ReLLa-lite QLoRA evaluation

Base model: `Qwen/Qwen3-4B-Instruct-2507`
Adapter: `/kaggle/input/datasets/lllezz/qwen-rella-lite-continued-20k`
Data: `/kaggle/input/datasets/lllezz/final-instruction-temporal-retrieved-topk`
Eval subset: `1000` examples per class
Threshold selected on validation: `0.3800`

## Validation

- ROC-AUC: `0.780475`
- PR-AUC: `0.763605`
- Accuracy: `0.705500`
- Precision: `0.664006`
- Recall: `0.832000`
- F1: `0.738571`

## Test

- ROC-AUC: `0.768358`
- PR-AUC: `0.742488`
- Accuracy: `0.691500`
- Precision: `0.652590`
- Recall: `0.819000`
- F1: `0.726386`
