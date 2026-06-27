# ReLLa-lite QLoRA evaluation

Base model: `Qwen/Qwen3-4B-Instruct-2507`
Adapter: `/kaggle/input/datasets/lllezz/adapter-qwen-rella-lite-continued-30k`
Data: `/kaggle/input/datasets/lllezz/final-instruction-temporal-retrieved-topk`
Eval subset: `1000` examples per class
Threshold selected on validation: `0.4200`

## Validation

- ROC-AUC: `0.769536`
- PR-AUC: `0.752860`
- Accuracy: `0.711000`
- Precision: `0.677311`
- Recall: `0.806000`
- F1: `0.736073`

## Test

- ROC-AUC: `0.766032`
- PR-AUC: `0.747297`
- Accuracy: `0.697000`
- Precision: `0.665825`
- Recall: `0.791000`
- F1: `0.723035`
