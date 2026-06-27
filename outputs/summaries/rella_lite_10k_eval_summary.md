# ReLLa-lite QLoRA evaluation

Base model: `Qwen/Qwen3-4B-Instruct-2507`
Adapter: `/kaggle/input/datasets/lllezz/qwen-rella-lite-train10000-adapter/adapter_qwen_rella_lite`
Data: `/kaggle/input/datasets/lllezz/final-instruction-temporal-retrieved-topk`
Eval subset: `1000` examples per class
Threshold selected on validation: `0.3700`

## Validation

- ROC-AUC: `0.770517`
- PR-AUC: `0.752282`
- Accuracy: `0.698500`
- Precision: `0.653757`
- Recall: `0.844000`
- F1: `0.736796`

## Test

- ROC-AUC: `0.760871`
- PR-AUC: `0.740172`
- Accuracy: `0.686500`
- Precision: `0.645589`
- Recall: `0.827000`
- F1: `0.725121`
