# Baselines summary

Source: `outputs/baselines/metrics_baselines.csv`

Best classical baseline: `safe_catboost_plus_popularity`.

Main result:
- val ROC-AUC: 0.8159916400000001
- test ROC-AUC: 0.79452272
- test PR-AUC: 0.7859442481402255
- test accuracy: 0.6961
- test precision: 0.6496261254387303
- test recall: 0.8514
- test F1: 0.736951441184108
- threshold: 0.387

Related artifacts:
- `outputs/baselines/metrics_baselines.csv`
- `outputs/baselines/thresholds.json`
- `outputs/baselines/catboost_plus_popularity_feature_importance.csv`
- `outputs/baselines/train_item_popularity.csv`

Interpretation:
`safe_catboost_plus_popularity` is the strongest current non-LLM baseline and remains the main classical reference point for QLoRA comparison.
