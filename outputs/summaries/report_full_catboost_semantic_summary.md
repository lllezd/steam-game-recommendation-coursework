# Full-data CatBoost semantic benchmark

This report compares the reproduced safe CatBoost + train-only popularity baseline with the same model extended by semantic embedding features.

The protocol follows the original train-only popularity logic: OOF encoding for train and full-train encoding for validation/test.

                                  method  val_roc_auc  test_roc_auc  test_pr_auc  test_accuracy  test_precision  test_recall  test_f1  threshold best_iteration n_features
safe_catboost_plus_popularity_reproduced     0.815992      0.794523     0.785944         0.6961        0.649626       0.8514 0.736951      0.387            136         27
  safe_catboost_plus_popularity_semantic     0.815797      0.794059     0.784248         0.7024        0.658422       0.8412 0.738672      0.403            118         34
           qwen3_4b_qlora_checkpoint2500     0.796293      0.795959     0.790084         0.7145        0.677126       0.8200 0.741746      0.440           None       None
