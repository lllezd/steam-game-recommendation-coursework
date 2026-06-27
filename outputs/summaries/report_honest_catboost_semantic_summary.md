# Honest CatBoost semantic comparison

This report compares safe CatBoost features, train-only popularity features, and semantic embedding features.

Forbidden features include labels, target hours, ids, and global Steam aggregate columns.

                                      method split  roc_auc   pr_auc  accuracy  precision  recall       f1  threshold  n_features
                          catboost_safe_base  test 0.782082 0.771481    0.6793   0.630022  0.8688 0.730391       0.37          20
         catboost_safe_plus_train_popularity  test 0.743752 0.746407    0.6824   0.667770  0.7260 0.695669       0.48          23
catboost_safe_plus_train_popularity_semantic  test 0.735148 0.737030    0.6760   0.653364  0.7498 0.698268       0.36          30
