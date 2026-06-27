# Long-tail / cold-start analysis

This report evaluates CatBoost, CatBoost with semantic features, and QLoRA across item popularity and user history groups.

## Overall metrics

                      method  test_roc_auc  test_pr_auc  test_accuracy  test_precision  test_recall  test_f1  test_threshold  val_roc_auc  n_features  best_iteration
         catboost_popularity      0.794523     0.785944         0.6961        0.649626       0.8514 0.736951           0.387     0.815992          27             136
catboost_popularity_semantic      0.794059     0.784248         0.7024        0.658422       0.8412 0.738672           0.403     0.815797          34             118

## Fair subset metrics on QLoRA samples

                                      method    n  positive_rate  accuracy  precision  recall       f1   pr_auc  roc_auc
         catboost_popularity_on_qlora_subset 2000            0.5    0.7060   0.660436   0.848 0.742557 0.800001 0.804198
catboost_popularity_semantic_on_qlora_subset 2000            0.5    0.7120   0.668254   0.842 0.745133 0.798319 0.803913
                   qwen_qlora_checkpoint2500 2000            0.5    0.7145   0.677126   0.820 0.741746 0.790084 0.795959

## Group metrics

                      method               group_type         group    n  positive_rate  accuracy  precision   recall       f1   pr_auc  roc_auc
         catboost_popularity        item_count_bucket        cold_0 1706       0.422626  0.614889   0.529685 0.791956 0.634797 0.608818 0.700698
         catboost_popularity        item_count_bucket head_100_plus 3342       0.453920  0.737882   0.670027 0.832564 0.742504 0.814278 0.840296
         catboost_popularity        item_count_bucket    mid_21_100 2566       0.571317  0.717459   0.694693 0.901774 0.784803 0.815194 0.796474
         catboost_popularity        item_count_bucket      rare_3_5  475       0.557895  0.675789   0.662757 0.852830 0.745875 0.762259 0.725984
         catboost_popularity        item_count_bucket     tail_6_20 1375       0.553455  0.680000   0.663276 0.856767 0.747706 0.801469 0.765198
         catboost_popularity        item_count_bucket very_rare_1_2  536       0.503731  0.651119   0.614325 0.825926 0.704581 0.715895 0.724715
catboost_popularity_semantic        item_count_bucket        cold_0 1706       0.422626  0.628957   0.541985 0.787795 0.642171 0.612261 0.703663
catboost_popularity_semantic        item_count_bucket head_100_plus 3342       0.453920  0.749252   0.687259 0.821358 0.748348 0.812345 0.839508
catboost_popularity_semantic        item_count_bucket    mid_21_100 2566       0.571317  0.716680   0.696438 0.893588 0.782791 0.811462 0.795027
catboost_popularity_semantic        item_count_bucket      rare_3_5  475       0.557895  0.669474   0.663636 0.826415 0.736134 0.759554 0.721599
catboost_popularity_semantic        item_count_bucket     tail_6_20 1375       0.553455  0.682909   0.669801 0.842313 0.746217 0.802213 0.766275
catboost_popularity_semantic        item_count_bucket very_rare_1_2  536       0.503731  0.654851   0.618384 0.822222 0.705882 0.713331 0.722041
         catboost_popularity item_popularity_quantile       Q1_tail 2500       0.440000  0.623600   0.549165 0.807273 0.653662 0.643795 0.712021
         catboost_popularity item_popularity_quantile    Q2_low_mid 2500       0.569200  0.688400   0.676149 0.868587 0.760381 0.792261 0.759575
         catboost_popularity item_popularity_quantile   Q3_high_mid 2500       0.581200  0.736000   0.715138 0.907089 0.799757 0.841901 0.816990
         catboost_popularity item_popularity_quantile       Q4_head 2500       0.409600  0.736400   0.644269 0.795898 0.712101 0.789843 0.834637
catboost_popularity_semantic item_popularity_quantile       Q1_tail 2500       0.440000  0.634800   0.559215 0.802727 0.659201 0.645279 0.712708
catboost_popularity_semantic item_popularity_quantile    Q2_low_mid 2500       0.569200  0.691200   0.682353 0.855938 0.759352 0.791688 0.759101
catboost_popularity_semantic item_popularity_quantile   Q3_high_mid 2500       0.581200  0.736800   0.719250 0.897454 0.798530 0.836081 0.815451
catboost_popularity_semantic item_popularity_quantile       Q4_head 2500       0.409600  0.746800   0.661437 0.782227 0.716779 0.788456 0.833767
         catboost_popularity       history_len_bucket      hist_3_5 2401       0.528530  0.680966   0.645124 0.881009 0.744837 0.782584 0.779973
         catboost_popularity       history_len_bucket     hist_6_10 7599       0.490986  0.700882   0.651245 0.841329 0.734183 0.787576 0.799128
catboost_popularity_semantic       history_len_bucket      hist_3_5 2401       0.528530  0.686381   0.652844 0.868400 0.745350 0.780335 0.779437
catboost_popularity_semantic       history_len_bucket     hist_6_10 7599       0.490986  0.707462   0.660426 0.831949 0.736330 0.786283 0.798780
   qwen_qlora_checkpoint2500        item_count_bucket        cold_0  343       0.431487  0.673469   0.585714 0.831081 0.687151 0.732445 0.780527
   qwen_qlora_checkpoint2500        item_count_bucket head_100_plus  647       0.459042  0.724884   0.667606 0.797980 0.726994 0.829887 0.837441
   qwen_qlora_checkpoint2500        item_count_bucket    mid_21_100  523       0.562141  0.745698   0.741742 0.840136 0.787879 0.816744 0.799119
   qwen_qlora_checkpoint2500        item_count_bucket      rare_3_5  113       0.584071  0.690265   0.724638 0.757576 0.740741 0.768508 0.714700
   qwen_qlora_checkpoint2500        item_count_bucket     tail_6_20  258       0.503876  0.717054   0.674847 0.846154 0.750853 0.769901 0.784255
   qwen_qlora_checkpoint2500        item_count_bucket very_rare_1_2  116       0.560345  0.655172   0.654321 0.815385 0.726027 0.654844 0.654299
   qwen_qlora_checkpoint2500 item_popularity_quantile       Q1_tail  518       0.469112  0.673745   0.613497 0.823045 0.702988 0.705657 0.747999
   qwen_qlora_checkpoint2500 item_popularity_quantile    Q2_low_mid  498       0.552209  0.692771   0.695513 0.789091 0.739353 0.764307 0.743465
   qwen_qlora_checkpoint2500 item_popularity_quantile   Q3_high_mid  499       0.555110  0.773547   0.746988 0.895307 0.814450 0.863066 0.846465
   qwen_qlora_checkpoint2500 item_popularity_quantile       Q4_head  485       0.422680  0.719588   0.643154 0.756098 0.695067 0.800334 0.827544
   qwen_qlora_checkpoint2500       history_len_bucket      hist_3_5  479       0.542797  0.703549   0.672515 0.884615 0.764120 0.805900 0.780989
   qwen_qlora_checkpoint2500       history_len_bucket     hist_6_10 1521       0.486522  0.717949   0.678941 0.797297 0.733375 0.785150 0.799231
