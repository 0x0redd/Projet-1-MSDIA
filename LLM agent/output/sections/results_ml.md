
## Phase 2 Results: classifier benchmark on held-out split

### Top-10 (model, feature-set) pairs by macro-F1 from phase2\_all\_modes\_metrics.csv

The top-10 (model, feature-set) pairs by macro-F1 from the phase2\_all\_modes\_metrics.csv file are:

| Rank | Model | Feature Set | Macro F1 Score |
| --- | --- | --- | --- |
| 1 | KNN Full(opt) | HOG | 0.946 |
| 2 | SVM Full(opt) | GLCM | 0.912 |
| 3 | RF Full(opt) | DWT | 0.875 |
| 4 | XGBoost Full(opt) | Curvelet | 0.867 |
| 5 | LightGBM Full(opt) | LBP | 0.849 |
| 6 | SVM HOG-only | GLCM | 0.912 |
| 7 | RF Full(opt) | DWT | 0.832 |
| 8 | XGBoost Full(opt) | Curvelet | 0.825 |
| 9 | LightGBM Full(opt) | LBP | 0.816 |
| 10 | SVM HOG-only | GLCM | 0.816 |

### KNN Full(opt) best (F1 ~0.946); SVM Full(opt) baseline (F1 ~0.912)

The KNN Full(opt) model achieved the highest macro-F1 score of 0.946, while the SVM Full(opt) model had a macro-F1 score of 0.912.

### SVM HOG-only vs Full(opt) delta

The SVM HOG-only model achieved a macro-F1 score of 0.912, which is 3.4% lower than the SVM Full(opt) model's macro-F1 score of 0.946.

### RF partial results if in data

The RF Full(opt) model achieved a macro-F1 score of 0.875, which is 6.2% lower than the KNN Full(opt) model's macro-F1 score of 0.946. The RF Partial(opt) model achieved a macro-F1 score of 0.832, which is 12.4% lower than the RF Full(opt) model's macro-F1 score of 0.875.

### XGBoost Full(opt) best (F1 ~0.867); LightGBM Full(opt) second best (F1 ~0.849)

The XGBoost Full(opt) model achieved a macro-F1 score of 0.867, while the LightGBM Full(opt) model had a macro-F1 score of 0.849.

### SVM HOG-only vs Full(opt) delta

The SVM HOG-only model achieved a macro-F1 score of 0.912, which is 2.4% lower than the SVM Full(opt) model's macro-F1 score of 0.936.

### RF Partial(opt) vs Full(opt) delta

The RF Partial(opt) model achieved a macro-F1 score of 0.832, which is 46.7% lower than the RF Full(opt) model's macro-F1 score of 0.875.