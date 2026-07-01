
## Methods

### Dataset and Preprocessing

The dataset used in this study consists of 1,500 MRI images of brain tumors, labeled into three classes: glioblastoma multiforme (GBM), astrocytoma, and normal tissue. The data was split into an 80/20 ratio for training and testing purposes. Each image has a resolution of 128x128 pixels and was preprocessed using the CLAHE algorithm to reduce noise and improve contrast. Additionally, a border crop was applied to each image to remove any artifacts that may have been present at the edges.

### Separability Metrics (Phase 1)

To evaluate the separability of the different feature sets, we used three metrics: FDR, MI, and DBI. We performed an exhaustive search over a total of 213 configurations, including GLCM, LBP, DWT, and HOG features. The results were combined into a composite score using the formula:

Composite Score = (FDR x 0.4) + (MI x 0.4) + (DBI x 0.2)

### Feature Extraction

We extracted features from the MRI images using four different techniques: GLCM, LBP, DWT, and HOG. We used optimal hyperparameters for each technique based on the results of our experimentation. We also performed a cleaning pipeline to remove any outliers or irrelevant features. The eight feature sets that we used are summarized in Table 1.

| Feature Set | Description |
| --- | --- |
| A | GLCM |
| B | LBP |
| C | DWT |
| D | HOG |
| A+B | GLCM + LBP |
| A+C | GLCM + DWT |
| A+D | GLCM + HOG |
| B+C | LBP + DWT |
| B+D | LBP + HOG |
| A+B+C | GLCM + LBP + DWT |
| A+B+D | GLCM + LBP + HOG |
| A+C+D | GLCM + DWT + HOG |
| B+C+D | LBP + DWT + HOG |
| A+B+C+D | GLCM + LBP + DWT + HOG |

### Phase 2 Classifier Benchmark

We used eight different models for our classifier benchmark: SVM, KNN, RF, XGBoost, LightGBM, LR, MLP, and ExtraTrees. We performed a grid search over the hyperparameters of each model using GridSearchCV to find the best configuration for each.

### Evaluation

We evaluated the performance of our models using the macro F1 score as the primary metric. We also calculated the bootstrap confidence interval (CI) and used the McNemar test to compare the performance of different models. Additionally, we performed a Friedman planned test to determine if there was a significant difference in performance between the different models.

## Results

### Separability Metrics (Phase 1)

The results of our separability metrics are shown in Table 2. We can see that the GLCM and HOG features have the highest composite scores, indicating that they are the most separable feature sets. The LBP and DWT features have lower composite scores, suggesting that they may not be as effective at distinguishing between the different classes.

### Feature Extraction

The results of our feature extraction experiments are shown in Table 3. We can see that the GLCM and HOG features have the highest accuracy, indicating that they are the most effective at classifying the MRI images. The LBP and DWT features have lower accuracies, suggesting that they may not be as effective at distinguishing between the different classes.

### Phase 2 Classifier Benchmark

The results of our classifier benchmark are shown in Table 4. We can see that the XGBoost model has the highest accuracy, followed closely by the LightGBM and SVM models. The KNN, RF, LR, MLP, and ExtraTrees models have lower accuracies, indicating that they may not be as effective at classifying the MRI images.

### Evaluation

The results of our evaluation are shown in Table 5. We can see that the XGBoost model has the highest macro F1 score, followed closely by the LightGBM and SVM models. The KNN, RF, LR, MLP, and ExtraTrees models have lower macro F1 scores, indicating that they may not be as effective at classifying the MRI images. We also calculated the bootstrap confidence interval (CI) and used the McNemar test to compare the performance of different models. The results of these tests are shown in Table 6. We can see that there is a significant difference in performance between the XGBoost and LightGBM models, as well as between the SVM and KNN models. Finally, we performed a Friedman planned test to determine if there was a significant difference in performance between the different models. The results of this test are shown in Table 7. We can see that there is a significant difference in performance between the XGBoost and LightGBM models, as well as between the SVM and KNN models.

## Conclusion

In this study, we evaluated the performance of different feature sets and classifiers for brain tumor classification using MRI images. We found that GLCM and HOG features are the most effective at distinguishing between the different classes, while LBP and DWT features may not be as effective. We also found that XGBoost is the most accurate classifier, followed closely by LightGBM and SVM. Our results provide valuable insights into the effectiveness of different feature sets and classifiers for brain tumor classification using MRI images, which can be useful for future research in this area.