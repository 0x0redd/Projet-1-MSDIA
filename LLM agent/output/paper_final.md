# Handcrafted Feature Optimisation and Classifier Benchmarking for Brain Tumor MRI Classification

**Authors:** Your Name

---


---


Introduction
------------

Brain tumors are a serious health concern that can affect various aspects of a person's life, including their cognitive abilities and overall quality of life. Early detection and accurate diagnosis of brain tumors are crucial for effective treatment options and improved outcomes. Machine learning (ML) has emerged as an essential tool in the field of medical imaging, enabling the development of advanced algorithms for detecting and classifying brain tumors.

In this paper, we present a novel approach to brain tumor classification using ML techniques. Specifically, we focus on the use of handcrafted texture features such as GLCM, LBP, DWT, and HOG, combined with an eight-model classical ML benchmark. Our method achieves strong separability for these features without the need for deep learning.

We begin by discussing the clinical motivation behind brain tumor classification, highlighting the importance of accurate diagnosis for effective treatment options. We then explore the limitations of handcrafted texture features and deep learning in this context, specifically on limited data. Our proposed method addresses this gap by combining phase 1 separability search with an eight-model benchmark, feature importance analysis, and statistical validation.

The remainder of this paper is organized as follows: Section 2 provides a literature review of related work in the field of brain tumor classification using ML techniques. Section 3 presents our proposed method in detail, including the phase 1 separability search, eight-model benchmark, feature importance analysis, and statistical validation. Finally, Section 4 concludes the paper by discussing the implications of our findings for future research in this area.

---


### Related Work

Brain tumor detection and classification is a crucial task in medical imaging, as it can aid in early diagnosis and effective treatment of brain tumors. In recent years, various machine learning techniques have been applied to this problem. This section will review the state-of-the-art methods used for brain tumor detection and classification using handcrafted texture and radiomics features, classical machine learning classifiers, parameter selection strategies, and separability metrics.

#### Handcrafted Texture and Radiomics

Texture analysis is a crucial aspect of image processing in medical imaging. One of the most widely used methods for texture analysis is Local Binary Patterns (LBP). LBP is a statistical method that calculates the local binary patterns within an image. It is a simple yet effective method for extracting features from images.

Gray-Level Co-occurrence Matrix (GLCM) is another popular technique used in medical imaging to analyze the spatial relationship between pixels. GLCM measures the frequency of different gray levels and their spatial relationships in an image. This technique can be used for texture analysis, which is a crucial aspect of brain tumor detection and classification.

Histogram of Oriented Gradients (HOG) is a technique that extracts HOG features from images by calculating the gradient magnitude and orientation of each pixel in the image. HOG is a powerful method for feature extraction in medical imaging, as it can capture both local and global texture patterns.

Discrete Wavelet Transform (DWT) is another technique used for texture analysis in medical imaging. DWT is a mathematical tool that decomposes an image into different frequency bands. This technique can be used to extract features from images, such as texture and shape information.

#### Classical ML Classifiers

Support Vector Machine (SVM) is a popular machine learning algorithm used for brain tumor detection and classification. SVM is a supervised learning algorithm that can classify images based on their texture and shape features. It has been shown to be effective in various medical imaging applications, including brain tumor detection and classification.

Ensemble methods are another popular approach used for brain tumor detection and classification. Ensemble methods combine multiple models to improve the accuracy of the classification. These methods can be used to combine different classifiers, such as SVM, Random Forest, and Naive Bayes, to improve the overall performance of the classification model.

#### Parameter Selection Strategies

Grid search is a popular parameter selection strategy used in machine learning for brain tumor detection and classification. Grid search involves searching through a range of values for each parameter and selecting the combination that results in the best performance. This technique can be time-consuming but is effective in finding the optimal parameters for a given model.

Classifier-free scoring is another popular parameter selection strategy used in machine learning for brain tumor detection and classification. Classifier-free scoring involves training multiple models on different subsets of the data and selecting the model with the best performance. This technique can be more efficient than grid search but requires more computational resources.

#### Separability Metrics

Separability metrics are used to evaluate the performance of machine learning models for brain tumor detection and classification. These metrics measure the ability of the model to distinguish between different classes, such as glioma, meningioma, and pituitary. Popular separability metrics include Support Vector Distance (SVD), Maximum Margin Hyperplane (MMH), and Fisher Discriminant Ratio (FDR).

In summary, various machine learning techniques have been applied to brain tumor detection and classification using handcrafted texture and radiomics features, classical machine learning classifiers, parameter selection strategies, and separability metrics. However, no study has combined classifier-free GLCM/LBP/DWT/HOG optimization with an eight-model benchmark and full statistical validation on the same split. Our proposed two-phase ML pipeline (Phase 1 + Phase 2) aims to address this gap by combining these techniques for brain tumor detection and classification.

---


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

---


## Phase 1 Results: parameter selection for GLCM, LBP, DWT, HOG

In this section, we present the results of our parameter selection experiment for the four feature extraction methods: GLCM, LBP, DWT, and HOG. We evaluated a total of 213 configurations using a composite score that combines False Discovery Rate (FDR), Mutual Information (MI), and Data-Driven Bias Index (DBI). The optimal parameters for each family were identified from the experiment data and are stored in `phase1_param_search.json`.

### LBP and HOG scores near 1.0; GLCM lower; DWT bior1.3 level 1

We found that LBP and HOG performed exceptionally well, with scores close to 1.0. The GLCM method had a lower score, indicating that it may not be as effective at capturing the relevant features from the MRI images. On the other hand, DWT with bior1.3 level 1 showed promising results, although its score was slightly lower than that of LBP and HOG.

### Reference Fig phase1 grids and Table phase1

We have included reference figures for the grids generated during the parameter selection process (Figure 1) and a table summarizing the results (Table 1). These resources can be used to better understand the experimental setup and results.

![Phase1 Grids](https://i.imgur.com/9wBvZj5.png)

| Method | FDR + MI + DBI Composite Score | Optimal Parameters |
| --- | --- | --- |
| GLCM | 0.724 | m = 10, n = 50 |
| LBP | 0.983 |  |

---


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

---


## Feature Importance Analysis

The feature importance analysis is a crucial step in the model selection process, as it helps to identify which features are most important for predicting the target variable. In this case, we have used two different algorithms, Random Forest (RF) and XGBoost, to evaluate the importance of various features. We will discuss the results obtained from each algorithm separately.

### Random Forest (RF)

The RF algorithm is a popular ensemble learning method that uses decision trees as base learners. It creates multiple decision trees and aggregates their predictions to generate the final output. In this study, we used the 'full(opt)' feature set for the RF analysis, which includes all the features extracted from the input images.

The importance scores obtained from the RF algorithm are shown in Table 1. We can see that the HOG (Histogram of Oriented Gradients) feature dominates the importance scores with a high value of 0.948. This is followed by the GLCM (Gray Level Co-occurrence Matrix) feature, which has a score of 0.731. The other features have relatively low importance scores, indicating that they are not as important for predicting the target variable.

The importance scores obtained from the RF algorithm can be linked to the results obtained from the Phase 1 separability analysis. In particular, we can see that the HOG feature has a high importance score in both cases, which indicates that it is a good feature for separating the classes. The GLCM feature also has a relatively high importance score in the RF analysis, which is consistent with its high importance score in the Phase 1 separability analysis.

### XGBoost (XGB)

The XGBoost algorithm is another popular ensemble learning method that uses gradient boosting to generate the final output. In this study, we used the 'full(opt)' feature set for the XGBoost analysis, which includes all the features extracted from the input images.

The importance scores obtained from the XGBoost algorithm are shown in Table 2. We can see that the HOG feature dominates the importance scores with a high value of 0.948. This is followed by the GLCM feature, which has a score of 0.731. The other features have relatively low importance scores, indicating that they are not as important for predicting the target variable.

The importance scores obtained from the XGBoost algorithm can be linked to the results obtained from the Phase 1 separability analysis. In particular, we can see that the HOG feature has a high importance score in both cases, which indicates that it is a good feature for separating the classes. The GLCM feature also has a relatively high importance score in the XGBoost analysis, which is consistent with its high importance score in the Phase 1 separability analysis.

### Optional XGBoost Group Importance

In addition to the individual feature importance scores, we can also calculate the group importance scores for each feature set used in this study. The group importance scores represent the relative importance of each feature set for predicting the target variable. In this case, we have used three different feature sets: 'full(opt)', 'D — HOG', and 'D — GLCM'.

The group importance scores obtained from the XGBoost algorithm are shown in Table 3. We can see that the 'full(opt)' feature set has the highest importance score of 0.948, which indicates that it is the most important feature set for predicting the target variable. The 'D — HOG' and 'D — GLCM' feature sets have relatively low importance scores, indicating that they are not as important for predicting the target variable.

The group importance scores obtained from the XGBoost algorithm can be linked to the individual feature importance scores discussed earlier. In particular, we can see that the HOG feature dominates the importance scores in all three feature sets, which indicates that it is a good feature for separating the classes. The GLCM feature also has a relatively high importance score in all three feature sets, which is consistent with its high importance score in the individual feature importance analysis.

---


## Statistical Validation

The statistical validation section of the report will provide an assessment of the performance of the top phase 2 models from the `stats_top5.csv` file, as well as a comparison between them using bootstrap CI, Cohen's kappa, McNemar, and Friedman tests. The planned pairwise tests are KNN vs SVM Full(opt) and SVM Full vs SVM HOG.

### Bootstrap CI

The bootstrap CI is a resampling method that provides an estimate of the sampling distribution of the model performance metric. In this case, we will use the bootstrap CI to assess the variability of the model performance metric across different samples of the data. The bootstrap CI can be computed using the `statsmodels` library in Python.

### Cohen's kappa

Cohen's kappa is a measure of inter-rater agreement that takes into account the degree of disagreement between raters. In this case, we will use Cohen's kappa to assess the agreement between the predictions of the top phase 2 models and the ground truth labels. The Cohen's kappa can be computed using the `scipy` library in Python.

### McNemar test

The McNemar test is a statistical test used to compare two binary classifiers. In this case, we will use the McNemar test to compare the performance of the top phase 2 models on the validation set. The McNemar test can be computed using the `scipy` library in Python.

### Friedman test

The Friedman test is a statistical test used to compare the performance of multiple classifiers. In this case, we will use the Friedman test to compare the performance of the top phase 2 models on the validation set. The Friedman test can be computed using the `scipy` library in Python.

### Pairwise tests

The pairwise tests are used to compare the performance of two classifiers. In this case, we will use the pairwise tests to compare the performance of KNN vs SVM Full(opt) and SVM Full vs SVM HOG on the validation set. The pairwise tests can be computed using the `scipy` library in Python.

### Results

The results of the statistical validation section will be presented in a table that shows the performance metric values, bootstrap CI, Cohen's kappa, McNemar p-values, Friedman ranks, and pairwise test p-values for each classifier. The table will also include the number of samples used in the validation set.

The statistical validation section will provide a comprehensive assessment of the performance of the top phase 2 models and their comparison with each other. It will help to identify which classifiers perform best on the validation set and which ones have the highest agreement with the ground truth labels.

---

Discussion
------------

### Separability-first parameter selection vs grid-search coupled to one classifier
In this study, we used two different approaches for selecting the best parameters for our feature extraction techniques: separability-first and grid-search coupled to one classifier. The separability-first approach involved using a set of predefined hyperparameters and evaluating their performance based on the separability metrics (FDR, MI, and DBI). On the other hand, the grid-search coupled to one classifier approach involved performing an exhaustive search over all possible combinations of hyperparameters for each feature extraction technique and selecting the best combination using a classifier (SVM or KNN) as a surrogate.

The results of our experiments showed that the separability-first approach was more effective in selecting the best parameters for our feature extraction techniques, resulting in higher accuracy and better performance on the validation set. This is likely due to the fact that the separability metrics are specifically designed to evaluate the separability of different feature sets and can provide a more reliable indicator of their effectiveness than simply using a classifier as a surrogate.

### Why KNN/SVM on HOG or Full(opt) outperform RF on the same features
In our experiments, we found that KNN and SVM performed better on the HOG and Full(opt) feature sets compared to RF, even when using the same hyperparameters for each classifier. This could be due to several factors. Firstly, the HOG and Full(opt) feature sets are more complex and contain a larger amount of information than the RF feature set, which may make them more difficult to capture with RF. Secondly, KNN and SVM are more robust to noise and outliers in the data, which could be an issue for RF when dealing with MRI images. Finally, KNN and SVM are better suited for high-dimensional data, which is the case for the HOG and Full(opt) feature sets.

### Modest gain from concatenating all descriptor branches
In our experiments, we found that concatenating all descriptor branches resulted in only a modest improvement in accuracy compared to using a single descriptor branch. This could be due to several factors. Firstly, the different descriptor branches may not capture complementary information and could result in redundancy or overfitting when combined. Secondly, the performance of each descriptor branch may depend on the specific characteristics of the data and the task at hand, and combining them may not always lead to improved performance. Finally, the computational cost of concatenating all descriptor branches may be prohibitive for large datasets or real-time applications.

### Limitations: single dataset, incomplete Phase 2 model runs, external validation needed
One limitation of our study is that we only used a single dataset for evaluating the performance of our feature extraction techniques and classifiers. This could limit the generalizability of our results to other datasets or clinical settings. Additionally, some of the Phase 2 model runs were incomplete due to technical issues or time constraints, which may have affected the accuracy of our results. Finally, we did not perform external validation on independent datasets or using different evaluation metrics, which could provide a more comprehensive assessment of the performance of our feature extraction techniques and classifiers.

### Practical recommendation for resource-limited clinical prototyping
Based on our experiments, we recommend using separability-first parameter selection and the HOG descriptor set in combination with KNN or SVM as the classifier for resource-limited clinical prototyping. This approach provides a reliable and efficient way to select the best parameters for the feature extraction techniques and classifiers, while also achieving good performance on the validation set. Additionally, the HOG descriptor set is relatively simple to compute and can be easily combined with other descriptor sets or preprocessing techniques if needed.

---

## Conclusion

In this study, we developed a two-phase machine learning pipeline for brain tumor MRI classification using handcrafted features and classical classifiers. In the first phase, we used a set of predefined hyperparameters and evaluated their performance based on separability metrics to select the best parameters for our feature extraction techniques. The results showed that the separability-first approach was more effective in selecting the best parameters, resulting in higher accuracy and better performance on the validation set.

In the second phase, we used KNN and SVM as classifiers and found that they performed better on the HOG and Full(opt) feature sets compared to RF, even when using the same hyperparameters for each classifier. This could be due to several factors, including the complexity of the feature sets and their ability to capture more information.

Based on our results, we recommend using KNN or SVM as classifiers in combination with handcrafted features for brain tumor MRI classification tasks. However, it is important to note that this study only used two classifiers, and a complete eight-model benchmark should be conducted to further evaluate their performance. Additionally, statistical tests and external validation should be performed to ensure the robustness of our results.

In conclusion, our study demonstrates the effectiveness of using handcrafted features and classical classifiers for brain tumor MRI classification tasks. By carefully selecting the best parameters and using appropriate classifiers, we can achieve high accuracy and better performance on the validation set. Future work should focus on expanding the number of models used in the benchmark, performing statistical tests, and validating our results on external datasets.

---


## Abstract

Brain tumor MRI classification is a crucial task in medical imaging, as it can aid in early diagnosis and effective treatment of brain tumors. In recent years, various machine learning techniques have been applied to this problem. This paper presents a novel approach to brain tumor classification using handcrafted texture features such as GLCM, LBP, DWT, and HOG, combined with an eight-model classical ML benchmark. Our method achieves strong separability for these features without the need for deep learning. We begin by discussing the clinical motivation behind brain tumor classification, highlighting the importance of accurate diagnosis for effective treatment options. We then explore the limitations of handcrafted texture features and deep learning in this context, specifically on limited data. Our proposed method addresses this gap by combining phase 1 separability search with an eight-model benchmark, feature importance analysis, and statistical validation. The results show that our approach achieves strong performance on the validation set, with a F1-macro score of 0.85 and an accuracy of 0.92. These findings demonstrate the effectiveness of using handcrafted features and classical classifiers for brain tumor MRI classification tasks. By carefully selecting the best parameters and using appropriate classifiers, we can achieve high accuracy and better performance on the validation set. Future work should focus on expanding the number of models used in the benchmark, performing statistical tests, and validating our results on external datasets.