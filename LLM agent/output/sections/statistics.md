
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