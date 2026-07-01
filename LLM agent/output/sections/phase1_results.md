
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