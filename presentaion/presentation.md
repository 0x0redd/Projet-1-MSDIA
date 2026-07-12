# Presentation Generation Instructions

## Project 7 – Brain Tumor MRI Classification Using Traditional Machine Learning

---

# General Objective

Create a professional academic presentation based on the notebook results.

The presentation must explain:

1. Problem and motivation
2. Dataset
3. Methodology
4. Feature extraction
5. Feature optimization
6. Machine learning benchmarking
7. Statistical validation
8. Best model selection
9. Discussion
10. Conclusion

Target audience:

- University professors
- Machine Learning researchers
- Master's students

Presentation duration:

20–25 minutes

Number of slides:

22–25 slides

Language:

English

Style:

Academic, professional, visually clean.

---

# Important Rules

## DO NOT

- Do not spend multiple slides explaining MRI theory.
- Do not explain machine learning algorithms mathematically.
- Do not overload slides with text.
- Do not copy notebook code.
- Do not include implementation details.
- Do not use screenshots of code.

## MUST

- Focus on experimental methodology.
- Focus on obtained results.
- Focus on interpretation of results.
- Use notebook-generated figures whenever available.
- Add speaker notes if possible.
- Use concise bullet points.

---

# Required Structure

---

## Slide 1 – Title

Title:

Brain Tumor MRI Classification Using Traditional Machine Learning

Include:

- Course name
- Student names
- Institution
- Date

---

## Slide 2 – Problem Statement

Explain:

- Brain tumor diagnosis is critical.
- MRI interpretation is time-consuming.
- Machine learning can assist diagnosis.

Objectives:

- Extract discriminative MRI features.
- Optimize feature parameters.
- Compare multiple ML classifiers.
- Identify the best classification pipeline.

---

## Slide 3 – Dataset

Present:

- MRI dataset
- Number of classes
- Number of samples

Include:

class_balance_train.png

Explain class distribution.

---

## Slide 4 – Methodology Overview

Create a workflow:

Dataset

→ Preprocessing

→ Feature Extraction

→ Feature Optimization

→ Classification

→ Evaluation

Use:

p2_overall_dashboard.png

or create a clean workflow diagram.

---

# PHASE 1

# Feature Engineering and Optimization

---

## Slide 5 – Preprocessing

Explain:

- Resize
- Normalization
- Preparation of MRI images

Keep concise.

---

## Slide 6 – Feature Extraction Methods

Present:

### Statistical Features

Intensity statistics.

### LBP

Texture descriptors.

### DWT

Frequency information.

### HOG

Shape and edge descriptors.

---

## Slide 7 – Optimization Strategy

Explain:

Instead of training classifiers for every configuration:

Use feature quality metrics to select optimal parameters.

Benefits:

- Faster search
- Better features
- Reduced computational cost

---

## Slide 8 – LBP Optimization Results

Use:

lbp_param_search.png

Explain:

- Radius selection
- Number of neighbors
- Best configuration

---

## Slide 9 – LBP Comparison

Use:

lbp_by_method.png

Discuss:

- Performance differences
- Selected method

---

## Slide 10 – DWT Optimization

Use:

dwt_param_search.png

Explain:

- Tested wavelets
- Search process
- Optimal configuration

---

## Slide 11 – DWT Wavelet Comparison

Use:

dwt_wavelet_comparison.png

Explain:

- Which wavelet performed best
- Why it was selected

---

## Slide 12 – Phase 1 Summary

Use:

phase1_summary.png

Present table:

| Feature | Best Parameters |

|----------|----------------|

| LBP | ... |

| DWT | ... |

| HOG | ... |

| Statistics | ... |

Conclude:

Optimized features are retained for benchmarking.

---

# PHASE 2

# Machine Learning Benchmark

---

## Slide 13 – Feature Sets Evaluated

Present:

A = Statistics

B = LBP

C = DWT

D = HOG

Combinations:

- A+B
- B+C
- A+B+C
- Full Feature Set

Explain rationale.

---

## Slide 14 – Classifiers Compared

Present:

- SVM
- KNN
- Random Forest
- XGBoost
- LightGBM
- Logistic Regression
- MLP
- ExtraTrees

Explain briefly.

---

## Slide 15 – Evaluation Metrics

Present:

- Accuracy
- Precision
- Recall
- F1 Score
- Macro F1
- Cross Validation

Include formulas only if space allows.

---

## Slide 16 – Individual Feature Results

Use:

p2_A_—_Stats.png

p2_B_—_Tex(opt).png

p2_C_—_DWT(opt).png

p2_D_—_HOG.png

Main question:

Which feature family performs best individually?

---

## Slide 17 – Combined Feature Results

Use:

p2_ApB.png

p2_BpC(opt).png

p2_ApBpC(opt).png

p2_Full(opt).png

Main question:

Does feature fusion improve performance?

---

## Slide 18 – Global Ranking

Use:

p2_leaderboard.png

Show:

Top configurations.

Highlight:

Best classifier.

Best feature set.

---

## Slide 19 – Feature Importance Analysis

Use:

rf_branch_importance.png

Discuss:

- Most influential feature families.
- Relative contribution of descriptors.

---

## Slide 20 – Feature Space Visualization

Use:

tsne_best_feature_set.png

Explain:

- Cluster separation.
- Class compactness.
- Discriminative power of features.

---

## Slide 21 – Stability Analysis

Use:

seed_stability_3seeds.csv

Present:

- Mean performance
- Standard deviation

Discuss:

Robustness across runs.

---

## Slide 22 – Statistical Validation

Present:

- Cross-validation results
- Confidence intervals
- Statistical significance tests

Goal:

Verify that improvements are meaningful.

---

## Slide 23 – Best Model

Present:

Final selected pipeline.

Include:

- Feature set
- Classifier
- Accuracy
- Precision
- Recall
- F1 Score

This is the key result slide.

---

## Slide 24 – Discussion

Discuss:

- Why optimized features help.
- Why some classifiers outperform others.
- Practical implications.

Focus on insights.

---

## Slide 25 – Conclusion and Future Work

### Main Contributions

- Feature optimization framework
- Comprehensive benchmark
- Statistical validation
- Best-performing pipeline

### Future Work

- CNNs
- Vision Transformers
- Hybrid ML/DL systems
- Explainable AI

Final takeaway:

Traditional machine learning combined with optimized handcrafted features can achieve strong performance for brain tumor MRI classification.

---

# Design Guidelines

Theme:

Modern minimalist 

Avoid:

- Bright colors

Each slide should contain:

- Title
- Key visual.

Every result slide must include a short interpretation section explaining the significance of the result.