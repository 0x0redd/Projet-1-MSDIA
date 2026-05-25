Skill Guide: Writing a Good LaTeX Research Article
1. Goal
This guide explains how to write a clean, professional, and publication-ready research article using LaTeX. It is especially useful for scientific papers in computer science, artificial intelligence, machine learning, medical imaging, and engineering.
---
2. Recommended Project Structure
Use a clean folder structure:
```text
paper_project/
│
├── main.tex
├── references.bib
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methodology.tex
│   ├── 04_results.tex
│   ├── 05_discussion.tex
│   └── 06_conclusion.tex
│
├── figures/
│   ├── architecture.png
│   ├── confusion_matrix.png
│   └── learning_curves.png
│
├── tables/
│   └── model_comparison.tex
│
└── styles/
    └── custom_commands.tex
```
This structure keeps the article organized and easy to maintain.
---
3. Basic LaTeX Article Template
```latex
\documentclass[conference]{IEEEtran}

\usepackage{graphicx}
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{float}
\usepackage{cite}
\usepackage{hyperref}
\usepackage{algorithm}
\usepackage{algorithmic}

\begin{document}

\title{Your Paper Title Here}

\author{
\IEEEauthorblockN{Author Name}
\IEEEauthorblockA{
Department / Laboratory \\
University Name \\
Email address
}
}

\maketitle

\begin{abstract}
Write a concise abstract explaining the problem, method, dataset, results, and contribution.
\end{abstract}

\begin{IEEEkeywords}
Brain tumor detection, MRI, machine learning, deep learning, ensemble learning, medical imaging.
\end{IEEEkeywords}

\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_methodology}
\input{sections/04_results}
\input{sections/05_discussion}
\input{sections/06_conclusion}

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
```
---
4. Main Sections of a Good Research Article
4.1 Abstract
The abstract should include:
Problem statement
Motivation
Proposed method
Dataset used
Main results
Contribution
Example structure:
```text
Brain tumor detection from MRI images is a challenging task due to tumor variability and image complexity. This paper proposes... The method is evaluated on... Experimental results show... The main contribution is...
```
---
4.2 Introduction
The introduction should answer:
What is the problem?
Why is it important?
What are the limitations of existing methods?
What does your paper propose?
What are your contributions?
Useful contribution format:
```latex
The main contributions of this work are summarized as follows:
\begin{itemize}
    \item We propose a hybrid machine learning and deep learning framework for brain tumor classification.
    \item We compare classical classifiers, including SVM, KNN, and Random Forest, with ensemble models such as XGBoost, AdaBoost, and Bagging.
    \item We evaluate the proposed framework using accuracy, precision, recall, F1-score, and confusion matrix analysis.
\end{itemize}
```
---
4.3 Related Work
The related work section should not only summarize papers. It should compare them critically.
Good structure:
```text
Classical machine learning methods
Deep learning methods
Hybrid and ensemble learning methods
Research gap
```
Example sentence:
```latex
Although several studies have achieved high classification accuracy using convolutional neural networks, many of them lack direct comparison with classical machine learning models and ensemble-based classifiers.
```
---
4.4 Methodology
The methodology must explain your full pipeline clearly.
Recommended subsections:
```latex
\section{Methodology}
\subsection{Dataset Description}
\subsection{Preprocessing}
\subsection{Feature Extraction}
\subsection{Classical Machine Learning Models}
\subsection{Ensemble Learning Models}
\subsection{Evaluation Metrics}
```
---
5. Writing Mathematical Equations
Inline Equation
```latex
The accuracy is calculated as $Acc = \frac{TP + TN}{TP + TN + FP + FN}$.
```
Display Equation
```latex
\begin{equation}
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\end{equation}
```
Common Metrics
```latex
\begin{equation}
Precision = \frac{TP}{TP + FP}
\end{equation}

\begin{equation}
Recall = \frac{TP}{TP + FN}
\end{equation}

\begin{equation}
F1\text{-}score = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\end{equation}
```
---
6. Adding Figures
Place all figures in the `figures/` folder.
```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\linewidth]{figures/architecture.png}
    \caption{Proposed model architecture.}
    \label{fig:architecture}
\end{figure}
```
Reference the figure:
```latex
As shown in Fig.~\ref{fig:architecture}, the proposed framework consists of preprocessing, feature extraction, classification, and evaluation stages.
```
Important rules:
Every figure must be referenced in the text.
Every figure must have a meaningful caption.
Do not use blurry screenshots.
Use high-resolution PNG, PDF, or SVG figures.
---
7. Creating Professional Tables
Use `booktabs` for clean tables.
```latex
\begin{table}[H]
\centering
\caption{Performance comparison of machine learning models.}
\label{tab:model_comparison}
\begin{tabular}{lcccc}
\toprule
Model & Accuracy & Precision & Recall & F1-score \\
\midrule
KNN & 91.20 & 90.80 & 91.00 & 90.90 \\
SVM & 94.50 & 94.20 & 94.10 & 94.15 \\
Random Forest & 95.30 & 95.00 & 95.20 & 95.10 \\
XGBoost & 96.80 & 96.50 & 96.70 & 96.60 \\
\bottomrule
\end{tabular}
\end{table}
```
Reference the table:
```latex
Table~\ref{tab:model_comparison} presents the comparison between classical and ensemble learning models.
```
---
8. Writing Algorithms
```latex
\begin{algorithm}[H]
\caption{Proposed Brain Tumor Classification Pipeline}
\begin{algorithmic}[1]
\STATE Load MRI dataset
\STATE Apply preprocessing and normalization
\STATE Extract features from images
\STATE Train classical and ensemble learning models
\STATE Evaluate models using standard metrics
\STATE Select the best-performing model
\end{algorithmic}
\end{algorithm}
```
---
9. Bibliography with BibTeX
In `main.tex`:
```latex
\bibliographystyle{IEEEtran}
\bibliography{references}
```
In `references.bib`:
```bibtex
@article{author2025brain,
  author  = {Author, First and Author, Second},
  title   = {Brain Tumor Detection Using Machine Learning},
  journal = {Journal Name},
  year    = {2025},
  volume  = {10},
  number  = {2},
  pages   = {100--115},
  doi     = {10.xxxx/example}
}
```
Cite inside the text:
```latex
Recent studies have shown strong performance for deep learning models in MRI-based tumor classification~\cite{author2025brain}.
```
---
10. Useful Packages
```latex
\usepackage{graphicx}      % Images
\usepackage{amsmath}       % Math equations
\usepackage{amssymb}       % Math symbols
\usepackage{booktabs}      % Professional tables
\usepackage{multirow}      % Multi-row tables
\usepackage{float}         % Figure placement
\usepackage{cite}          % IEEE citations
\usepackage{hyperref}      % Clickable links
\usepackage{algorithm}     % Algorithms
\usepackage{algorithmic}   % Algorithm steps
\usepackage{subcaption}    % Subfigures
\usepackage{xcolor}        % Colors
```
---
11. Custom Commands
Create reusable commands for repeated terms.
```latex
\newcommand{\acc}{\textit{Accuracy}}
\newcommand{\fscore}{F1\text{-}score}
\newcommand{\mri}{MRI}
```
Use them like this:
```latex
The proposed model achieved high \acc{} and \fscore{} on the \mri{} dataset.
```
---
12. Good Academic Writing Rules
Use formal scientific language:
Bad:
```text
The model is very good and gives amazing results.
```
Good:
```text
The proposed model achieved superior classification performance compared with baseline methods across all evaluation metrics.
```
Avoid:
“very good”
“amazing”
“perfect”
“we can see that”
unsupported claims
Use:
“demonstrates”
“indicates”
“suggests”
“outperforms”
“achieves”
“is consistent with”
---
13. Common Mistakes to Avoid
Using figures without references in the text
Using tables without captions
Writing long paragraphs without structure
Mixing citation styles
Reporting only accuracy
Not explaining the dataset split
Not mentioning hyperparameters
Not comparing with previous work
Not discussing limitations
---
14. Checklist Before Submission
Before submitting your paper, verify:
[ ] The title is clear and specific.
[ ] The abstract contains problem, method, results, and contribution.
[ ] All figures are referenced.
[ ] All tables are referenced.
[ ] All equations are numbered if important.
[ ] All citations are included in the `.bib` file.
[ ] The methodology is reproducible.
[ ] The results include multiple metrics.
[ ] The discussion explains why the results matter.
[ ] The conclusion summarizes contributions and limitations.
---
15. Recommended Paper Writing Workflow
Prepare the dataset and results.
Create tables and plots.
Write the methodology first.
Write the results section.
Write the introduction and related work.
Write the abstract last.
Revise the paper for clarity and consistency.
Check references and formatting.
---
16. Best Practice for ML Papers
For machine learning and deep learning papers, always include:
Dataset description
Preprocessing steps
Train/test split
Model hyperparameters
Evaluation metrics
Confusion matrix
Comparison table
Discussion of limitations
Reproducibility details
---
17. Example Methodology Paragraph
```latex
The proposed framework consists of four main stages: preprocessing, feature extraction, classification, and evaluation. First, MRI images are resized and normalized to ensure input consistency. Then, handcrafted and deep features are extracted to represent discriminative tumor patterns. Several machine learning classifiers, including SVM, KNN, Random Forest, XGBoost, AdaBoost, and Bagging, are trained and evaluated. Finally, model performance is assessed using accuracy, precision, recall, F1-score, and confusion matrix analysis.
```
---
18. Example Contribution Paragraph
```latex
The main contribution of this work is the development of a comparative and ensemble-based framework for brain tumor classification using MRI images. Unlike studies that focus on a single model, this work evaluates classical machine learning models, tree-based ensemble methods, and hybrid classification strategies under the same experimental setting. This allows a fair comparison of model performance and provides insights into the suitability of each approach for medical image classification.
```
---
19. Final Advice
A good LaTeX article is not only about formatting. It must be:
Scientifically clear
Well structured
Reproducible
Properly cited
Supported by strong tables and figures
Written in formal academic language
Focus on clarity before complexity.