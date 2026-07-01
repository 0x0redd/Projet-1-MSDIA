
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