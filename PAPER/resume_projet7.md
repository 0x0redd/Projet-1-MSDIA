# Résumé complet — Projet 7 : Classification ML

## 1. Présentation générale du projet

Le **Projet 7** porte sur la **classification d’images par machine learning classique**. L’objectif principal est de construire un système capable de reconnaître automatiquement des images en utilisant trois algorithmes traditionnels :

- **SVM** (*Support Vector Machine*)
- **KNN** (*K-Nearest Neighbors*)
- **Random Forest**

Le travail demandé comprend aussi une **analyse des performances** ainsi qu’une **optimisation des features**. Cela signifie que le projet ne se limite pas à entraîner des modèles, mais cherche aussi à comprendre **pourquoi un modèle fonctionne mieux qu’un autre**, dans quelles conditions, et comment améliorer les résultats à l’aide d’une meilleure représentation des images.

---

## 2. Intérêt du projet

Ce projet est important car il permet d’étudier la vision par ordinateur **avant l’ère du deep learning généralisé**. Aujourd’hui, les CNN dominent souvent la classification d’images, mais les méthodes classiques gardent un intérêt pédagogique et scientifique :

- elles aident à comprendre les bases de la classification ;
- elles montrent le rôle central de la **représentation des données** ;
- elles permettent de comparer des approches simples, rapides et interprétables ;
- elles sont particulièrement utiles lorsque le volume de données est limité.

Dans ce type de projet, la qualité des **features extraites** joue un rôle fondamental. Contrairement au deep learning qui apprend lui-même des représentations, les approches classiques dépendent fortement du choix des descripteurs visuels.

---

## 3. Ce que demande exactement le projet

Le sujet demande de mettre en place une chaîne complète de classification d’images comprenant :

1. **Préparation des données**
2. **Extraction ou construction de features**
3. **Entraînement de trois classifieurs** : SVM, KNN et Random Forest
4. **Évaluation comparative**
5. **Optimisation des features** et éventuellement des hyperparamètres

L’idée générale est donc de comparer plusieurs modèles classiques sur un même problème d’image et d’analyser leurs différences en termes de :

- précision ;
- robustesse ;
- temps d’exécution ;
- sensibilité à la taille du dataset ;
- sensibilité à la qualité des caractéristiques extraites.

---

## 4. Résumé de l’article de référence

L’article de référence cité dans le projet s’intitule :

**“Comparative Analysis of Image Classification Algorithms Based on Traditional Machine Learning and Deep Learning”**

Cet article compare principalement **SVM** (machine learning classique) et **CNN** (deep learning) pour la classification d’images. Son objectif est de montrer que les performances dépendent fortement de la **taille du dataset**, du **type d’images** et du **coût de calcul**.

### Idée principale de l’article

Les auteurs montrent qu’il n’existe pas un unique modèle meilleur dans tous les cas :

- sur des **petits jeux de données**, une méthode classique comme **SVM** peut être très compétitive, voire meilleure ;
- sur des **grands jeux de données**, les modèles profonds comme **CNN** prennent généralement l’avantage.

Ainsi, le choix de l’algorithme dépend fortement du contexte expérimental.

---

## 5. Partie théorique résumée

### 5.1 Machine Learning classique

Le machine learning classique repose sur une idée simple : les images sont transformées en vecteurs numériques, puis un classifieur apprend à séparer les différentes classes.

Cela suppose deux éléments essentiels :

- une **bonne représentation des images** ;
- un **algorithme de décision** capable de séparer les classes.

Ces approches sont souvent dites “shallow” car elles ne construisent pas automatiquement des représentations hiérarchiques complexes comme le deep learning.

### 5.2 Deep Learning

Le deep learning apprend automatiquement des représentations de plus en plus abstraites grâce à plusieurs couches successives. Dans le cas des images, les CNN apprennent progressivement :

- des contours et motifs simples ;
- des textures ou formes intermédiaires ;
- des structures plus globales ;
- puis des catégories complètes.

L’article insiste sur le fait que le deep learning est particulièrement performant lorsque les données sont nombreuses.

---

## 6. Résumé des algorithmes importants pour ton projet

### 6.1 SVM

Le **Support Vector Machine** cherche une frontière de séparation optimale entre les classes. Son principe est de maximiser la **marge** entre les exemples de différentes catégories.

#### Points forts

- très bon pouvoir de généralisation ;
- efficace sur des datasets petits ou moyens ;
- performant avec des features bien construites ;
- peut traiter des cas non linéaires grâce aux **fonctions noyau**.

#### Limites

- sensible au choix du noyau et des hyperparamètres ;
- moins pratique sur des datasets très grands ;
- dépend fortement de la qualité des features.

L’article rappelle trois familles de noyaux :

- **polynomial** ;
- **RBF** ;
- **sigmoid**.

Le noyau RBF est souvent l’un des plus utilisés en pratique.

### 6.2 KNN

Le **K-Nearest Neighbors** classe une image selon les classes de ses plus proches voisins dans l’espace des features.

#### Points forts

- très simple à comprendre et à implémenter ;
- pas de vrai apprentissage complexe ;
- bon baseline pour comparer les autres modèles.

#### Limites

- coûteux au moment de la prédiction ;
- sensible au bruit et au choix de **k** ;
- dépend énormément de la normalisation et de la qualité des features.

Même si l’article de référence ne développe pas KNN comme modèle principal, le sujet du projet te demande explicitement de l’intégrer dans la comparaison.

### 6.3 Random Forest

Le **Random Forest** est un ensemble d’arbres de décision. Chaque arbre vote pour une classe, puis la forêt choisit la classe majoritaire.

#### Points forts

- robuste et relativement simple à utiliser ;
- moins sensible au surapprentissage qu’un arbre unique ;
- peut bien fonctionner sur des features variées.

#### Limites

- moins adapté que SVM quand les frontières de décision sont très fines dans un espace de features bien structuré ;
- peut devenir volumineux si le nombre d’arbres est grand ;
- n’exploite pas directement la structure spatiale brute des images.

Là aussi, ce modèle ne constitue pas le cœur de l’article de référence, mais il est indispensable dans la réalisation du projet.

---

## 7. Résultats expérimentaux principaux de l’article

Les auteurs ont comparé SVM et CNN sur deux cas :

### 7.1 Grand dataset : MNIST

Sur le dataset **MNIST** :

- **SVM** atteint une précision de **0.88** ;
- **CNN** atteint une précision de **0.98**.

Concernant le temps d’exécution :

- **SVM** prend **27.6 minutes** ;
- **CNN** prend **23.2 minutes**.

### Interprétation

Sur un grand nombre d’images, le CNN surpasse clairement le SVM en précision. Cela confirme que les méthodes profondes deviennent très fortes quand elles disposent d’assez de données pour apprendre automatiquement les bonnes représentations.

### 7.2 Petit dataset : COREL1000

Sur le dataset **COREL1000** :

- **SVM** obtient **0.86** ;
- **CNN** obtient **0.83**.

Temps d’exécution :

- **SVM** : **1.02 minute** ;
- **CNN** : **2.01 minutes**.

### Interprétation

Quand le dataset est plus petit, le SVM devient plus intéressant :

- meilleure précision que le CNN ;
- temps d’exécution plus faible.

Cela soutient l’idée que les méthodes classiques restent très pertinentes dans des contextes à données limitées.

---

## 8. Influence de la taille des images

L’article étudie aussi l’impact de la taille des images sur les performances.

Pour des images de tailles différentes :

- **64×64** : SVM = **0.62**, CNN = **0.71**
- **128×128** : SVM = **0.64**, CNN = **0.74**
- **256×256** : SVM = **0.61**, CNN = **0.95**

### Interprétation

Quand la résolution augmente, le CNN profite beaucoup plus de l’information visuelle disponible. Le SVM, lui, n’améliore pas forcément ses performances, car il dépend d’une représentation déjà fixée. Cela montre que la richesse de l’image profite surtout aux méthodes capables d’apprendre elles-mêmes des caractéristiques complexes.

---

## 9. Influence du nombre de classes

Les auteurs ont aussi comparé les performances selon le nombre de catégories d’images :

- 2 classes
- 4 classes
- 6 classes

Dans ce cadre, l’article indique que :

- les précisions de SVM et CNN restent globalement proches ;
- le **temps de test du SVM** est plus court ;
- le **temps de test du CNN** est plus long.

### Interprétation

Quand la diversité des classes change mais que les autres conditions restent proches, les deux approches peuvent rester compétitives. Toutefois, en pratique, les méthodes classiques gardent souvent un avantage en simplicité et en coût de calcul sur des expériences modestes.

---

## 10. Ce qu’il faut retenir scientifiquement

L’idée scientifique centrale est la suivante :

> **Le meilleur classifieur dépend du dataset, de la taille des données, du nombre de classes, de la résolution des images et surtout de la qualité des features.**

Le projet 7 s’inscrit parfaitement dans cette logique, sauf qu’au lieu de comparer **SVM contre CNN**, il te demande d’étudier **SVM, KNN et Random Forest**, donc trois approches de machine learning classique.

Autrement dit, l’article fournit un **cadre méthodologique**, mais ton travail doit aller plus loin sur les modèles classiques.

---

## 11. Comment adapter cet article à ton projet

Pour ton projet, la démarche la plus cohérente serait :

### Étape 1 — Choisir un dataset

Par exemple :

- CIFAR-10 ;
- un dataset thématique ;
- ou un jeu de données plus ciblé selon ton sujet.

### Étape 2 — Prétraiter les images

- redimensionnement ;
- normalisation ;
- éventuellement conversion en niveaux de gris ;
- séparation train / test.

### Étape 3 — Extraire des features

Comme les modèles classiques n’apprennent pas directement depuis l’image brute aussi bien qu’un CNN, il faut construire des caractéristiques pertinentes, par exemple :

- pixels bruts aplatis ;
- histogrammes de couleur ;
- HOG ;
- LBP ;
- SIFT / SURF / ORB ;
- PCA pour réduire la dimension.

### Étape 4 — Entraîner les modèles

- SVM
- KNN
- Random Forest

### Étape 5 — Comparer les performances

Utiliser plusieurs métriques :

- accuracy ;
- precision ;
- recall ;
- F1-score ;
- matrice de confusion ;
- temps d’entraînement ;
- temps d’inférence.

### Étape 6 — Optimiser

Deux axes d’optimisation sont essentiels :

1. **optimisation des features** ;
2. **optimisation des hyperparamètres**.

Exemples :

- SVM : `C`, noyau, `gamma`
- KNN : `k`, distance, pondération
- Random Forest : nombre d’arbres, profondeur maximale, nombre de variables par split

---

## 12. Message principal pour le rapport

Dans ton rapport, le message final peut être formulé ainsi :

- Les méthodes classiques restent très utiles pour la classification d’images.
- Leur performance dépend moins d’une architecture profonde que de la qualité des caractéristiques extraites.
- SVM est souvent très fort sur des représentations bien construites.
- KNN fournit une bonne base de comparaison mais devient sensible à la dimension et au bruit.
- Random Forest offre une solution robuste et stable.
- L’optimisation des features peut parfois avoir autant d’impact que le choix du classifieur lui-même.

---

## 13. Conclusion générale

Le **Projet 7** vise à faire comprendre les fondements de la **classification d’images par machine learning classique**. Il ne s’agit pas seulement d’appliquer trois algorithmes, mais de construire une vraie étude comparative autour de **SVM, KNN et Random Forest**.

L’article de référence montre clairement que la performance dépend du contexte :

- **sur de petits datasets**, les approches classiques comme SVM peuvent être excellentes ;
- **sur de grands datasets**, les méthodes profondes prennent souvent l’avantage ;
- la **taille des images** et la **nature des données** influencent fortement les résultats.

Pour ton projet, la vraie valeur scientifique viendra de :

- la qualité du pipeline de prétraitement ;
- le choix des descripteurs visuels ;
- la comparaison rigoureuse entre les trois algorithmes ;
- l’analyse critique des résultats.

En résumé, ce projet est une excellente occasion de montrer que la classification d’images ne dépend pas uniquement du deep learning, mais aussi d’une compréhension solide des **features**, des **algorithmes classiques**, et de la **méthodologie expérimentale**.
