# PrediX

### Multi-Domain Predictive Intelligence Using Machine Learning

PrediX is a Machine Learning project that explores predictive modeling and
classification across three real-world datasets from the UCI Machine Learning
Repository.

The project applies a complete Machine Learning workflow including data
understanding, exploratory data analysis, preprocessing, feature engineering,
model training, evaluation, and comparison.

---

## Project Overview

PrediX focuses on three different real-world problem domains:

| Dataset | UCI ID | Problem Type | Domain |
|---|---:|---|---|
| Combined Cycle Power Plant | 294 | Regression | Energy |
| Steel Plates Faults | 198 | Classification | Manufacturing |
| Human Activity Kinematic Profiling | 240 | Classification | Human Activity |

The objective is to study how different Machine Learning algorithms perform
across different datasets and problem types.

---

## Objectives

- Perform Exploratory Data Analysis (EDA)
- Identify patterns, relationships, and distributions in the datasets
- Handle missing values, duplicates, and outliers
- Perform appropriate feature preprocessing
- Apply feature engineering
- Build and compare multiple Machine Learning models
- Evaluate models using appropriate performance metrics
- Tune selected models using hyperparameter optimization
- Develop a reproducible Machine Learning pipeline

---

#  Datasets

## 1. Combined Cycle Power Plant

**UCI ID:** 294  
**Problem Type:** Regression

The dataset is used to predict the electrical energy output of a combined
cycle power plant based on available operating and environmental features.

### Regression Target

**Net hourly electrical energy output**

### Review 1 Algorithms

The following regression algorithms are implemented:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Polynomial Regression
6. Decision Tree Regressor
7. Random Forest Regressor
8. Gradient Boosting Regressor
9. Support Vector Regressor (SVR)
10. K-Nearest Neighbors Regressor

### Evaluation Metrics

- R² Score
- RMSE
- MAE
- 5-Fold Cross-Validated R² for the two best-performing models

---

#  2. Steel Plates Faults

**UCI ID:** 198  
**Problem Type:** Classification

This dataset is used to classify steel plates based on their measured
characteristics and identify the corresponding fault categories.

### Review 1 Algorithms — Classification Part A

1. Logistic Regression
2. K-Nearest Neighbors
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. Support Vector Machine (SVC)

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- Weighted F1-Score
- Confusion Matrix

---

#  Review 1 Workflow

The Review 1 implementation follows the workflow:

```text
Dataset
   ↓
Data Loading
   ↓
Data Audit
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Encoding & Scaling
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
