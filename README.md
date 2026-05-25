
# Fraud Detection Project

This repository contains a complete fraud detection workflow implemented in the Jupyter notebook `Final_0.ipynb`. The project uses a credit card transaction dataset from Kaggle and demonstrates data cleaning, feature engineering, class imbalance handling, model training, and performance evaluation for fraud classification.

## Dataset

- File: `fraudTrain_noisy_dataset.csv`
- Kaggle link: https://www.kaggle.com/datasets/kartik2112/fraud-detection
- Description: Each row represents a single transaction and includes customer demographics, merchant details, location data, transaction amount, and a fraud label (`is_fraud`).

## Project Overview

This notebook implements a full end-to-end fraud detection pipeline with the following steps:

1. Data loading and initial inspection
2. Data cleaning and preprocessing
3. Exploratory data analysis (EDA)
4. Feature engineering and transformation
5. Train/validation/test split
6. Handling class imbalance using SMOTE and SMOTETomek
7. Model training using Random Forest and LightGBM
8. Performance evaluation with ROC-AUC, PR-AUC, precision, recall, and confusion matrices

## Notebook Structure

### 1. Data Cleaning

- Load raw CSV data
- Remove duplicate rows
- Convert date columns to datetime
- Impute missing transaction amounts by category median
- Standardize categorical text fields
- Drop unnecessary identifiers such as `Unnamed: 0`, `first`, `last`, `street`, and `trans_num`

### 2. Exploratory Data Analysis

- Pivot tables to compare average amounts by category and fraud label
- KDE plots for legitimate vs fraudulent amount distributions
- Bar plots for categories with the highest fraud rates
- Fraud rate by hour of day
- Visualize outliers with boxplots and retain them because they may represent important fraud signals

### 3. Feature Engineering

- Extract temporal features: transaction hour, weekday, weekend indicator, night indicator
- Apply cyclical encoding to hour and day-of-week features using sine/cosine transforms
- Derive customer age from date of birth and transaction datetime
- Compute geographic distance between cardholder and merchant using Haversine distance
- Add log-transformed amount and log-distance features
- Create population density bins for `city_pop`
- Build interaction feature `amt_distance_interaction`
- Drop redundant or potentially leaky columns such as `dob`, `unix_time`, `zip`, `city`, `merch_lat`, `merch_long`, `merchant`, and `job`

### 4. Data Preparation

- Split data into train/validation/test sets with stratification on fraud label
- Encode categorical variables (`gender`, `category`, `state`) using `LabelEncoder`
- Select final modeling features with numeric and categorical pools
- Use mutual information, ANOVA F-test, and chi-square tests to rank and select the most informative features

### 5. Imbalance Handling

- Explore the extreme class imbalance present in `is_fraud`
- Apply SMOTE to the training set only, preserving the validation/test distribution
- Evaluate risk of synthetic oversampling and compare with class-weighted approaches

### 6. Model Training

The notebook trains and compares multiple models:

- Random Forest on imbalanced data without class weights
- Random Forest on imbalanced data with class weights
- Random Forest on SMOTE-balanced data without class weights
- Random Forest on SMOTE-balanced data with class weights
- LightGBM with class weights
- LightGBM with SMOTETomek balancing

### 7. Evaluation

- Evaluate each model on the validation set using:
  - ROC-AUC
  - PR-AUC
  - Classification report (precision, recall, f1-score)
  - Confusion matrix
- Visualize ROC and precision-recall curves
- Compare model performance across strategies
- Run a final test-set evaluation for the selected best model

## Key Lessons

- Fraud detection is a highly imbalanced classification problem where precision-recall performance is more meaningful than raw accuracy.
- Feature engineering is critical: time features, geographic distance, log transforms, and feature interactions help capture fraud signals.
- Class imbalance should be handled carefully: oversampling is applied only on training data, while validation and test sets remain untouched.
- Comparing multiple imbalance strategies, such as class weighting and SMOTE, helps identify the best approach for the data.

## Requirements

The notebook uses standard data science libraries and machine learning tools:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- lightgbm

## How to Use

1. Open `Final_0.ipynb` in Jupyter Lab or Jupyter Notebook.
2. Make sure `fraudTrain_noisy_dataset.csv` is in the same folder.
3. Install required packages if needed.
4. Run notebook cells sequentially from top to bottom.

## Notes

- The notebook contains analysis, visualizations, and commentary that explain each step.
- The final best model is selected from a comparison of Random Forest and LightGBM strategies.
- The evaluation focuses on real-world fraud detection needs by using PR-AUC and a held-out test set.
