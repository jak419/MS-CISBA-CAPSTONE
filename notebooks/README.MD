# 🧠 Model Development and Experimentation

## Overview

This directory contains the **exploratory analysis and model experimentation notebook** used during the first phase of the fraud detection project. The notebook served as a sandbox environment for data exploration, visualization, feature selection, and model comparison.

---

## 🔍 Contents

### 📓 `Model Training and Tuning.ipynb`

This notebook documents the following steps in detail:

1. **Data Sampling and Preparation**
   - A stratified random sample of 600,000 transactions was drawn from the original dataset to reduce processing time while maintaining class balance.
   - Irrelevant or redundant features (`nameOrig`, `nameDest`, `isFlaggedFraud`) were dropped or masked.

2. **Exploratory Data Analysis (EDA)**
   - Analyzed class imbalance and the distribution of transaction types.
   - Identified that **fraudulent transactions are most common in `TRANSFER` and `CASH_OUT` types**.
   - Used visualizations to understand relationships between balance changes, amounts, and fraud outcomes.

3. **Model Training and Evaluation**
   - Compared three different classifiers:
     - **Logistic Regression**
     - **K-Nearest Neighbors (KNN)**
     - **Decision Tree Classifier**
     - **Deep Neural Network**
   - Evaluated each model based on:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1-Score**
     - **ROC AUC**

4. **Model Selection**
   - The **Decision Tree Classifier** outperformed other models with:
     - Accuracy: 0.9995
     - Recall: 0.71
     - F1-Score: 0.82
   - Another plus of why **Decision Tree Classifier** was selected for deployment due to:
     - Simpler interpretability
     - Faster inference time
     - Seamless integration with Scikit-learn pipelines

5. **Hyperparameter Tuning**
   - The Decision Tree was tuned with:
     - `max_depth = 10`
     - `min_samples_split = 2`
     - `min_samples_leaf = 2`

---

## 🛠 Purpose

The insights, metrics, and model results from this notebook were critical in guiding the architecture of the production pipeline implemented in the `fraud_detection/` directory.

> This notebook phase represents the **research and development backbone** of the entire fraud detection system.

---

## 📎 Dependencies

Ensure you have the following installed to run the notebook:

```bash
pip install pandas scikit-learn matplotlib seaborn notebook