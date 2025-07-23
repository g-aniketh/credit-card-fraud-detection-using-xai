# Credit Card Fraud Detection Using Ensemble Learning and XAI

This repository implements a machine learning pipeline to detect credit card fraud using several ensemble classifiers combined with Explainable Artificial Intelligence (XAI) techniques. Designed to handle the rare-event nature of fraudulent transactions, the system applies SMOTE to balance the dataset and trains multiple models including Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, and XGBoost.

To meet the critical need for transparency in financial applications, model interpretability is provided through SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-Agnostic Explanations). The approach ensures high predictive accuracy while offering both global and local explanations for model decisions.

---

## Key Features

- **Robust data preprocessing:** feature scaling, train/test split with stratified sampling, and class imbalance handling using SMOTE.
- **Multiple ensemble models:** Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, and XGBoost trained under the same settings for performance comparison.
- **Comprehensive evaluation metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, AUPRC, Log Loss.
- **Visualization:** Confusion matrices, ROC curves, Precision-Recall curves for all models.
- **Explainability:** Global and local explanations using SHAP values and LIME on the best-performing model (XGBoost).
- **Reproducible and modular codebase** supporting straightforward dataset download via Kaggle API.

---

## Models & Algorithms Used

- **Logistic Regression:** Baseline linear model with liblinear solver.
- **Random Forest:** Ensemble of decision trees with balanced subsampling to reduce bias.
- **Gradient Boosting:** Boosted trees with controlled depth and learning rate.
- **AdaBoost:** Boosting ensemble using SAMME algorithm.
- **XGBoost:** Advanced gradient boosting optimized for accuracy and speed.

- **Data Preprocessing:**
  - **StandardScaler:** Normalizes features to zero mean and unit variance.
  - **SMOTE:** Synthesizes minority class samples to alleviate severe class imbalance.
- **Explainability:**
  - **SHAP:** Provides both global feature importance plots (bar and dot) and local force plots explaining individual predictions.
  - **LIME:** Generates local surrogate models, explaining predictions for specific fraud and non-fraud cases with intuitive HTML visualizations.

---

## Dataset

- Credit Card Fraud Detection dataset from Kaggle: `"mlg-ulb/creditcardfraud"`.
- Dataset contains 284,807 transactions with only 492 labeled as fraudulent (~0.172%).
- Features include anonymized PCA components (V1–V28), plus `Time` and `Amount`.

---

## Authors

- Vivin Chandrra Paasam
- Topalle Siddha Sankalp
- Aniketh Gandhari
- Syeda Sayeeda Farhath
- Dhiren Rao B
- Y. Vijayalata
