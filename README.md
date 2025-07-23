# Credit Card Fraud Detection Using XAI

This repository contains an end-to-end machine learning pipeline for credit card fraud detection combining ensemble models with Explainable Artificial Intelligence (XAI) techniques. The project addresses the challenge of highly imbalanced fraud data by applying SMOTE for oversampling and trains multiple models including Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, and XGBoost.

XGBoost emerged as the best-performing model, and its predictions are explained using two popular XAI frameworks:

- **SHAP (SHapley Additive exPlanations)** for both global and local feature importance insights.
- **LIME (Local Interpretable Model-Agnostic Explanations)** for instance-level explanations that clarify individual predictions.

## Features

- Data preprocessing: stratified train/test split, feature scaling, and handling class imbalance with SMOTE.
- Model training and evaluation with robust metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC, AUPRC, and Log Loss.
- Visualization of confusion matrices, ROC curves, and precision-recall curves for clear model comparison.
- Explainability to build trust and transparency necessary for financial and regulatory use cases.
