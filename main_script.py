################################################################################
# Phase 0: Import Libraries
################################################################################
print("Phase 0: Importing Libraries...")
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt

import kagglehub

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, precision_recall_curve, auc,
    log_loss                                         
)

# XAI Libraries
import shap
import lime
import lime.lime_tabular

print("✅ Libraries imported successfully.")
print("-" * 60)

################################################################################
# Phase 1: Load Data and Initial Exploration
################################################################################
print("Phase 1: Loading Data and Initial Exploration...")

print("Attempting to download dataset from Kaggle Hub...")
try:
    dataset_path_dir = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_file_name = "creditcard.csv"
    csv_file_path = os.path.join(dataset_path_dir, csv_file_name)
    print(f"Dataset downloaded to directory: {dataset_path_dir}")
    print(f"Full path to CSV: {csv_file_path}")

except Exception as e:
    print(f"Error downloading dataset from Kaggle Hub: {e}")
    print("Please ensure you have set up your Kaggle API token (~/.kaggle/kaggle.json).")
    print("You might need to run 'pip install kaggle kagglehub'.")
    exit()

if not os.path.exists(csv_file_path):
    print(f"Error: File not found at {csv_file_path} after Kaggle Hub download attempt.")
    print("Please check the download path and file name.")
    exit()

try:
    df = pd.read_csv(csv_file_path)
except UnicodeDecodeError:
    print("UTF-8 decoding failed, trying ISO-8859-1...")
    df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

print("✅ Dataset loaded successfully.")
print("\nFirst 5 records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe())

print("\nClass Distribution (0: Non-Fraud, 1: Fraud):")
print(df['Class'].value_counts())
print(f"Fraudulent transactions: {df['Class'].value_counts()[1] / len(df) * 100:.2f}%")
print("-" * 60)

################################################################################
# Phase 2: Data Preprocessing
################################################################################
print("Phase 2: Data Preprocessing...")

# Define features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"Training class distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Test class distribution:\n{y_test.value_counts(normalize=True)}")

# --- Feature Scaling ---
print("\nScaling features ('Time', 'Amount', and V1-V28)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# --- Handling Class Imbalance using SMOTE ---
print("\nApplying SMOTE to the scaled training data...")
smote = SMOTE(random_state=42)
X_train_resampled_scaled, y_train_resampled = smote.fit_resample(X_train_scaled_df, y_train)

print("Class distribution in original scaled training data:")
print(y_train.value_counts())
print("Class distribution after SMOTE on scaled training data:")
print(y_train_resampled.value_counts())
print("✅ Data preprocessing complete.")
print("-" * 60)

################################################################################
# Phase 3: Model Training
################################################################################
print("Phase 3: Model Training...")

models = {}

# --- 1. Logistic Regression ---
print("\nTraining Logistic Regression model...")
start_time = time.time()
log_reg = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
log_reg.fit(X_train_resampled_scaled, y_train_resampled)
models['Logistic Regression'] = log_reg
print(f"✅ Logistic Regression trained in {time.time() - start_time:.2f} seconds.")

# --- 2. Random Forest ---
print("\nTraining Random Forest model...")
start_time = time.time()
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced_subsample')
rf_clf.fit(X_train_resampled_scaled, y_train_resampled)
models['Random Forest'] = rf_clf
print(f"✅ Random Forest trained in {time.time() - start_time:.2f} seconds.")

# --- 3. Gradient Boosting ---
print("\nTraining Gradient Boosting model...")
start_time = time.time()
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_clf.fit(X_train_resampled_scaled, y_train_resampled)
models['Gradient Boosting'] = gb_clf
print(f"✅ Gradient Boosting trained in {time.time() - start_time:.2f} seconds.")

# --- 4. AdaBoost --- 
print("\nTraining AdaBoost model...")
start_time = time.time()
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME') 
ada_clf.fit(X_train_resampled_scaled, y_train_resampled)
models['AdaBoost'] = ada_clf
print(f"✅ AdaBoost trained in {time.time() - start_time:.2f} seconds.")

# --- 5. XGBoost --- 
print("\nTraining XGBoost model...")
start_time = time.time()
xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
use_label_encoder=False
xgb_clf.fit(X_train_resampled_scaled, y_train_resampled)
models['XGBoost'] = xgb_clf
print(f"✅ XGBoost trained in {time.time() - start_time:.2f} seconds.")
print("-" * 60)

################################################################################
# Phase 4: Model Evaluation
################################################################################
print("Phase 4: Model Evaluation...")

results = {}
if not os.path.exists("plots"):
    os.makedirs("plots")

def pr_auc_score(y_true, y_probas):
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probas)
    return auc(recall_vals, precision_vals)

for model_name, model in models.items():
    print(f"\n--- Evaluating: {model_name} ---")
    y_pred = model.predict(X_test_scaled_df)
    y_proba = model.predict_proba(X_test_scaled_df)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc_val = roc_auc_score(y_test, y_proba)
    auprc = pr_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)

    results[model_name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1, 'ROC-AUC': roc_auc_val, 'AUPRC': auprc, 'LogLoss': logloss}

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f} (Out of all transactions predicted as fraud, how many were actually fraud?)")
    print(f"Recall:    {rec:.4f} (Out of all actual fraud transactions, how many did the model correctly identify?)")
    print(f"F1-Score:  {f1:.4f} (Harmonic mean of Precision and Recall)")
    print(f"ROC-AUC:   {roc_auc_val:.4f}")
    print(f"AUPRC:     {auprc:.4f} (Recommended for imbalanced datasets)")
    print(f"LogLoss:   {logloss:.4f} (Lower is better)") 

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.savefig(f"plots/{model_name}_confusion_matrix.png")
    plt.close()

plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled_df)[:, 1]
    RocCurveDisplay.from_predictions(y_test, y_proba, name=model_name, ax=plt.gca())
plt.title("ROC Curve Comparison")
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.legend()
plt.savefig("plots/all_models_roc_curves.png")
plt.close()

plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled_df)[:, 1]
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba) # <<< MODIFIED >>>
    plt.plot(recall_vals, precision_vals, label=f'{model_name} (AUPRC = {pr_auc_score(y_test, y_proba):.4f})')

no_skill_auprc = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill_auprc, no_skill_auprc], 'k--', label=f'No Skill (AUPRC = {no_skill_auprc:.4f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.savefig("plots/all_models_pr_curves.png")
plt.close()

print("\n✅ Model evaluation complete. Plots saved in 'plots' directory.")
print("Evaluation Metrics Summary:")
results_df = pd.DataFrame(results).T
print(results_df[['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'AUPRC', 'LogLoss']])
print("-" * 60)


################################################################################
# Phase 5: Explainable AI (XAI)
################################################################################
print("Phase 5: Explainable AI (XAI)...")
chosen_model_name = 'XGBoost'
if chosen_model_name not in models:
    print(f"Error: Chosen model for XAI '{chosen_model_name}' not found in trained models. Defaulting to first model.")
    if models:
        chosen_model_name = list(models.keys())[0]
        chosen_model_for_xai = models[chosen_model_name]
        print(f"Using '{chosen_model_name}' for XAI instead.")
    else:
        print("Error: No models trained. Cannot proceed with XAI.")
        exit()
else:
    chosen_model_for_xai = models[chosen_model_name]


if not os.path.exists("xai_explanations"):
    os.makedirs("xai_explanations")

# --- 5.1 SHAP (SHapley Additive exPlanations) ---
print(f"\nRunning SHAP explanations for {chosen_model_name} model...")
# For tree-based models like Random Forest, Gradient Boosting, and XGBoost, TreeExplainer is efficient.
try:
    explainer_shap = shap.TreeExplainer(chosen_model_for_xai)
except Exception as e:
    print(f"Could not use TreeExplainer for {chosen_model_name}, possibly not a compatible tree model or XGBoost version issue: {e}")
    print("Attempting KernelExplainer (slower)...")
    if isinstance(X_train_resampled_scaled, pd.DataFrame):
        background_data = shap.kmeans(X_train_resampled_scaled, 100)
    else:
        background_data_df = pd.DataFrame(X_train_resampled_scaled, columns=X_train_scaled_df.columns)
        background_data = shap.kmeans(background_data_df, 100)

    explainer_shap = shap.KernelExplainer(chosen_model_for_xai.predict_proba, background_data)


X_test_sample_shap = X_test_scaled_df.sample(100, random_state=42)
shap_values = explainer_shap.shap_values(X_test_sample_shap)

if isinstance(shap_values, list):
    shap_values_for_summary = shap_values[1]
else:
    shap_values_for_summary = shap_values


print("Generating SHAP summary plot...")
shap.summary_plot(shap_values_for_summary, X_test_sample_shap, plot_type="bar", show=False)
plt.title(f"SHAP Feature Importance for {chosen_model_name} (Bar Plot)")
plt.tight_layout()
plt.savefig("xai_explanations/shap_summary_bar_plot.png")
plt.close()

shap.summary_plot(shap_values_for_summary, X_test_sample_shap, show=False)
plt.title(f"SHAP Feature Importance for {chosen_model_name} (Dot Plot)")
plt.tight_layout()
plt.savefig("xai_explanations/shap_summary_dot_plot.png")
plt.close()
print("✅ SHAP summary plots saved.")

# --- SHAP Force Plot for individual predictions ---
y_test_reset = y_test.reset_index(drop=True)
X_test_scaled_df_reset = X_test_scaled_df.reset_index(drop=True)

fraud_indices = y_test_reset[y_test_reset == 1].index
non_fraud_indices = y_test_reset[y_test_reset == 0].index

if len(fraud_indices) > 0 and len(non_fraud_indices) > 0:
    idx_fraud = fraud_indices[0]
    idx_non_fraud = non_fraud_indices[0]

    instance_fraud = X_test_scaled_df_reset.iloc[[idx_fraud]]
    instance_non_fraud = X_test_scaled_df_reset.iloc[[idx_non_fraud]]

    # Force plot for a fraudulent transaction
    print(f"Generating SHAP force plot for a fraudulent instance (index {idx_fraud})...")
    shap_values_fraud_instance_calc = explainer_shap.shap_values(instance_fraud)
    expected_value_calc = explainer_shap.expected_value

    if isinstance(shap_values_fraud_instance_calc, list):
        shap_values_fraud_instance_class1 = shap_values_fraud_instance_calc[1]
        if isinstance(expected_value_calc, list):
            expected_value_class1 = expected_value_calc[1]
        else:
            expected_value_class1 = expected_value_calc
    else:
        shap_values_fraud_instance_class1 = shap_values_fraud_instance_calc
        expected_value_class1 = expected_value_calc


    force_plot_fraud_html = shap.force_plot(
        expected_value_class1,
        shap_values_fraud_instance_class1,
        instance_fraud,
        matplotlib=False
    )
    shap.save_html("xai_explanations/shap_force_plot_fraud_instance.html", force_plot_fraud_html)
    print("✅ SHAP force plot for fraud instance saved. Open .html file in a browser.")

    # Force plot for a non-fraudulent transaction
    print(f"Generating SHAP force plot for a non-fraudulent instance (index {idx_non_fraud})...")
    shap_values_non_fraud_instance_calc = explainer_shap.shap_values(instance_non_fraud)

    if isinstance(shap_values_non_fraud_instance_calc, list):
        shap_values_non_fraud_instance_class1 = shap_values_non_fraud_instance_calc[1]
    else:
        shap_values_non_fraud_instance_class1 = shap_values_non_fraud_instance_calc

    force_plot_non_fraud_html = shap.force_plot(
        expected_value_class1,
        shap_values_non_fraud_instance_class1,
        instance_non_fraud,
        matplotlib=False
    )
    shap.save_html("xai_explanations/shap_force_plot_non_fraud_instance.html", force_plot_non_fraud_html)
    print("✅ SHAP force plot for non-fraud instance saved.")
else:
    print("Could not find both fraud and non-fraud instances in the test set for SHAP force plots.")


# --- 5.2 LIME (Local Interpretable Model-agnostic Explanations) ---
print(f"\nRunning LIME explanations for {chosen_model_name} model...")

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train_resampled_scaled),
    feature_names=X_train_scaled_df.columns.tolist(),
    class_names=['Not Fraud', 'Fraud'],
    mode='classification',
    discretize_continuous=True
)

if len(fraud_indices) > 0 and len(non_fraud_indices) > 0:
    print(f"Generating LIME plot for a fraudulent instance (index {idx_fraud})...")
    lime_exp_fraud = lime_explainer.explain_instance(
        data_row=instance_fraud.values.ravel(),
        predict_fn=chosen_model_for_xai.predict_proba,
        num_features=10,
        top_labels=1
    )
    lime_exp_fraud.save_to_file('xai_explanations/lime_explanation_fraud_instance.html')
    print("✅ LIME explanation for fraud instance saved as HTML.")

    print(f"Generating LIME plot for a non-fraudulent instance (index {idx_non_fraud})...")
    lime_exp_non_fraud = lime_explainer.explain_instance(
        data_row=instance_non_fraud.values.ravel(),
        predict_fn=chosen_model_for_xai.predict_proba,
        num_features=10,
        top_labels=1
    )
    lime_exp_non_fraud.save_to_file('xai_explanations/lime_explanation_non_fraud_instance.html')
    print("✅ LIME explanation for non-fraud instance saved as HTML.")
else:
    print("Could not find both fraud and non-fraud instances in the test set for LIME plots.")

print("\n✅ XAI explanations generation complete. Check the 'xai_explanations' directory.")
print("-" * 60)

################################################################################
# Phase 6: Discussion of XAI Insights (Placeholder for your paper)
################################################################################
print("Phase 6: Analysis and Discussion of XAI Insights (To be done by you)")
print("""
(Same discussion points as before - this part requires your manual analysis of the plots)
""")
print("-" * 60)
print("Congratulations! Script execution finished.")