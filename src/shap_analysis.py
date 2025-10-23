import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import time

start_time = time.time()

df = pd.read_csv("Data/METABRIC_RNA_Mutation.csv", low_memory=False)
TARGET_COL = 'pam50_+_claudin-low_subtype'
le = LabelEncoder()
df[TARGET_COL] = le.fit_transform(df[TARGET_COL].astype(str))

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

scaler = StandardScaler()
X[X.select_dtypes(include='number').columns] = scaler.fit_transform(X.select_dtypes(include='number'))
X = X.apply(pd.to_numeric, errors='coerce').fillna(X.median())

print(f"Data ready for SHAP analysis. Shape: {X.shape}")
print(f"Data preprocessing took: {time.time() - start_time:.2f} seconds")

# Load models
log_reg = joblib.load("output/models/logistic_regression_model.joblib")
rf_clf = joblib.load("output/models/random_forest_model.joblib")

os.makedirs("output/plots", exist_ok=True)
os.makedirs("output/tables", exist_ok=True)

def get_shap_array(shap_values):
    """Convert SHAP outputs into 2D array (samples × features)"""
    if isinstance(shap_values, list):
        # For multi-class TreeExplainer outputs
        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif hasattr(shap_values, "values"):
        shap_values = shap_values.values
    shap_values = np.array(shap_values)
    if shap_values.ndim > 2:
        # Average over classes for multi-class models
        shap_values = np.mean(np.abs(shap_values), axis=2)
    return shap_values

# Logistic Regression SHAP
lr_start = time.time()
masker = shap.maskers.Independent(X)
explainer_lr = shap.LinearExplainer(log_reg, masker=masker)
shap_values_lr = explainer_lr(X)
shap_values_lr_plot = get_shap_array(shap_values_lr)

print(f"Logistic Regression SHAP shape (samples x features): {shap_values_lr_plot.shape}")

# Plot summary bar
shap.summary_plot(shap_values_lr_plot, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("output/plots/shap_summary_log_reg.png", bbox_inches='tight')
plt.close()

# Save top features
feature_importance_lr = pd.DataFrame({
    "Feature": X.columns,
    "Mean_Abs_SHAP": np.mean(shap_values_lr_plot, axis=0)
}).sort_values("Mean_Abs_SHAP", ascending=False)
feature_importance_lr.to_csv("output/tables/top_features_log_reg.csv", index=False)
print(f"Logistic Regression SHAP done in {time.time() - lr_start:.2f} seconds")

# Random Forest SHAP
rf_start = time.time()
explainer_rf = shap.TreeExplainer(rf_clf)
shap_values_rf = explainer_rf.shap_values(X)
shap_values_rf_plot = get_shap_array(shap_values_rf)

# Plot summary bar
shap.summary_plot(shap_values_rf_plot, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("output/plots/shap_summary_rf.png", bbox_inches='tight')
plt.close()

# Save top features
feature_importance_rf = pd.DataFrame({
    "Feature": X.columns,
    "Mean_Abs_SHAP": np.mean(shap_values_rf_plot, axis=0)
}).sort_values("Mean_Abs_SHAP", ascending=False)
feature_importance_rf.to_csv("output/tables/top_features_rf.csv", index=False)
print(f"Random Forest SHAP done in {time.time() - rf_start:.2f} seconds")

total_time = time.time() - start_time
print(f"Total runtime: {total_time/60:.2f} minutes")
print("SHAP analysis complete. Plots saved in output/plots/, feature CSVs saved in output/tables/")
