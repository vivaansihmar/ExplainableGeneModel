import shap 
import joblib
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler  
import os 

start_time = time.time()
print("\n Loading METABRIC dataset...")
df = pd.read_csv("Data/METABRIC_RNA_Mutation.csv", low_memory=False)
TARGET_COL = 'pam50_+_claudin-low_subtype'
le = LabelEncoder()
df[TARGET_COL] = le.fit_transform(df[TARGET_COL].astype(str))
class_names = le.classes_.tolist()

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

scaler = StandardScaler()
X[X.select_dtypes(include='number').columns] = scaler.fit_transform(X.select_dtypes(include='number'))

X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median())
print(f" Data ready for SHAP analysis. Shape: {X.shape}")
print(f" Data preprocessing took: {time.time() - start_time:.2f} seconds")
print("\n Loading trained models...")
model_load_start = time.time()

log_reg = joblib.load("output/logistic_regression_model.joblib")
rf_clf = joblib.load("output/random_forest_model.joblib")
print(f" Models loaded successfully in {time.time() - model_load_start:.2f} seconds")
os.makedirs("output/plots", exist_ok=True)


#shap logrg
lr_start = time.time()
print("\n SHAP Analysis for Logistic Regression...")


masker = shap.maskers.Independent(X)
explainer_lr = shap.LinearExplainer(log_reg, masker=masker)
shap_values_lr = explainer_lr(X)
shap_values_array_lr = shap_values_lr.values

print(f"Logistic Regression SHAP computation took: {time.time() - lr_start:.2f} seconds")

#summ plot
plt.title("SHAP Summary - Logistic Regression")
shap.summary_plot(shap_values_lr, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("output/plots/shap_summary_log_reg.png", bbox_inches='tight')
plt.close()

# shap-rf
rf_start = time.time()
print("\n SHAP Analysis for Random Forest...")

explainer_rf = shap.TreeExplainer(rf_clf)
shap_values_rf = explainer_rf.shap_values(X)

if isinstance(shap_values_rf, list):
    shap_values_rf = np.mean(np.abs(shap_values_rf), axis=0)

print(f" Random Forest SHAP computation took: {time.time() - rf_start:.2f} seconds")


plt.title("SHAP Summary - Random Forest")
shap.summary_plot(shap_values_rf, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("output/plots/shap_summary_rf.png", bbox_inches='tight')
plt.close()


tables_start = time.time()
print("\n Generating feature importance tables...")
feature_importance_lr = pd.DataFrame({
    'Feature': X.columns,
    'Mean_Abs_SHAP': np.abs(shap_values_lr).mean(axis=0)
}).sort_values('Mean_Abs_SHAP', ascending=False)

feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Mean_Abs_SHAP': np.abs(shap_values_rf).mean(axis=0)
}).sort_values('Mean_Abs_SHAP', ascending=False)

os.makedirs("output", exist_ok=True)
feature_importance_lr.head(20).to_csv("output/top_features_log_reg.csv", index=False)
feature_importance_rf.head(20).to_csv("output/top_features_rf.csv", index=False)
print(f" Feature importance CSVs saved in {time.time() - tables_start:.2f} seconds")


total_time = time.time() - start_time
print(f"\n Total runtime: {total_time/60:.2f} minutes")
print("\n SHAP analysis complete. Top features saved in output/.")
print("Top 20 feature CSVs saved in /output:")
print(" - top_features_log_reg.csv")
print(" - top_features_rf.csv")
print("\n plots saved")