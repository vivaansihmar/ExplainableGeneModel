import shap 
import joblib
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder, StandardScaler  

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

log_reg = joblib.load("output/logistic_regression_model.joblib")
rf_clf = joblib.load("output/random_forest_model.joblib")

#shap logrg
print("\n🧠 SHAP Analysis for Logistic Regression...")


explainer_lr = shap.LinearExplainer(log_reg, X, feature_dependence="independent")
shap_values_lr = explainer_lr.shap_values(X)

#summ plot
plt.title("SHAP Summary - Logistic Regression")
shap.summary_plot(shap_values_lr, X, plot_type="bar", show=False)
plt.savefig("output/plots/shap_summary_log_reg.png", bbox_inches='tight')
plt.show()

# shap-rf
print("\n🌳 SHAP Analysis for Random Forest...")

explainer_rf = shap.TreeExplainer(rf_clf)
shap_values_rf = explainer_rf.shap_values(X)

plt.title("SHAP Summary - Random Forest")
shap.summary_plot(shap_values_rf, X, plot_type="bar", show=False)
plt.savefig("output/plots/shap_summary_rf.png", bbox_inches='tight')
plt.show()

import numpy as np

feature_importance_lr = pd.DataFrame({
    'Feature': X.columns,
    'Mean_Abs_SHAP': np.abs(shap_values_lr).mean(axis=0)
}).sort_values('Mean_Abs_SHAP', ascending=False)

feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Mean_Abs_SHAP': np.abs(shap_values_rf).mean(axis=0)
}).sort_values('Mean_Abs_SHAP', ascending=False)

feature_importance_lr.head(20).to_csv("output/top_features_log_reg.csv", index=False)
feature_importance_rf.head(20).to_csv("output/top_features_rf.csv", index=False)

print("\n✅ SHAP analysis complete. Top features saved in output/.")