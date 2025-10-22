import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

print(f" Missing values after cleanup: {X.isnull().sum().sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

log_reg = LogisticRegression(
    max_iter=1000,
    solver='saga',
    penalty='l2',
    n_jobs=-1,
    random_state=42
)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')

print("\n Logistic Regression Results:")
print(f"Accuracy: {acc_lr:.3f}, F1-Score: {f1_lr:.3f}")
print(classification_report(y_test, y_pred_lr, target_names=class_names))

cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_lr, annot=True, fmt='d',
            xticklabels=class_names, yticklabels=class_names,
            cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

joblib.dump(log_reg, os.path.join("output", "logistic_regression_model.joblib"))
print("Logistic Regression model saved.")

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42
)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print("\n Random Forest Results:")
print(f"Accuracy: {acc_rf:.3f}, F1-Score: {f1_rf:.3f}")
print(classification_report(y_test, y_pred_rf, target_names=class_names))

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_rf, annot=True, fmt='d',
            xticklabels=class_names, yticklabels=class_names,
            cmap='Greens')
plt.title("Confusion Matrix - Random Forest")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

joblib.dump(rf_clf, os.path.join("output", "random_forest_model.joblib"))
print("Random Forest model saved.")
