import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

df = pd.read_csv("Data\METABRIC_RNA_Mutation.csv")
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train:{X_train.shape}, Test: {X_test.shape} ")
log_reg = LogisticRegression(
    max_iter=1000,
    solver = 'saga',
    penalty='l2',
    n_jobs=-1,
    random_state=42
)

log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

acc_lr = accuracy_score(y_test,y_pred_lr)
f1_lr = f1_score(y_test,y_pred_lr, average='weighted')

print("\n Logistic Regression Results:")
print(f"Accuracy: {acc_lr:.3f}, F1-Score: {f1_lr:.3f}")
print(classification_report(y_test, y_pred_lr, target_names=class_names))

cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(10,7))
sns.heatmap(cm_lr, annot=True, fmt= 'd',
            xticklabels=class_names, yticklabels=class_names, 
            cmap='Blues')
plt.title("confusion matrix - Logistic Regression")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
joblib.dump(log_reg, os.path.join("output", "logistic_regression_model.joblib"))
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42
)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
acc_rf = accuracy_score(y_test,y_pred_rf)