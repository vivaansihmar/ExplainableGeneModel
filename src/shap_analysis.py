import shap 
import joblib
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import labelEncoder, StandardScaler

df = pd.read_csv("Data/METABRIC_RNA_Mutation.csv", low_memory=False)
TARGET_COL = 'pam50_+_claudin-low_subtype'
le = LabelEncoder()
df[TARGET_COL] = le.fit_transform(df[TARGET_COL].astype(str))
class_names = le.classes_.tolist()