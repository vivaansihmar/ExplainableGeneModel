# pyright: reportMissingModuleSource=false
from tensorflow.keras import models, layers, optimizers, callbacks
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os, random
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

df = pd.read_csv("Data\METABRIC_RNA_Mutation.csv")
print(df.shape)
print(df.columns[:50])
df.head()