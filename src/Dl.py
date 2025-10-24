# pyright: reportMissingModuleSource=false
from tensorflow.keras import models, layers, optimizers, callbacks
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Layer, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import os, random
import time

start_timme = time.time()

DATA_PATH = "Data/METABRIC_RNA_Mutation.csv"
MODEL_OUT = "output/models/attention_model.keras"
PLOT_OUT = "output/plots/training_curves.png"
ATTN_OUT = "output/tables/attention_weights.csv"

data = pd.read_csv(DATA_PATH)
print("Data loaded:", data.shape)

data = data.dropna()
TARGET_COL = 'pam50_+_claudin-low_subtype'

X =data.dropna(columns=[TARGET_COL])
y = data[TARGET_COL]
