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

start_time = time.time()

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

lable_Encoder  = LabelEncoder()
y_encoded =lable_Encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))
y_onehot = keras.utlis.to_categorical(y_encoded, num_classes=num_classes)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_train = train_test_split(
    X_scaled, y_onehot, train_size=0.8, test_size=0.2, stratify=y_onehot, 
    random_state = 42
)
print(f"Train_shape: {X_train.shape}, Test_shape: {X_test.shape}")
print(f"data processing took: {time.time() -start_time:.2f} seconds")

class AttentionLayer(layers):
    def _init_(self):
        super(AttentionLayer, self)._init_()
        def build(self, input_shape):
            self.attention_weights = self.add_weight(
                shape = (input_shape[-1],1),
                initializer = "random_normal",
                trainable = True,
                name = "AttentionLayer_weights"
                )
            def call(self, inputs):
                attention_scores = tf.nn.softmax(tf.matmul(inputs, self.attention_weights), axis=1)
                attended_output = inputs * attention_scores
                return attended_output
print(AttentionLayer().summary())
def build_model(input_shape, num_classes ):
    inputs = Input(shape=(input_shape,))