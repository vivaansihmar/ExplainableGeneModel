# pyright: reportMissingModuleSource=false
from tensorflow.keras import Model, optimizers, callbacks
import tensorflow_addons as tfa
import numpy as np
from tensorflow.keras import layers,backend as K
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import  classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Layer, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import os, random, time 


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

class AttentionLayer(layers.Layer):
    def _init_(self, Kwargs):
        super()._init_(Kwargs)

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
print("Attention Layer defined, time taken: {:.2f} seconds".format(time.time() - start_time))
print(AttentionLayer().summary())



def f1_score(y_true, y_pred):
    y_pred  = tf.rounds(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'),axis =0)
    fp = K.sum(K.cast((1-y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1-y_pred), 'float'), axis=0)
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.mean(f1)

#model
Input_dim = X_train.shape[1]
inputs = keras.Input(shape=(Input_dim,), name='gene_features')
X = layers.Dense(512, activation='relu')(inputs)
X = layers.Dropout(0.3)(X)
X = layers.Reshape((512,1))(X)
X = AttentionLayer()(X)
X = layers.Flatten()(X)
X = layers.Dense(256, activation='relu')(X)
X = layers.Dropout(0.3)(X)
outputs = layers.Dense(num_classes, activation='softmax')(X)
model = keras.Model(inputs,outputs, name='Attention_DNN')
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', f1_score])
print(model.summary())
print("Model compiled  time taken: {:.2f} seconds".format(time.time() - start_time))

#train
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights= True)
history =model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

#evaluation
