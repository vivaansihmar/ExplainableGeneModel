# pyright: reportMissingModuleSource=false
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K, Model, callbacks, optimizers
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, Layer
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os, time

start_time = time.time()

DATA_PATH = "Data/METABRIC_RNA_Mutation.csv"
MODEL_OUT = "output/models/attention_model.keras"
ATTN_OUT = "output/tables/attention_weights.csv"

data = pd.read_csv(DATA_PATH)
print("Data loaded:", data.shape)

data = data.dropna()
TARGET_COL = 'pam50_+_claudin-low_subtype'

X=data.drop(columns=[TARGET_COL])
y = data[TARGET_COL]

categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(exclude=['object']).columns

print(f"Categorical columns: {list(categorical_cols)}")
print(f"Numeric columns: {list(numeric_cols)}")

X_encoded = X.copy()
for col in categorical_cols:
    encoder = LabelEncoder()
    X_encoded[col] = encoder.fit_transform(X[col].astype(str))

lable_Encoder  = LabelEncoder()
y_encoded =lable_Encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))
y_onehot = to_categorical(y_encoded, num_classes=num_classes)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_onehot, train_size=0.8, test_size=0.2, stratify=y_onehot, 
    random_state = 42
)
print(f"Train_shape: {X_train.shape}, Test_shape: {X_test.shape}")
print(f"data processing took: {time.time() -start_time:.2f} seconds")

class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.attention_weights = self.add_weight(
            shape = (input_shape[-1],1),
            initializer = "random_normal",
            trainable = True,
            name = "AttentionLayer_weights"
            )
        super().build(input_shape)

    def call(self, inputs):
        attention_scores = tf.nn.softmax(tf.matmul(inputs, self.attention_weights), axis=1)
        attended_output = inputs * attention_scores
        return attended_output
print("Attention Layer defined, time taken: {:.2f} seconds".format(time.time() - start_time))

def f1_score(y_true, y_pred):
    y_pred  = tf.round(y_pred)
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
    validation_data=(X_test,y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

#evaluation
loss, accuracy, f1 = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {accuracy:.4f}, Test F1: {f1:.4f}")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
unique_labels = sorted(np.unique(y_true))
target_names = [lable_Encoder.classes_[i] for i in unique_labels]
print("\nClassification Report:")
print(classification_report(
    y_true, y_pred, 
    labels=unique_labels,
    target_names=target_names,
    zero_division=0
))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=lable_Encoder.classes_,
            yticklabels=lable_Encoder.classes_)
plt.title('Confusion Matrix-attention DNN')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig("output/plots/CM_DNN.png")
plt.close()

model.save(MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")
print("time taken: {:.2f} seconds".format(time.time() - start_time))

plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('training-valdiation accuracy')
plt.legend()
plt.savefig("output/plots/training_accuracy.png")
plt.close()

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('training-valdiationloss')
plt.legend()
plt.savefig("output/plots/training_loss.png")
plt.close()

total_time = (time.time() - start_time)/60
print(f"\nTotal runtime: {total_time:.2f} minutes")
print("DL atttention model training complete.")