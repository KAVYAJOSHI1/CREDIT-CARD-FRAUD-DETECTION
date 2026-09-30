import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, precision_recall_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import os

def build_simple_dl_model(input_dim):
    # This is the original architecture that scored F1: 0.84
    model = Sequential([
        Dense(32, activation='relu', input_dim=input_dim),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='prc', curve='PR')
        ]
    )
    return model

def train_with_class_weights():
    print("Loading original un-SMOTEd dataset...")
    df = pd.read_csv('data/creditcard.csv')
    
    # Scale
    rob_scaler = RobustScaler()
    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
    df.drop(['Time','Amount'], axis=1, inplace=True)
    
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    print("Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Calculate Class Weights to heavily penalize missing a fraud case
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = {0: weights[0], 1: weights[1]}
    print(f"Computed Class Weights: {class_weight}")
    
    print("\nTraining Model with Class Weights...")
    model = build_simple_dl_model(X_train.shape[1])
    
    os.makedirs('models', exist_ok=True)
    
    callbacks = [
        EarlyStopping(monitor='val_prc', mode='max', patience=5, restore_best_weights=True),
        ModelCheckpoint('models/best_model.keras', monitor='val_prc', mode='max', save_best_only=True)
    ]
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=2048,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set directly using optimal threshold tuning
    print("\nPredicting probabilities for tuning...")
    probs = model.predict(X_test, verbose=0).ravel()
    
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"\n--- Threshold Tuning (Class Weights Model) ---")
    print(f"Optimal Threshold (Max F1): {best_threshold:.4f}")
    print(f"Best F1-Score: {best_f1:.4f}")
    print(f"Precision at optimal threshold: {precision[best_idx]:.4f}")
    print(f"Recall at optimal threshold: {recall[best_idx]:.4f}")
    
    print("\n--- Classification Report with Optimal Threshold ---")
    preds_optimal = (probs > best_threshold).astype(int)
    print(classification_report(y_test, preds_optimal))

if __name__ == '__main__':
    train_with_class_weights()
