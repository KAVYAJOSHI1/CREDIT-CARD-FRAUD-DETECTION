import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, precision_recall_curve, auc
import os

def load_data(data_dir='data/processed'):
    print("Loading datasets...")
    train = pd.read_csv(f'{data_dir}/train_smote.csv')
    val = pd.read_csv(f'{data_dir}/val.csv')
    test = pd.read_csv(f'{data_dir}/test.csv')
    
    X_train = train.drop('Class', axis=1).values
    y_train = train['Class'].values
    
    X_val = val.drop('Class', axis=1).values
    y_val = val['Class'].values
    
    X_test = test.drop('Class', axis=1).values
    y_test = test['Class'].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_advanced_dl_model(input_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        
        Dense(32),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        
        Dense(16),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='prc', curve='PR')
        ]
    )
    return model

def train_dl_model(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\nTraining Advanced Deep Learning Model...")
    model = build_advanced_dl_model(X_train.shape[1])
    
    os.makedirs('models', exist_ok=True)
    
    callbacks = [
        EarlyStopping(monitor='val_prc', mode='max', patience=8, restore_best_weights=True),
        ModelCheckpoint('models/best_model.keras', monitor='val_prc', mode='max', save_best_only=True),
        ReduceLROnPlateau(monitor='val_prc', mode='max', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=2048, 
        callbacks=callbacks,
        verbose=1
    )
    
    # Quick evaluation on default threshold (0.5)
    print("\n--- Advanced DL Model Evaluation (Default Threshold 0.5) ---")
    probs = model.predict(X_test, verbose=0)
    preds = (probs > 0.5).astype(int)
    print(classification_report(y_test, preds))
    
    return model

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    model = train_dl_model(X_train, y_train, X_val, y_val, X_test, y_test)
    print("Advanced Training process completed successfully.")
