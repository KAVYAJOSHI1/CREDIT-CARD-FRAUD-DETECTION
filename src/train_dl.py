import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import pickle
import os

def train_dl_model():
    print("Loading 1M Behavioral Dataset for Deep Learning...")
    df = pd.read_csv('data/behavioral_transactions.csv')
    
    X = df.drop('fraud', axis=1).values
    y = df['fraud'].values
    
    print("Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Scaling Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs('models', exist_ok=True)
    with open('models/dl_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Building Advanced Deep Neural Network...")
    model = Sequential([
        Dense(64, input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.3),
        
        Dense(32),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.2),
        
        Dense(16),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.1),
        
        Dense(8),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    print("Configuring Callbacks...")
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    checkpoint = ModelCheckpoint('models/deep_learning_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    
    print("Training DNN...")
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=10, 
        batch_size=512, 
        validation_data=(X_test_scaled, y_test),
        callbacks=[early_stop, reduce_lr, checkpoint]
    )
    
    print("Generating Training History Plot...")
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Epoch vs Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.title('Epoch vs Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    print("Training plot saved to models/training_history.png")
    
if __name__ == '__main__':
    train_dl_model()
