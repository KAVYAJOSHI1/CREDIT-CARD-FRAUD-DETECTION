import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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
    
    print("Building Deep Neural Network...")
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Training DNN...")
    # Because of the large dataset, just 3 epochs with a large batch size will yield excellent results fast
    model.fit(X_train_scaled, y_train, epochs=3, batch_size=256, validation_data=(X_test_scaled, y_test))
    
    print("Saving Model...")
    model.save('models/deep_learning_model.keras')
    print("Model saved to models/deep_learning_model.keras")
    
if __name__ == '__main__':
    train_dl_model()
