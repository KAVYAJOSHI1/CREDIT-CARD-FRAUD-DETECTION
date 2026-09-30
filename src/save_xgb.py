import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import os
import pickle

def save_xgb_model():
    print("Loading dataset...")
    df = pd.read_csv('data/creditcard.csv')
    
    # Scale
    rob_scaler = RobustScaler()
    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
    df.drop(['Time','Amount'], axis=1, inplace=True)
    
    # Save the scaler
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/robust_scaler.pkl', 'wb') as f:
        pickle.dump(rob_scaler, f)
    
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=ratio, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    
    print("Saving model...")
    os.makedirs('models', exist_ok=True)
    xgb_model.save_model('models/xgb_model.json')
    print("Model saved to models/xgb_model.json")

if __name__ == '__main__':
    save_xgb_model()
