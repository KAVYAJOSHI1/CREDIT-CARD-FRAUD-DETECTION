import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

def train_behavioral_model():
    print("Loading 1M Behavioral Dataset...")
    df = pd.read_csv('data/behavioral_transactions.csv')
    
    X = df.drop('fraud', axis=1).values
    y = df['fraud'].values
    
    print("Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    print("Training Behavioral XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=ratio, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    print("Saving Model...")
    xgb_model.save_model('models/behavioral_xgb_model.json')
    print("Model saved to models/behavioral_xgb_model.json")
    
if __name__ == '__main__':
    train_behavioral_model()
