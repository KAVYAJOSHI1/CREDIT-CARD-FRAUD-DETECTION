import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
import pickle

def train_genuine_model():
    print("Loading Genuine Dataset...")
    df = pd.read_csv('data/genuine_transactions.csv')
    
    # Drop IDs and Account Numbers for training (AI shouldn't learn specific account numbers)
    df_model = df.drop(['Transaction_ID', 'Account_Number'], axis=1)
    
    # Convert categorical to dummy variables (One-Hot Encoding)
    df_model = pd.get_dummies(df_model, columns=['Merchant_Category', 'Location_Type'])
    
    # Save the expected column order so the Flask API knows how to structure incoming data
    expected_columns = list(df_model.drop('Is_Fraud', axis=1).columns)
    
    os.makedirs('models', exist_ok=True)
    with open('models/expected_columns.pkl', 'wb') as f:
        pickle.dump(expected_columns, f)
        
    X = df_model.drop('Is_Fraud', axis=1).values
    y = df_model['Is_Fraud'].values
    
    print("Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    print("Training XGBoost...")
    # Using enable_categorical=False since we manually did one-hot encoding
    xgb_model = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=ratio, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    
    print("Saving Model...")
    xgb_model.save_model('models/genuine_xgb_model.json')
    print("Model saved to models/genuine_xgb_model.json")
    
if __name__ == '__main__':
    train_genuine_model()
