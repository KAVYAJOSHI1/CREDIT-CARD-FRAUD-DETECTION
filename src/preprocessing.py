import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import pickle
import os

def load_and_preprocess_data(filepath='data/creditcard.csv', output_dir='data/processed'):
    """
    Loads raw data, scales features, splits it, and applies SMOTE to the training set.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    print("Scaling 'Time' and 'Amount' features using RobustScaler...")
    # RobustScaler is less prone to outliers than StandardScaler
    rob_scaler = RobustScaler()
    
    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
    
    df.drop(['Time','Amount'], axis=1, inplace=True)
    
    # Move scaled features to the front (optional, but good for consistency)
    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']
    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)
    
    # Split data into X and y
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Train-test split (using stratify to ensure class ratio is maintained)
    print("Splitting data into train, val, and test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    print(f"Original training fraud distribution:\n{y_train.value_counts(normalize=True)}")
    
    # Apply SMOTE to training data only to avoid data leakage
    print("Applying SMOTE to training data to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"SMOTE training shape: {X_train_smote.shape}")
    print(f"SMOTE training fraud distribution:\n{y_train_smote.value_counts(normalize=True)}")
    
    # Save the processed datasets
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving processed data to {output_dir}...")
    
    # Converting back to dataframes for easy saving
    pd.concat([X_train_smote, y_train_smote], axis=1).to_csv(f'{output_dir}/train_smote.csv', index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(f'{output_dir}/val.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(f'{output_dir}/test.csv', index=False)
    
    # Save scaler for inference time
    with open(f'{output_dir}/robust_scaler.pkl', 'wb') as f:
        pickle.dump(rob_scaler, f)
        
    print("Preprocessing completed successfully!")

if __name__ == '__main__':
    load_and_preprocess_data()
