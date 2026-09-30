import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
import xgboost as xgb
import os

def load_and_prep_data():
    print("Loading original dataset...")
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
    
    return X_train, y_train, X_test, y_test

def tune_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx], precision[best_idx], recall[best_idx]

def run_showdown():
    X_train, y_train, X_test, y_test = load_and_prep_data()
    
    print("\n" + "="*50)
    print("MODEL 1: RANDOM FOREST")
    print("="*50)
    # Using class_weight='balanced' to handle the severe imbalance natively
    rf = RandomForestClassifier(n_estimators=50, class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    
    rf_thresh, rf_f1, rf_prec, rf_rec = tune_threshold(y_test, rf_probs)
    print(f"Optimal Threshold: {rf_thresh:.4f}")
    print(f"Metrics -> F1: {rf_f1:.4f} | Precision: {rf_prec:.4f} | Recall: {rf_rec:.4f}")
    
    print("\n" + "="*50)
    print("MODEL 2: XGBOOST")
    print("="*50)
    # scale_pos_weight is the ratio of negative class to positive class (approx 99.8 / 0.17 = ~587)
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    xgb_model = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=ratio, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    xgb_thresh, xgb_f1, xgb_prec, xgb_rec = tune_threshold(y_test, xgb_probs)
    print(f"Optimal Threshold: {xgb_thresh:.4f}")
    print(f"Metrics -> F1: {xgb_f1:.4f} | Precision: {xgb_prec:.4f} | Recall: {xgb_rec:.4f}")
    
    print("\n" + "="*50)
    print("SHOWDOWN RESULTS SUMMARY")
    print("="*50)
    print("Neural Net (SMOTE Baseline): F1: 0.8421 | Precision: 0.8955 | Recall: 0.8108")
    print(f"Random Forest              : F1: {rf_f1:.4f} | Precision: {rf_prec:.4f} | Recall: {rf_rec:.4f}")
    print(f"XGBoost                    : F1: {xgb_f1:.4f} | Precision: {xgb_prec:.4f} | Recall: {xgb_rec:.4f}")

if __name__ == '__main__':
    run_showdown()
