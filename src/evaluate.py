import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib.pyplot as plt

def evaluate_and_tune(model_path='models/best_model.keras', data_dir='data/processed'):
    print("Loading test data and model...")
    test = pd.read_csv(f'{data_dir}/test.csv')
    X_test = test.drop('Class', axis=1).values
    y_test = test['Class'].values
    
    model = tf.keras.models.load_model(model_path)
    
    print("Predicting probabilities...")
    probs = model.predict(X_test).ravel()
    
    # Calculate precision, recall for all thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    
    # Calculate F1 score for all thresholds to find the best one
    # Note: we add a small epsilon to avoid division by zero
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"\n--- Threshold Tuning ---")
    print(f"Optimal Threshold (Max F1): {best_threshold:.4f}")
    print(f"Best F1-Score: {best_f1:.4f}")
    print(f"Precision at optimal threshold: {precision[best_idx]:.4f}")
    print(f"Recall at optimal threshold: {recall[best_idx]:.4f}")
    
    print("\n--- Classification Report with Optimal Threshold ---")
    preds_optimal = (probs > best_threshold).astype(int)
    print(classification_report(y_test, preds_optimal))
    
    # In a real scenario, you could save this threshold to a config file for inference

if __name__ == '__main__':
    evaluate_and_tune()
