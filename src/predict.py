import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# We use the optimal threshold discovered during Phase 4
OPTIMAL_THRESHOLD = 0.9977

class FraudDetector:
    def __init__(self, model_path='models/best_model.keras', scaler_path='data/processed/robust_scaler.pkl'):
        print("Loading deep learning model...")
        self.model = tf.keras.models.load_model(model_path)
        
        print("Loading feature scaler...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
    def predict(self, transactions_df):
        """
        Takes a pandas DataFrame of transactions with original features (Time, Amount, V1-V28)
        and returns fraud probabilities and boolean predictions.
        """
        # Make a copy to avoid modifying original
        df = transactions_df.copy()
        
        # 1. Preprocess: Scale Time and Amount
        df['scaled_amount'] = self.scaler.transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = self.scaler.transform(df['Time'].values.reshape(-1, 1))
        
        # Reorder to match training (scaled_amount, scaled_time, V1...V28)
        df.drop(['Time', 'Amount'], axis=1, inplace=True)
        scaled_amount = df.pop('scaled_amount')
        scaled_time = df.pop('scaled_time')
        df.insert(0, 'scaled_amount', scaled_amount)
        df.insert(1, 'scaled_time', scaled_time)
        
        # 2. Predict probabilities
        features = df.values
        probs = self.model.predict(features, verbose=0).ravel()
        
        # 3. Apply threshold
        predictions = (probs > OPTIMAL_THRESHOLD).astype(bool)
        
        return probs, predictions

if __name__ == '__main__':
    # Test the inference script on a few random samples from the test set
    print("Testing Inference Script...")
    raw_data = pd.read_csv('data/creditcard.csv')
    
    # Pick 2 normal and 2 known fraud transactions just to test
    normal_samples = raw_data[raw_data['Class'] == 0].sample(2, random_state=42)
    fraud_samples = raw_data[raw_data['Class'] == 1].sample(2, random_state=42)
    test_samples = pd.concat([normal_samples, fraud_samples])
    
    actual_classes = test_samples.pop('Class').values
    
    detector = FraudDetector()
    probs, preds = detector.predict(test_samples)
    
    print("\n--- Inference Results ---")
    for i in range(len(preds)):
        print(f"Transaction {i+1}:")
        print(f"  Actual: {'FRAUD' if actual_classes[i] == 1 else 'Normal'}")
        print(f"  Predicted: {'FRAUD' if preds[i] else 'Normal'} (Probability: {probs[i]:.4f})")
