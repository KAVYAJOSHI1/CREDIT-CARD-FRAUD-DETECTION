import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os

def generate_synthetic_credit_card_data(filepath='data/creditcard.csv'):
    """
    Generates a synthetic dataset that mimics the Kaggle Credit Card Fraud Detection dataset.
    - ~284,800 samples
    - 30 features (V1-V28, Time, Amount)
    - Highly imbalanced (~0.17% fraud)
    """
    print("Generating synthetic credit card fraud dataset...")
    
    # 284807 samples, 492 frauds -> ~0.00172 ratio
    n_samples = 284800
    weights = [0.99827, 0.00173]
    
    # We need 30 features total (V1-V28, Time, Amount)
    # make_classification will generate the V1-V28 features
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=28, 
        n_informative=10, 
        n_redundant=2, 
        n_repeated=0, 
        n_classes=2, 
        n_clusters_per_class=2, 
        weights=weights, 
        flip_y=0.01, 
        random_state=42
    )
    
    # Create DataFrame for V1-V28
    feature_names = [f'V{i}' for i in range(1, 29)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add 'Time' feature (simulated as seconds elapsed between transactions)
    # The real dataset spans 2 days (172800 seconds)
    df.insert(0, 'Time', np.sort(np.random.uniform(0, 172800, n_samples)))
    
    # Add 'Amount' feature (simulated monetary value, highly skewed)
    # Legitimate transactions typically have a different distribution than fraudulent ones
    amount_legit = np.random.lognormal(mean=3, sigma=1.2, size=n_samples)
    amount_fraud = np.random.lognormal(mean=5, sigma=1.5, size=n_samples)
    
    # Assign amounts based on class
    df['Amount'] = np.where(y == 0, amount_legit, amount_fraud)
    
    # Ensure amount is not negative and round to 2 decimal places
    df['Amount'] = df['Amount'].clip(lower=0).round(2)
    
    # Add the target column
    df['Class'] = y
    
    # Save to CSV
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    
    print(f"Dataset successfully generated and saved to {filepath}")
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud distribution:\n{df['Class'].value_counts(normalize=True)}")

if __name__ == '__main__':
    generate_synthetic_credit_card_data()
