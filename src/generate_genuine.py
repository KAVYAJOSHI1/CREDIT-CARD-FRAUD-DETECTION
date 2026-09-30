import pandas as pd
import numpy as np
import os
import random

def generate_genuine_dataset(num_samples=20000):
    np.random.seed(42)
    random.seed(42)
    
    categories = ['Retail', 'Travel', 'Dining', 'Online', 'Electronics', 'Groceries']
    locations = ['Domestic', 'International']
    
    data = []
    
    for i in range(num_samples):
        account_num = f"4{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"
        amount = round(np.random.lognormal(mean=3, sigma=1.2), 2)  # Most transactions are small, some are huge
        time_hour = random.randint(0, 23)
        category = random.choice(categories)
        
        # 90% Domestic, 10% International
        location = 'Domestic' if random.random() > 0.1 else 'International'
        
        # Most people have 0 failed attempts
        failed_attempts = 0
        if random.random() > 0.9:
            failed_attempts = random.randint(1, 4)
            
        is_fraud = 0
        
        # Generate Fraud based on logical rules (so XGBoost can learn them)
        fraud_chance = 0.001 # Base chance
        
        if location == 'International' and amount > 500:
            fraud_chance = 0.4
        
        if failed_attempts >= 3:
            fraud_chance = 0.7
            
        if 2 <= time_hour <= 5 and amount > 300:
            fraud_chance = 0.5
            
        if category == 'Electronics' and amount > 1000:
            fraud_chance = 0.3
            
        if random.random() < fraud_chance:
            is_fraud = 1
            
        # Ensure we have enough fraud by forcing some
        # (This is just for demonstration purposes)
        if i < 200: 
            is_fraud = 1
            amount = round(random.uniform(500, 5000), 2)
            if random.random() > 0.5: location = 'International'
            if random.random() > 0.5: failed_attempts = random.randint(2, 4)
            
        data.append({
            'Transaction_ID': f"TXN{100000 + i}",
            'Account_Number': account_num,
            'Transaction_Amount': amount,
            'Transaction_Time': time_hour,
            'Merchant_Category': category,
            'Location_Type': location,
            'Failed_PIN_Attempts': failed_attempts,
            'Is_Fraud': is_fraud
        })
        
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/genuine_transactions.csv', index=False)
    
    fraud_count = df['Is_Fraud'].sum()
    print(f"Generated {num_samples} transactions.")
    print(f"Total Frauds: {fraud_count} ({(fraud_count/num_samples)*100:.2f}%)")
    print("Dataset saved to 'data/genuine_transactions.csv'")

if __name__ == '__main__':
    generate_genuine_dataset()
