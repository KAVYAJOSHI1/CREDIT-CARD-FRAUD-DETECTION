import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

def evaluate():
    print("Loading Dataset...")
    df = pd.read_csv('data/behavioral_transactions.csv')
    X = df.drop('fraud', axis=1).values
    y = df['fraud'].values
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Loading Model...")
    model = xgb.XGBClassifier()
    model.load_model('models/behavioral_xgb_model.json')
    
    print("Predicting...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy:  {acc*100:.4f}%")
    print(f"Precision: {prec*100:.4f}%")
    print(f"Recall:    {rec*100:.4f}%")
    print(f"F1-Score:  {f1*100:.4f}%")

if __name__ == '__main__':
    evaluate()
