"""Data loading for the real Kaggle creditcard.csv benchmark (284,807 txns, 0.17% fraud).

No SMOTE: synthetic oversampling distorts the PCA feature manifold and inflates
validation scores. Imbalance is handled in the loss (focal loss) instead.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

SEED = 42


def load_splits(path="data/creditcard.csv"):
    df = pd.read_csv(path)
    df["Amount"] = np.log1p(df["Amount"])          # heavy-tailed -> log scale
    df["Time"] = (df["Time"] % 86400) / 86400.0    # time-of-day only (dataset spans 2 days)
    feats = [c for c in df.columns if c != "Class"]
    X, y = df[feats].values.astype("float32"), df["Class"].values.astype("float32")

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    scaler = RobustScaler().fit(X_tr)              # fit on train only -> no leakage
    f = lambda a: np.clip(scaler.transform(a), -10, 10).astype("float32")
    return dict(X_tr=f(X_tr), y_tr=y_tr, X_va=f(X_va), y_va=y_va,
                X_te=f(X_te), y_te=y_te, scaler=scaler, features=feats)
