# FraudShield AI — Deep Learning Credit Card Fraud Detection

Deep ensemble on the real Kaggle credit-card dataset (284,807 txns, 0.17% fraud).

**Models** (`src/deep/`): residual MLP ×3 seeds, FT-Transformer ×2 seeds, denoising autoencoder (unsupervised),
logistic stacker, XGBoost baseline. Focal loss, AdamW + cosine schedule, no SMOTE, scaler fit on train only.
Extras: MC-dropout uncertainty, Integrated Gradients explanations.

**Held-out test results (99 frauds)** — see `reports/metrics.json`:

| Model | PR-AUC | Precision | Recall |
|---|---|---|---|
| XGBoost baseline | 0.879 | 0.931 | 0.818 |
| ResMLP | 0.876 | 0.965 | 0.828 |
| Stacked ensemble | 0.870 | 0.848 | 0.848 |
| FT-Transformer | 0.869 | 0.976 | 0.828 |
| Autoencoder (no labels) | 0.701 | 0.699 | 0.798 |

The deep models match, not beat, XGBoost; differences are within noise at this test-set size.

## Run
```
pip install -r requirements.txt
# put creditcard.csv in data/ (Kaggle: mlg-ulb/creditcardfraud)
python -m src.deep.train      # ~1 hour on CPU; QUICK=1 for a smoke test
python app.py                 # http://localhost:5000  (SQLite log in db/)
```
