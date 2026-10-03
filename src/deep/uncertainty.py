"""Recompute MC-dropout uncertainty + Integrated-Gradients plot from saved models.
(The first full run executed BatchNorm in training mode during MC-dropout; this is the corrected version.)"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import load_splits
from .inference import FraudEnsemble, integrated_gradients

d = load_splits(); ens = FraudEnsemble(); y = d["y_te"]
mcs = np.stack([ens.mc_model(d["X_te"], training=True).numpy().ravel() for _ in range(30)])
mu, sd = mcs.mean(0), mcs.std(0)
wrong = (mu >= 0.5) != (y == 1); q = np.quantile(sd, 0.9)
unc = dict(error_rate_low_uncertainty=float(wrong[sd < q].mean()), error_rate_top10pct_uncertainty=float(wrong[sd >= q].mean()),
           mean_std_fraud=float(sd[y == 1].mean()), mean_std_legit=float(sd[y == 0].mean()), note="BatchNorm frozen; dropout only")
json.dump(unc, open("reports/uncertainty.json", "w"), indent=2); print(unc)

ig = integrated_gradients(ens.mlps[0], d["X_te"][y == 1])
imp = np.abs(ig).mean(0); order = np.argsort(imp)[::-1][:15]
plt.figure(figsize=(8, 6)); plt.barh([d["features"][i] for i in order][::-1], imp[order][::-1], color="#c0392b")
plt.title("Integrated Gradients — what drives fraud predictions"); plt.tight_layout()
plt.savefig("reports/feature_attribution.png", dpi=140)
