"""Serving wrapper around the trained ensemble (ResMLP x3 + FT-Transformer x2 + DAE -> stacker)."""
import glob, json, os, pickle
import numpy as np
import keras
import tensorflow as tf
from . import models as _models  # noqa: F401  (registers custom layers/losses for deserialization)

MODELS, REPORTS = "models/deep", "reports"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def freeze_batchnorm(model):
    """Keras 3 BatchNorm uses moving statistics even under training=True when trainable=False,
    so training=True then switches on *only* Dropout -> proper MC-dropout."""
    for layer in model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    return model


def integrated_gradients(model, X, baseline=None, steps=32):
    X = tf.constant(X, dtype=tf.float32)
    base = tf.zeros_like(X) if baseline is None else tf.constant(baseline, dtype=tf.float32)
    total = tf.zeros_like(X)
    for a in np.linspace(0, 1, steps):
        xi = base + a * (X - base)
        with tf.GradientTape() as t:
            t.watch(xi)
            out = model(xi, training=False)
        total += t.gradient(out, xi)
    return ((X - base) * total / steps).numpy()


class FraudEnsemble:
    def __init__(self, models_dir=MODELS, reports_dir=REPORTS):
        art = pickle.load(open(f"{models_dir}/scaler.pkl", "rb"))
        self.scaler, self.features = art["scaler"], art["features"]
        self.stacker = pickle.load(open(f"{models_dir}/stacker.pkl", "rb"))
        load = lambda pat: [keras.saving.load_model(p, compile=False) for p in sorted(glob.glob(f"{models_dir}/{pat}"))]
        self.mlps, self.ftts = load("resmlp_s*.keras"), load("fttransformer_s*.keras")
        self.dae = keras.saving.load_model(f"{models_dir}/dae.keras", compile=False)
        self.mc_model = freeze_batchnorm(self.mlps[0])
        self.metrics = json.load(open(f"{reports_dir}/metrics.json"))
        self.threshold = next(r["threshold"] for r in self.metrics if r["name"] == "Stacked Ensemble")

    # ---- preprocessing must mirror data.load_splits exactly
    def preprocess(self, raw):
        """raw: (n, 30) array in original units, columns ordered as self.features (Time, V1..V28, Amount)."""
        df = np.array(raw, dtype="float64").copy()
        t, a = self.features.index("Time"), self.features.index("Amount")
        df[:, a] = np.log1p(df[:, a])
        df[:, t] = (df[:, t] % 86400) / 86400.0
        return np.clip(self.scaler.transform(df), -10, 10).astype("float32")

    def predict(self, raw, mc_samples=30, explain=True):
        X = self.preprocess(raw)
        pm = np.mean([m.predict(X, verbose=0).ravel() for m in self.mlps], axis=0)
        pf = np.mean([m.predict(X, verbose=0).ravel() for m in self.ftts], axis=0)
        err = np.mean((X - self.dae.predict(X, verbose=0)) ** 2, axis=1)
        p = self.stacker.predict_proba(np.column_stack([_logit(pm), _logit(pf), np.log(err + 1e-9)]))[:, 1]
        mc = np.stack([self.mc_model(X, training=True).numpy().ravel() for _ in range(mc_samples)])
        igs = integrated_gradients(self.mlps[0], X) if explain else None  # one batched pass for all rows
        out = []
        for i in range(len(X)):
            row = dict(probability=float(p[i]), resmlp=float(pm[i]), ft_transformer=float(pf[i]),
                       ae_error=float(err[i]), uncertainty=float(mc[:, i].std()),
                       is_fraud=bool(p[i] >= self.threshold))
            if explain:
                ig = igs[i]
                top = np.argsort(-np.abs(ig))[:5]
                row["top_features"] = [dict(feature=self.features[j], raw_value=float(np.asarray(raw)[i][j]),
                                            attribution=float(ig[j])) for j in top]
            out.append(row)
        return out

    def risk_level(self, p):
        return "critical" if p >= self.threshold else "high" if p >= 0.9 else "medium" if p >= 0.5 else "low"
