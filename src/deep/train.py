"""End-to-end deep learning pipeline: train -> stack -> evaluate -> explain.

Run from project root:  python -m src.deep.train          (full)
                        QUICK=1 python -m src.deep.train  (smoke test)
"""
import json, os, pickle, time
import numpy as np
import tensorflow as tf
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             roc_curve, confusion_matrix, f1_score, matthews_corrcoef, brier_score_loss)
from sklearn.calibration import calibration_curve

from .data import load_splits
from .models import residual_mlp, tabular_transformer, denoising_autoencoder, focal_loss

QUICK = bool(os.environ.get("QUICK"))
EPOCHS = 2 if QUICK else 60
MODELS, REPORTS = "models/deep", "reports"
os.makedirs(MODELS, exist_ok=True); os.makedirs(REPORTS, exist_ok=True)


def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def cosine_lr(base, total):
    return keras.callbacks.LearningRateScheduler(
        lambda e: float(base * 0.5 * (1 + np.cos(np.pi * min(e, total) / total))) + 1e-6)


def fit_classifier(builder, d, seed, lr=2e-3, bs=1024, epochs=None, **kw):
    epochs = min(epochs or EPOCHS, EPOCHS)
    keras.utils.set_random_seed(seed)
    m = builder(d["X_tr"].shape[1], **kw)
    m.compile(keras.optimizers.AdamW(lr, weight_decay=1e-4), loss=focal_loss,
              metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")])
    m.fit(d["X_tr"], d["y_tr"], validation_data=(d["X_va"], d["y_va"]), epochs=epochs, batch_size=bs,
          verbose=2, callbacks=[cosine_lr(lr, epochs),
          keras.callbacks.EarlyStopping("val_pr_auc", mode="max", patience=12, restore_best_weights=True)])
    return m


def metrics_at(y, p, thr):
    pred = p >= thr
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return dict(threshold=float(thr), precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)),
                f1=float(f1_score(y, pred)), mcc=float(matthews_corrcoef(y, pred)),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    return th[np.argmax(f1[:-1])]


def summarize(name, y, p, thr):
    return dict(name=name, pr_auc=float(average_precision_score(y, p)), roc_auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)) if p.max() <= 1 else None, **metrics_at(y, p, thr))


def mc_dropout(model, X, n=30):
    ps = np.stack([model(X, training=True).numpy().ravel() for _ in range(n)])
    return ps.mean(0), ps.std(0)


def integrated_gradients(model, X, baseline, steps=32):
    X = tf.constant(X); base = tf.constant(np.tile(baseline, (len(X), 1)))
    total = tf.zeros_like(X)
    for a in np.linspace(0, 1, steps):
        xi = base + a * (X - base)
        with tf.GradientTape() as t:
            t.watch(xi); out = model(xi, training=False)
        total += t.gradient(out, xi)
    return ((X - base) * total / steps).numpy()


def main():
    t0 = time.time()
    d = load_splits()
    n_feat = d["X_tr"].shape[1]
    print(f"train {d['X_tr'].shape} fraud={int(d['y_tr'].sum())} | val fraud={int(d['y_va'].sum())} | test fraud={int(d['y_te'].sum())}")
    with open(f"{MODELS}/scaler.pkl", "wb") as f: pickle.dump(dict(scaler=d["scaler"], features=d["features"]), f)

    # ---- 1. unsupervised branch: denoising autoencoder on legitimate txns only
    ae = denoising_autoencoder(n_feat)
    ae.compile(keras.optimizers.Adam(1e-3), loss="mse")
    Xn = d["X_tr"][d["y_tr"] == 0]
    ae.fit(Xn, Xn, epochs=3 if QUICK else 25, batch_size=1024, validation_split=0.1, verbose=2,
           callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    ae.save(f"{MODELS}/dae.keras")
    ae_err = lambda X: np.mean((X - ae.predict(X, verbose=0, batch_size=4096)) ** 2, axis=1)
    err = {k: ae_err(d[f"X_{k}"]) for k in ("tr", "va", "te")}
    print("AE-only PR-AUC (test):", average_precision_score(d["y_te"], err["te"]))

    # ---- 2. supervised deep models (seed ensembles)
    members = {}
    n_seeds = 1 if QUICK else 3
    for s in range(n_seeds):
        members[f"resmlp_s{s}"] = fit_classifier(residual_mlp, d, seed=s)
    for s in range(1 if QUICK else 2):
        members[f"fttransformer_s{s}"] = fit_classifier(tabular_transformer, d, seed=100 + s, lr=1e-3, bs=1024, epochs=25)
    for k, m in members.items(): m.save(f"{MODELS}/{k}.keras")

    P = {k: {s: m.predict(d[f"X_{s}"], verbose=0, batch_size=4096).ravel() for s in ("va", "te")}
         for k, m in members.items()}
    fam = lambda pref, s: np.mean([P[k][s] for k in P if k.startswith(pref)], axis=0)
    base_p = {"ResMLP (seed-avg)": {s: fam("resmlp", s) for s in ("va", "te")},
              "FT-Transformer (seed-avg)": {s: fam("fttransformer", s) for s in ("va", "te")},
              "Autoencoder (unsup.)": {"va": err["va"], "te": err["te"]}}

    # ---- 3. stacking: logistic meta-learner over [logit(resmlp), logit(ftt), log(AE err)]
    meta = lambda s: np.column_stack([logit(base_p["ResMLP (seed-avg)"][s]),
                                      logit(base_p["FT-Transformer (seed-avg)"][s]),
                                      np.log(err[s] + 1e-9)])
    stack = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    # threshold chosen on out-of-fold val predictions -> test set stays untouched
    oof = cross_val_predict(stack, meta("va"), d["y_va"], cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            method="predict_proba")[:, 1]
    stack.fit(meta("va"), d["y_va"])
    base_p["Stacked Ensemble"] = {"va": oof, "te": stack.predict_proba(meta("te"))[:, 1]}
    with open(f"{MODELS}/stacker.pkl", "wb") as f: pickle.dump(stack, f)

    # ---- 4. XGBoost baseline (the thing faculty will ask about)
    import xgboost as xgb
    xg = xgb.XGBClassifier(n_estimators=100 if QUICK else 600, max_depth=6, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, scale_pos_weight=(d["y_tr"] == 0).sum() / d["y_tr"].sum(),
                           eval_metric="aucpr", n_jobs=8, random_state=0).fit(d["X_tr"], d["y_tr"])
    base_p["XGBoost baseline"] = {s: xg.predict_proba(d[f"X_{s}"])[:, 1] for s in ("va", "te")}

    # ---- 5. evaluation (thresholds tuned on val, numbers reported on test)
    rows = []
    for name, ps in base_p.items():
        thr = best_f1_threshold(d["y_va"], ps["va"])
        rows.append(summarize(name, d["y_te"], ps["te"], thr))
    rows.sort(key=lambda r: -r["pr_auc"])
    json.dump(rows, open(f"{REPORTS}/metrics.json", "w"), indent=2)
    print(f"\n{'Model':28s} PR-AUC ROC-AUC  Prec   Rec    F1    MCC")
    for r in rows:
        print(f"{r['name']:28s} {r['pr_auc']:.4f} {r['roc_auc']:.4f}  {r['precision']:.3f}  {r['recall']:.3f}  {r['f1']:.3f}  {r['mcc']:.3f}")

    # ---- 6. figures
    y = d["y_te"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for name, ps in base_p.items():
        pr, rc, _ = precision_recall_curve(y, ps["te"]); ax[0].plot(rc, pr, label=f"{name} ({average_precision_score(y, ps['te']):.3f})")
        fpr, tpr, _ = roc_curve(y, ps["te"]); ax[1].plot(fpr, tpr, label=name)
    ax[0].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision"); ax[0].legend(fontsize=8)
    ax[1].set(title="ROC (test)", xlabel="FPR", ylabel="TPR", xscale="log", xlim=(1e-5, 1)); ax[1].legend(fontsize=8)
    pt, pp = calibration_curve(y, base_p["Stacked Ensemble"]["te"], n_bins=8, strategy="quantile")
    ax[2].plot(pp, pt, "o-", label="Stacked"); ax[2].plot([0, 1], [0, 1], "k--"); ax[2].set(title="Calibration", xlabel="Predicted", ylabel="Observed")
    plt.tight_layout(); plt.savefig(f"{REPORTS}/model_comparison.png", dpi=140); plt.close()

    # ---- 7. uncertainty: MC-dropout on ResMLP
    mlp = members["resmlp_s0"]
    mu, sd = mc_dropout(mlp, d["X_te"], n=10 if QUICK else 30)
    wrong = (mu >= 0.5) != (y == 1)
    q = np.quantile(sd, 0.9)
    unc = dict(error_rate_low_uncertainty=float(wrong[sd < q].mean()), error_rate_top10pct_uncertainty=float(wrong[sd >= q].mean()),
               mean_std_fraud=float(sd[y == 1].mean()), mean_std_legit=float(sd[y == 0].mean()))
    json.dump(unc, open(f"{REPORTS}/uncertainty.json", "w"), indent=2); print("\nMC-dropout:", unc)

    # ---- 8. explainability: Integrated Gradients (global attribution on true frauds)
    rng = np.random.default_rng(0)
    fr = d["X_te"][y == 1]; lg = d["X_te"][rng.choice(np.where(y == 0)[0], 300, replace=False)]
    base = np.median(d["X_tr"][d["y_tr"] == 0], axis=0, keepdims=True).astype("float32")
    ig = integrated_gradients(mlp, fr, base)
    imp = np.abs(ig).mean(0); order = np.argsort(imp)[::-1][:15]
    plt.figure(figsize=(8, 6)); plt.barh([d["features"][i] for i in order][::-1], imp[order][::-1], color="#c0392b")
    plt.title("Integrated Gradients — what drives fraud predictions"); plt.tight_layout()
    plt.savefig(f"{REPORTS}/feature_attribution.png", dpi=140); plt.close()

    # ---- 9. latent space: AE bottleneck projected with PCA
    from sklearn.decomposition import PCA
    enc = keras.Model(ae.input, ae.get_layer("bottleneck").output)
    idx = np.concatenate([np.where(y == 1)[0], rng.choice(np.where(y == 0)[0], 3000, replace=False)])
    z = PCA(2).fit_transform(enc.predict(d["X_te"][idx], verbose=0))
    plt.figure(figsize=(7, 6)); plt.scatter(*z[y[idx] == 0].T, s=4, alpha=.3, label="legit"); plt.scatter(*z[y[idx] == 1].T, s=14, c="r", label="fraud")
    plt.legend(); plt.title("Autoencoder latent space (PCA of bottleneck)"); plt.tight_layout()
    plt.savefig(f"{REPORTS}/latent_space.png", dpi=140); plt.close()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Artifacts: {MODELS}/, {REPORTS}/")


if __name__ == "__main__":
    main()
