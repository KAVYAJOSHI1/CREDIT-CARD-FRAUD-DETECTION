import json, os, random, time
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from src.deep.inference import FraudEnsemble, FEATURES
from src.deep.data import load_splits
from src.deep import store

app = Flask(__name__)
CORS(app)

print("Loading deep ensemble (3x ResMLP, 2x FT-Transformer, autoencoder, stacker)...")
ens = FraudEnsemble()
assert ens.features == FEATURES

# Held-out TEST rows (never seen in training) in original units, used for demo replay.
_raw = pd.read_csv("data/creditcard.csv")
_split = load_splits()
_te_idx, _te_y = _split["idx_te"], _split["y_te"]
_test_raw = _raw.iloc[_te_idx][FEATURES].values
del _split


def seed_db(n_fraud=25, n_legit=175):
    """Populate an empty log with a replay of real held-out transactions (fraud enriched for demo)."""
    c = store.connect()
    if c.execute("SELECT COUNT(*) FROM scored_transactions").fetchone()[0]:
        return
    rng = np.random.default_rng(7)
    pick = np.concatenate([rng.choice(np.where(_te_y == 1)[0], n_fraud, replace=False),
                           rng.choice(np.where(_te_y == 0)[0], n_legit, replace=False)])
    rng.shuffle(pick)
    results = ens.predict(_test_raw[pick], mc_samples=10, explain=True)
    now = time.time()
    for k, (i, res) in enumerate(zip(pick, results)):
        store.insert(c, "replay", _test_raw[i], res, ens.risk_level(res["probability"]), _te_y[i],
                     created_at=now - k * 600)
    print(f"Seeded DB with {len(pick)} scored held-out transactions.")


seed_db()


@app.route("/")
def home():
    return render_template("index.html")


def _public(r):
    r = dict(r); r.pop("features", None); r["top_features"] = json.loads(r["top_features"] or "[]"); return r


@app.route("/api/transactions")
def transactions():
    return jsonify([_public(r) for r in store.rows(store.connect(), limit=int(request.args.get("limit", 50)))])


@app.route("/api/alerts")
def alerts():
    rs = store.rows(store.connect(), "WHERE status='blocked'", limit=20)
    return jsonify([_public(r) for r in rs])


@app.route("/api/sample")
def sample():
    """A random real held-out transaction (kind=fraud|legit) so the analyzer can be demoed without typing 30 numbers."""
    kind = request.args.get("kind", "fraud")
    i = random.choice(np.where(_te_y == (1 if kind == "fraud" else 0))[0])
    return jsonify(dict(zip(FEATURES, map(float, _test_raw[i])), true_label=int(_te_y[i])))


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    try:
        raw = np.array([[float(data[f]) for f in FEATURES]])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify(error=f"Need numeric values for all 30 features ({FEATURES[0]}, V1..V28, {FEATURES[-1]}): {e}"), 400
    res = ens.predict(raw, mc_samples=30, explain=True)[0]
    level = ens.risk_level(res["probability"])
    store.insert(store.connect(), "analyzer", raw[0], res, level, data.get("true_label"))
    return jsonify(dict(res, level=level, status="blocked" if res["is_fraud"] else "approved",
                        threshold=ens.threshold,
                        confidence="low (model unsure)" if res["uncertainty"] > 0.05 else "high"))


@app.route("/api/stats")
def stats():
    c = store.connect()
    q = lambda s: c.execute(s).fetchone()[0] or 0
    return jsonify(total=q("SELECT COUNT(*) FROM scored_transactions"),
                   blocked=q("SELECT COUNT(*) FROM scored_transactions WHERE status='blocked'"),
                   amount_blocked=q("SELECT SUM(amount) FROM scored_transactions WHERE status='blocked'"),
                   labelled=q("SELECT COUNT(*) FROM scored_transactions WHERE true_label IS NOT NULL"),
                   correct=q("SELECT COUNT(*) FROM scored_transactions WHERE true_label IS NOT NULL AND (status='blocked')=(true_label=1)"))


@app.route("/api/models")
def model_info():
    return jsonify(metrics=ens.metrics, uncertainty=json.load(open("reports/uncertainty.json")), threshold=ens.threshold)


@app.route("/reports/<path:name>")
def report_file(name):
    from flask import send_from_directory
    return send_from_directory("reports", name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
