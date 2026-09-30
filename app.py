from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import time
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

print("Loading Deep Neural Network Model...")
# Ensure TensorFlow does not allocate all GPU memory if one exists
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

dl_model = load_model('models/deep_learning_model.keras')

print("Loading Scaler...")
with open('models/dl_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    mock_transactions = []
    for i in range(20):
        is_fraud = random.random() > 0.85
        score = (0.9 + random.random()*0.1) * 100 if is_fraud else random.random() * 30
        
        mock_transactions.append({
            "account": f"4532-{random.randint(1000,9999)}-{random.randint(1000,9999)}",
            "dist_home": round(random.uniform(200, 3000) if is_fraud else random.uniform(0.5, 20), 1),
            "dist_last": round(random.uniform(50, 1000) if is_fraud else random.uniform(0.1, 10), 1),
            "ratio": round(random.uniform(10, 50) if is_fraud else random.uniform(0.1, 3), 2),
            "repeat": 0 if is_fraud else 1,
            "chip": 0 if is_fraud else 1,
            "pin": 0 if is_fraud else 1,
            "online": 1 if is_fraud else random.choice([0, 1]),
            "risk_level": "critical" if score > 80 else ("high" if score > 60 else "low"),
            "risk_score": score,
            "status": "blocked" if score > 80 else "approved",
            "created_at": time.time() * 1000 - (i * 3600000)
        })
    return jsonify(mock_transactions)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    
    dist_home = float(data.get('dist_home', 0))
    dist_last = float(data.get('dist_last', 0))
    ratio = float(data.get('ratio', 0))
    repeat = float(data.get('repeat', 0))
    chip = float(data.get('chip', 0))
    pin = float(data.get('pin', 0))
    online = float(data.get('online', 0))
    
    # 1. Scale the input using the scaler saved during training
    X = np.array([[dist_home, dist_last, ratio, repeat, chip, pin, online]])
    X_scaled = scaler.transform(X)
    
    # 2. Predict with Deep Learning Model
    prob = float(dl_model.predict(X_scaled, verbose=0)[0][0])
    score = prob * 100.0
    
    level = "critical" if score >= 80 else ("high" if score >= 55 else ("medium" if score >= 30 else "low"))
    
    explanation = "Deep Neural Network Analysis complete."
    if score > 80:
        explanation += " The behavioral and spatial parameters trigger a definitive block."
    elif score < 30:
        explanation += " The transaction falls within normal behavioral baselines."
        
    reasons = [
        f"Spatial: {dist_home} miles from home",
        f"Behavioral: {ratio}x normal spend ratio",
        f"Auth: Chip={bool(chip)}, PIN={bool(pin)}"
    ]
    
    return jsonify({
        "score": score,
        "level": level,
        "explanation": explanation,
        "reasons": reasons
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
