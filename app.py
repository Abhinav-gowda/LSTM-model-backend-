import os
import re
import string
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://smart-incrident-analyzer.vercel.app", "http://localhost:5173", "http://localhost:3000", "*"])

# ─── Load model and tokenizer ───────────────────────────────────────────────
print("Loading model and tokenizer...")

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model = load_model("lstm_model.keras")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model load error: {e}")
    model = None

try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("✅ Tokenizer loaded successfully")
except Exception as e:
    print(f"❌ Tokenizer load error: {e}")
    tokenizer = None

try:
    df_ingredients = pd.read_csv("ingredient_effects.csv")
    print(f"✅ Dataset loaded: {df_ingredients.shape[0]} ingredients")
except Exception as e:
    print(f"❌ Dataset load error: {e}")
    df_ingredients = pd.DataFrame(columns=["Ingredient_Name", "Harmfulness_Score", "Effect_On_Human_Body"])

MAX_SEQUENCE_LENGTH = 12  # from training


# ─── Helper functions ────────────────────────────────────────────────────────
def clean_text(text):
    """Lowercase, remove punctuation, strip whitespace."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text.strip()


def predict_harm_score(raw_effect_text):
    """Predict harm score from effect text using LSTM model."""
    if model is None or tokenizer is None:
        return 5.0  # neutral fallback
    cleaned = clean_text(raw_effect_text)
    encoded = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(encoded, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    score = model.predict(padded, verbose=0)[0][0]
    return float(np.clip(score, 1, 10))


def calculate_product_scores(scores):
    if not scores:
        return 0.0, 0.0, 0.0
    avg = float(np.mean(scores))
    mx = float(np.max(scores))
    final = (avg + mx) / 2
    return avg, mx, final


def classify_product_risk(final_score):
    if final_score <= 5:
        return "Safe"
    elif final_score <= 7:
        return "Moderate Risk"
    else:
        return "Harmful"


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Ingredient IQ API is running 🚀"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST body (JSON or URL-encoded):
    {
        "ingredients": ["Water", "Parabens", "Unknown Chemical X"]
    }
    
    Returns analysis results.
    """
    try:
        # Handle both JSON and URL-encoded formats
        content_type = request.headers.get('Content-Type', '')
        
        if 'application/json' in content_type:
            data = request.get_json(force=True)
            ingredients_raw = data.get("ingredients", [])
        elif 'application/x-www-form-urlencoded' in content_type:
            # Parse URL-encoded data
            ingredients_str = request.form.get('ingredients', '[]')
            import json
            ingredients_raw = json.loads(ingredients_str)
        else:
            # Try to parse as JSON anyway
            data = request.get_json(force=True)
            ingredients_raw = data.get("ingredients", []) if data else []

        if not ingredients_raw:
            return jsonify({"error": "No ingredients provided"}), 400

        # Normalize input
        ingredients = [i.strip() for i in ingredients_raw if isinstance(i, str) and i.strip()]

        if not ingredients:
            return jsonify({"error": "No valid ingredient names found"}), 400

        results = []
        harm_scores = []

        for name in ingredients:
            # ── Look up in dataset ──────────────────────────────────────────
            match = df_ingredients[
                df_ingredients["Ingredient_Name"].str.lower() == name.lower()
            ]

            if not match.empty:
                row = match.iloc[0]
                score = float(row["Harmfulness_Score"])
                effect = str(row["Effect_On_Human_Body"])
                source = "dataset"
            else:
                # ── Predict for unknown ingredient ──────────────────────────
                effect = "Unknown compound with uncertain biological activity."
                score = predict_harm_score(effect)
                source = "predicted"

            harm_scores.append(score)
            results.append(
                {
                    "name": name,
                    "effect": effect,
                    "score": round(score, 2),
                    "source": source,
                    "risk_level": classify_product_risk(score),
                }
            )

        # ── Product-level scores ───────────────────────────────────────────
        avg_score, max_score, final_score = calculate_product_scores(harm_scores)
        classification = classify_product_risk(final_score)

        high_risk = [r for r in results if r["score"] >= 7]

        return jsonify(
            {
                "ingredients": results,
                "product_summary": {
                    "average_score": round(avg_score, 2),
                    "maximum_score": round(max_score, 2),
                    "final_score": round(final_score, 2),
                    "classification": classification,
                    "high_risk_ingredients": high_risk,
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST body (JSON):
    {
        "ingredients": ["Water", "Parabens", "Unknown Chemical X"]
    }
    """
    try:
        data = request.get_json(force=True)
        ingredients_raw = data.get("ingredients", [])

        if not ingredients_raw:
            return jsonify({"error": "No ingredients provided"}), 400

        # Normalize input
        ingredients = [i.strip() for i in ingredients_raw if isinstance(i, str) and i.strip()]

        if not ingredients:
            return jsonify({"error": "No valid ingredient names found"}), 400

        results = []
        harm_scores = []

        for name in ingredients:
            # ── Look up in dataset ──────────────────────────────────────────
            match = df_ingredients[
                df_ingredients["Ingredient_Name"].str.lower() == name.lower()
            ]

            if not match.empty:
                row = match.iloc[0]
                score = float(row["Harmfulness_Score"])
                effect = str(row["Effect_On_Human_Body"])
                source = "dataset"
            else:
                # ── Predict for unknown ingredient ──────────────────────────
                effect = "Unknown compound with uncertain biological activity."
                score = predict_harm_score(effect)
                source = "predicted"

            harm_scores.append(score)
            results.append(
                {
                    "name": name,
                    "effect": effect,
                    "score": round(score, 2),
                    "source": source,
                    "risk_level": classify_product_risk(score),
                }
            )

        # ── Product-level scores ────────────────────────────────────────────
        avg_score, max_score, final_score = calculate_product_scores(harm_scores)
        classification = classify_product_risk(final_score)

        high_risk = [r for r in results if r["score"] >= 7]

        return jsonify(
            {
                "ingredients": results,
                "product_summary": {
                    "average_score": round(avg_score, 2),
                    "maximum_score": round(max_score, 2),
                    "final_score": round(final_score, 2),
                    "classification": classification,
                    "high_risk_ingredients": high_risk,
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ingredient/<name>", methods=["GET"])
def get_ingredient(name):
    """Look up a single ingredient."""
    match = df_ingredients[
        df_ingredients["Ingredient_Name"].str.lower() == name.lower()
    ]
    if match.empty:
        return jsonify({"found": False, "name": name}), 404

    row = match.iloc[0]
    return jsonify(
        {
            "found": True,
            "name": str(row["Ingredient_Name"]),
            "effect": str(row["Effect_On_Human_Body"]),
            "score": float(row["Harmfulness_Score"]),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
