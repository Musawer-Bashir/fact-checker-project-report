"""
STEP 4: FLASK WEB APPLICATION
================================
Loads the trained BERT model and SHAP explainer once at startup,
then serves a web interface for claim submission and explanation.

Run: python app.py
Then open: http://localhost:5000
Requires: pip install flask transformers torch shap
"""

import os
import json
import torch
import shap
from flask import Flask, render_template, request, jsonify
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          TextClassificationPipeline)

app = Flask(__name__)

# ── Load model ONCE at startup (not per request) ─────────────────────────────
MODEL_PATH = "model/bert_factchecker"
print("Loading BERT model...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model     = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1
)

print("Initialising SHAP explainer...")
explainer = shap.Explainer(pipe, masker=shap.maskers.Text(tokenizer))
print("Ready. Visit http://localhost:5000")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/check", methods=["POST"])
def check_claim():
    data  = request.get_json()
    claim = data.get("claim", "").strip()

    if not claim or len(claim) < 10:
        return jsonify({"error": "Please enter a valid claim (at least 10 characters)."}), 400

    # ── Prediction ──────────────────────────────────────────────────────────
    result = pipe(claim)[0]
    scores = {r["label"]: round(r["score"] * 100, 1) for r in result}
    pred_label  = max(scores, key=scores.get)
    confidence  = scores[pred_label]
    false_score = scores.get("False", 0)
    true_score  = scores.get("True", 0)

    # ── SHAP Explanation ─────────────────────────────────────────────────────
    shap_values = explainer([claim])

    # Build token attribution list for frontend
    tokens      = shap_values.data[0]
    attributions = shap_values.values[0, :, 1].tolist()  # contributions to True

    token_data = [
        {"token": tok, "score": round(float(score), 4)}
        for tok, score in zip(tokens, attributions)
        if tok not in ["[CLS]", "[SEP]", "[PAD]"]
    ]

    # SHAP HTML plot (embed in page)
    shap_html = shap.plots.text(shap_values[0], display=False)

    return jsonify({
        "claim":       claim,
        "prediction":  pred_label,
        "confidence":  confidence,
        "true_score":  true_score,
        "false_score": false_score,
        "tokens":      token_data,
        "shap_html":   shap_html,
    })

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
