"""
STEP 3: GENERATE SHAP EXPLANATIONS

Loads the trained model, generates SHAP token attributions for
individual claims, and runs the 5 geopolitical case studies for
your report's Section 5.2.

Run: python 3_shap_analysis.py
Requires: pip install shap transformers torch matplotlib
"""

import os, json
import numpy as np
import pandas as pd
import shap
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures

from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          TextClassificationPipeline)

os.makedirs("results/shap_figures", exist_ok=True)

#  Load trained model 
print("Loading trained model...")
MODEL_PATH = "model/bert_factchecker"
tokenizer  = BertTokenizerFast.from_pretrained(MODEL_PATH)
model      = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Wrap in HuggingFace pipeline (required for shap.Explainer)
pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1
)

#  SHAP Explainer 
# Uses partition-based approximation of Shapley values
# masker = tokenizer mask (replaces tokens with [MASK] for perturbations)
print("Initialising SHAP explainer (this may take a moment)...")
explainer = shap.Explainer(pipe, masker=shap.maskers.Text(tokenizer))

#  Helper: explain a single claim 
def explain_claim(claim_text, save_path=None):
    """
    Takes a claim string.
    Returns: prediction label, confidence %, shap_values object.
    Optionally saves the SHAP text plot as an HTML file.
    """
    # Get prediction
    result = pipe(claim_text)[0]
    scores = {r["label"]: r["score"] for r in result}
    pred_label = max(scores, key=scores.get)
    confidence = scores[pred_label] * 100

    # Get SHAP values
    shap_values = explainer([claim_text])

    print(f"\nClaim:      {claim_text}")
    print(f"Prediction: {pred_label} ({confidence:.1f}% confidence)")

    # Top contributing tokens
    # shap_values.values shape: (1, n_tokens, n_classes)
    # Class index 1 = True label
    token_names   = shap_values.data[0]
    token_scores  = shap_values.values[0, :, 1]  # contributions to True class

    top_positive = sorted(zip(token_names, token_scores),
                          key=lambda x: x[1], reverse=True)[:5]
    top_negative = sorted(zip(token_names, token_scores),
                          key=lambda x: x[1])[:5]

    print("Top tokens -> True:")
    for tok, score in top_positive:
        print(f"  {tok:20s}  {score:+.4f}")
    print("Top tokens -> False:")
    for tok, score in top_negative:
        print(f"  {tok:20s}  {score:+.4f}")

    if save_path:
        # Save HTML visualisation
        html = shap.plots.text(shap_values[0], display=False)
        with open(save_path, "w") as f:
            f.write(html)
        print(f"SHAP plot saved: {save_path}")

    return pred_label, confidence, shap_values

#  5 Case Studies for Report Section 5.2 
# These are the claims you will analyse in your report.
# Replace with actual claims from your test set for best results.
CASE_STUDIES = [
    {
        "id": 1,
        "topic": "COVID-19 Origin",
        "claim": "The COVID-19 virus originated from the Wuhan Institute of Virology in China.",
        "notes": "Analyse: does SHAP highlight Wuhan, virus, China correctly?"
    },
    {
        "id": 2,
        "topic": "Surveillance Balloon Incident",
        "claim": "The Chinese balloon shot down over the United States was a civilian weather research vessel.",
        "notes": "Analyse: does SHAP weight balloon, Chinese, United States?"
    },
    {
        "id": 3,
        "topic": "USA-China Trade Relations",
        "claim": "The United States imposed 25% tariffs on $200 billion worth of Chinese goods during the trade war.",
        "notes": "Analyse: does SHAP weight numerical and trade-specific terms?"
    },
    {
        "id": 4,
        "topic": "Taiwan Strait Policy",
        "claim": "China has stated that Taiwan is an inalienable part of Chinese territory under the one-China principle.",
        "notes": "Analyse: does SHAP handle diplomatic/legal language correctly?"
    },
    {
        "id": 5,
        "topic": "Huawei Security Concerns",
        "claim": "The US government banned Huawei equipment citing national security risks related to Chinese espionage.",
        "notes": "Analyse: does SHAP identify security-relevant technical terms?"
    },
]

print("\n" + "="*60)
print("CASE STUDY ANALYSIS")
print("="*60)

case_results = []
for cs in CASE_STUDIES:
    print(f"\n--- Case Study {cs['id']}: {cs['topic']} ---")
    save_path = f"results/shap_figures/case_study_{cs['id']}.html"
    label, conf, sv = explain_claim(cs["claim"], save_path=save_path)

    case_results.append({
        "case_study": cs["id"],
        "topic": cs["topic"],
        "claim": cs["claim"],
        "prediction": label,
        "confidence": round(conf, 2),
        "shap_figure": save_path,
        "analysis_notes": cs["notes"]
    })

# Save case study results
with open("results/case_studies.json", "w") as f:
    json.dump(case_results, f, indent=2)
print(f"\n\nCase study results saved to results/case_studies.json")

#  Generate confusion matrix figure 
print("\nGenerating confusion matrix figure...")
import json
try:
    with open("results/metrics.json") as f:
        metrics = json.load(f)

    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    classes = ["False", "True"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes, fontsize=12)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")

    ax.set_ylabel("Actual Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix — BERT Fact-Checker (Test Set)", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Confusion matrix saved to results/confusion_matrix.png")
except FileNotFoundError:
    print("Run 2_train_model.py first to generate metrics.json")

#  Training curve figure (from training logs) 
print("\nTo generate training curves, add this to 2_train_model.py:")
print("  trainer.state.log_history contains all epoch metrics.")
print("  Already included in the full trainer output above.")

print("\n=== SHAP ANALYSIS COMPLETE ===")
print("Files saved in results/shap_figures/")
print("Insert HTML files into your report by opening them in a browser")
print("and taking screenshots for your figures.")
