"""
Extended SHAP case study runner: 15 cases instead of 5.
Run: python3 3b_extended_shap.py
"""
import os
import json
import numpy as np
import torch
import shap
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          TextClassificationPipeline)

os.makedirs("results/shap_figures", exist_ok=True)

MODEL_PATH = "model/bert_factchecker"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

pipe = TextClassificationPipeline(
    model=model, tokenizer=tokenizer, return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1)

explainer = shap.Explainer(pipe, masker=shap.maskers.Text(tokenizer))

CASE_STUDIES = [
    {"id": 1, "topic": "COVID-19 Origin",
     "claim": "The COVID-19 virus originated from the Wuhan Institute of Virology in China."},
    {"id": 2, "topic": "Surveillance Balloon Incident",
     "claim": "The Chinese balloon shot down over the United States was a civilian weather research vessel."},
    {"id": 3, "topic": "USA-China Trade Relations",
     "claim": "The United States imposed 25% tariffs on $200 billion worth of Chinese goods during the trade war."},
    {"id": 4, "topic": "Taiwan Strait Policy",
     "claim": "China has stated that Taiwan is an inalienable part of Chinese territory under the one-China principle."},
    {"id": 5, "topic": "Huawei Security Concerns",
     "claim": "The US government banned Huawei equipment citing national security risks related to Chinese espionage."},
    {"id": 6, "topic": "Belt and Road Initiative",
     "claim": "China's Belt and Road Initiative has invested over $1 trillion in global infrastructure since 2013."},
    {"id": 7, "topic": "Hong Kong Sovereignty",
     "claim": "Hong Kong was returned to Chinese sovereignty in 1997 under a 'one country, two systems' framework."},
    {"id": 8, "topic": "Trade Deficit",
     "claim": "The United States ran a trade deficit with China of approximately $300 billion in 2022."},
    {"id": 9, "topic": "South China Sea",
     "claim": "China constructed artificial islands in the South China Sea featuring military airstrips."},
    {"id": 10, "topic": "TikTok Divestiture",
     "claim": "TikTok's parent company ByteDance has been ordered by US courts to divest its American operations."},
    {"id": 11, "topic": "Beijing Olympics Boycott",
     "claim": "The 2022 Beijing Winter Olympics faced diplomatic boycotts from the US, UK and Canada."},
    {"id": 12, "topic": "Currency Manipulation",
     "claim": "The US Treasury formally designated China a currency manipulator in 2019."},
    {"id": 13, "topic": "Xinjiang Designation",
     "claim": "The US government has classified actions in Xinjiang against Uyghur populations as genocide."},
    {"id": 14, "topic": "Semiconductor Export Controls",
     "claim": "The US has imposed export controls preventing advanced semiconductor sales to Chinese firms."},
    {"id": 15, "topic": "Houston Consulate Closure",
     "claim": "The US closed China's consulate in Houston in 2020 over alleged espionage activities."},
]


def predict(claim):
    out = pipe(claim, truncation=True, max_length=128)
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        scores = out[0]
    elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
        scores = out
    else:
        scores = out
    top = max(scores, key=lambda d: d["score"])
    return top["label"], float(top["score"])


def explain(claim, save_html_path=None):
    sv = explainer([claim])
    if save_html_path:
        try:
            html = shap.plots.text(sv[0], display=False)
            with open(save_html_path, "w") as f:
                f.write(html)
        except Exception as e:
            print(f"  Could not save HTML: {e}")
    return sv


def top_k_tokens(sv, k=3):
    vals = sv.values[0]
    pred_class = int(np.argmax(np.abs(vals).sum(axis=0)))
    abs_attr = np.abs(vals[:, pred_class])
    tokens = sv.data[0]
    candidates = [(i, abs_attr[i]) for i, t in enumerate(tokens)
                  if t not in ("[CLS]", "[SEP]", "[PAD]", "")]
    candidates.sort(key=lambda x: -x[1])
    return [i for i, _ in candidates[:k]]


def mask_and_repredict(claim, sv, k=3):
    tokens = list(sv.data[0])
    top_idx = top_k_tokens(sv, k=k)
    original_tokens = [tokens[i] for i in top_idx]  # capture BEFORE masking
    for i in top_idx:
        tokens[i] = "[MASK]"
    masked_text = "".join(tokens).replace("[CLS]", "").replace("[SEP]", "").strip()
    label, conf = predict(masked_text)
    return label, conf, original_tokens


print("\n" + "=" * 60)
print("EXTENDED CASE STUDY ANALYSIS (15 cases)")
print("=" * 60)

case_results = []

for cs in CASE_STUDIES:
    print(f"\n--- Case {cs['id']:>2}: {cs['topic']} ---")
    label, conf = predict(cs["claim"])
    print(f"  Predicted: {label}  ({conf*100:.1f}%)")

    save_path = f"results/shap_figures/case_study_{cs['id']}.html"
    sv = explain(cs["claim"], save_html_path=save_path)

    masked_label, masked_conf, masked_tokens = mask_and_repredict(
        cs["claim"], sv, k=3)
    flipped = (masked_label != label)
    conf_drop = conf - masked_conf if not flipped else conf - (1 - masked_conf)

    print(f"  Top-3 masked: {masked_tokens}")
    print(f"  After mask: {masked_label} ({masked_conf*100:.1f}%) "
          f"flipped={flipped}, drop={conf_drop*100:.1f}pp")

    case_results.append({
        "case_study": cs["id"],
        "topic": cs["topic"],
        "claim": cs["claim"],
        "prediction": label,
        "confidence": round(conf * 100, 2),
        "top3_masked_tokens": masked_tokens,
        "post_mask_prediction": masked_label,
        "post_mask_confidence": round(masked_conf * 100, 2),
        "label_flipped": flipped,
        "confidence_drop_pp": round(conf_drop * 100, 2),
    })

with open("results/case_studies_extended.json", "w") as f:
    json.dump(case_results, f, indent=2)

n_total = len(case_results)
n_flipped = sum(1 for c in case_results if c["label_flipped"])
mean_drop = np.mean([c["confidence_drop_pp"] for c in case_results])

print("\n" + "=" * 60)
print("AGGREGATE SUMMARY")
print("=" * 60)
print(f"Total cases: {n_total}")
print(f"Label flipped under top-3 mask: {n_flipped}/{n_total} "
      f"({100*n_flipped/n_total:.0f}%)")
print(f"Mean confidence drop: {mean_drop:.1f} percentage points")
print("\nResults saved to results/case_studies_extended.json")