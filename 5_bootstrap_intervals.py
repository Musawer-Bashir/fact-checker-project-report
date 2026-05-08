"""
STEP 5: Bootstrap 95% confidence intervals for all models.
Run: python3 5_bootstrap_intervals.py
"""
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          TextClassificationPipeline)

RNG = np.random.default_rng(42)
N_BOOT = 1000


def bootstrap_ci(y_true, y_pred, n_boot=N_BOOT):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    f1s = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    f1s = np.array(f1s)
    return {
        "mean":    round(float(f1s.mean()), 4),
        "ci_low":  round(float(np.percentile(f1s, 2.5)), 4),
        "ci_high": round(float(np.percentile(f1s, 97.5)), 4),
    }


df = pd.read_csv("data/dataset.csv")
train_df, temp = train_test_split(df, test_size=0.30, random_state=42,
                                  stratify=df["label_binary"])
val_df, test_df = train_test_split(temp, test_size=0.50, random_state=42,
                                   stratify=temp["label_binary"])

X_train, y_train = train_df["text"].tolist(), train_df["label_binary"].tolist()
X_test,  y_test  = test_df["text"].tolist(),  test_df["label_binary"].tolist()

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

results = {}

dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
results["majority_class"] = bootstrap_ci(y_test, dummy.predict(X_test))

lr = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, y_train)
results["tfidf_logreg"] = bootstrap_ci(y_test, lr.predict(Xte))

nb = MultinomialNB().fit(Xtr, y_train)
results["naive_bayes"] = bootstrap_ci(y_test, nb.predict(Xte))

svm = LinearSVC().fit(Xtr, y_train)
results["linear_svm"] = bootstrap_ci(y_test, svm.predict(Xte))

print("\nLoading fine-tuned BERT...")
tok = BertTokenizerFast.from_pretrained("model/bert_factchecker")
mdl = BertForSequenceClassification.from_pretrained("model/bert_factchecker")
mdl.eval()
pipe = TextClassificationPipeline(
    model=mdl, tokenizer=tok, return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1)
label2id = {"False": 0, "True": 1}

bert_preds = []
print("Running BERT inference on test set...")
for i, txt in enumerate(X_test):
    out = pipe(txt, truncation=True, max_length=128)
    # Handle both old and new transformers output formats
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        scores = out[0]  # old format: [[{label, score}, {label, score}]]
    elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
        scores = out      # new format: [{label, score}, {label, score}]
    else:
        scores = out
    top = max(scores, key=lambda d: d["score"])["label"]
    bert_preds.append(label2id[top])
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(X_test)}")

results["bert"] = bootstrap_ci(y_test, bert_preds)

print("\n" + "=" * 70)
print(f"{'Model':<20} {'Mean F1':>10} {'CI Low':>10} {'CI High':>10}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:<20} {r['mean']:>10.4f} {r['ci_low']:>10.4f} {r['ci_high']:>10.4f}")
print("=" * 70)

import os
os.makedirs("results", exist_ok=True)
with open("results/bootstrap_ci.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/bootstrap_ci.json")