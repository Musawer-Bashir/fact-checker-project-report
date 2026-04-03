"""
STEP 1: BUILD YOUR DATASET

This script downloads FEVER and PolitiFact data, filters for
USA-China geopolitical claims, maps to binary labels, and saves
a clean CSV ready for training.

Run: python 1_build_dataset.py
Requires: pip install datasets pandas scikit-learn
"""

import pandas as pd
import re
from datasets import load_dataset
from sklearn.utils import resample

# Keywdrds for USA-China geopolitical filtering
KEYWORDS = [
    "china", "chinese", "beijing", "people's republic", "prc",
    "covid", "coronavirus", "wuhan",
    "trade war", "huawei", "tiktok",
    "taiwan", "south china sea", "surveillance balloon",
    "belt and road", "xi jinping",
    "sino", "ccp", "communist party china",
    "uyghur", "xinjiang", "hong kong",
    "fentanyl china", "spy balloon",
    "us china", "china us", "china trade",
    "chinese government", "chinese military"
]

def is_geopolitical(text):
    """Return True if claim contains at least one USA-China keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in KEYWORDS)

def clean_text(text):
    """Basic text cleaning."""
    text = re.sub(r'http\S+', '', text)          # remove URLs
    text = re.sub(r'<[^>]+>', '', text)           # remove HTML tags
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # remove non-ASCII
    text = re.sub(r'\s+', ' ', text).strip()      # normalise whitespace
    return text

# 1. FEVER Dataset
print("Loading FEVER dataset...")
fever = load_dataset("fever", "v1.0", split="train", trust_remote_code=True)
fever_df = pd.DataFrame(fever)

# Keep only SUPPORTED and REFUTED
fever_df = fever_df[fever_df["label"].isin(["SUPPORTS", "REFUTES"])].copy()
fever_df["text"] = fever_df["claim"].apply(clean_text)
fever_df["label_binary"] = fever_df["label"].map({"SUPPORTS": 1, "REFUTES": 0})

# Filter for geopolitical relevance
fever_geo = fever_df[fever_df["text"].apply(is_geopolitical)][["text", "label_binary"]]
fever_geo["source"] = "FEVER"
print(f"  FEVER geopolitical claims: {len(fever_geo)}")

# 2. LIAR Dataset (PolitiFact-derived, publicly available)
print("Loading LIAR dataset (PolitiFact)...")
liar = load_dataset("liar", split="train", trust_remote_code=True)
liar_df = pd.DataFrame(liar)

# LIAR label mapping: 0=false, 1=half-true, 2=mostly-true, 3=true,
#                     4=barely-true, 5=pants-fire
# Binary: true/mostly-true -> 1, false/pants-fire/barely-true -> 0
# Exclude half-true (1) to keep clean binary signal
TRUE_LABELS  = [2, 3]   # mostly-true, true
FALSE_LABELS = [0, 4, 5] # false, barely-true, pants-fire

liar_df = liar_df[liar_df["label"].isin(TRUE_LABELS + FALSE_LABELS)].copy()
liar_df["text"] = liar_df["statement"].apply(clean_text)
liar_df["label_binary"] = liar_df["label"].apply(
    lambda x: 1 if x in TRUE_LABELS else 0
)
liar_geo = liar_df[liar_df["text"].apply(is_geopolitical)][["text", "label_binary"]]
liar_geo["source"] = "LIAR/PolitiFact"
print(f"  LIAR geopolitical claims: {len(liar_geo)}")

# 3. Combine and deduplicate
combined = pd.concat([fever_geo, liar_geo], ignore_index=True)
combined = combined.drop_duplicates(subset="text")
combined = combined[combined["text"].str.len() > 20]  # remove very short claims
print(f"\nCombined dataset: {len(combined)} claims")
print(combined["label_binary"].value_counts().rename({1: "True", 0: "False"}))

#  4. Balance classes 
true_df  = combined[combined["label_binary"] == 1]
false_df = combined[combined["label_binary"] == 0]
min_size = min(len(true_df), len(false_df))

# Undersample majority class for balance
true_df  = resample(true_df,  n_samples=min_size, random_state=42)
false_df = resample(false_df, n_samples=min_size, random_state=42)
balanced = pd.concat([true_df, false_df]).sample(frac=1, random_state=42)

print(f"\nBalanced dataset: {len(balanced)} claims ({min_size} per class)")

#  5. Save 
balanced.to_csv("data/dataset.csv", index=False)
print("\nSaved to data/dataset.csv")
print("\nSample claims:")
print(balanced[["text", "label_binary", "source"]].head(10).to_string())
