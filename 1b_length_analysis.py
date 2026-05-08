"""
Token length distribution analysis for the dataset.
Run: python 1b_length_analysis.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast

os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/dataset.csv")
tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

lengths = [len(tok.encode(t, add_special_tokens=True)) for t in df["text"]]

p50 = np.percentile(lengths, 50)
p95 = np.percentile(lengths, 95)
p99 = np.percentile(lengths, 99)
under_128 = 100 * sum(l <= 128 for l in lengths) / len(lengths)

print(f"Total claims: {len(lengths)}")
print(f"Mean tokens: {np.mean(lengths):.1f}")
print(f"Median (50th pct): {p50:.0f}")
print(f"95th percentile: {p95:.0f}")
print(f"99th percentile: {p99:.0f}")
print(f"Max: {max(lengths)}")
print(f"% of claims under 128 tokens: {under_128:.1f}%")

plt.figure(figsize=(8, 4.5))
plt.hist(lengths, bins=40, edgecolor="black", alpha=0.85, color="#5b8db8")
plt.axvline(128, color="red", linestyle="--", linewidth=1.5,
            label="Truncation limit (128)")
plt.axvline(p95, color="orange", linestyle=":", linewidth=1.5,
            label=f"95th percentile ({p95:.0f})")
plt.xlabel("BERT WordPiece tokens per claim")
plt.ylabel("Number of claims")
plt.title(f"Token Length Distribution of Filtered USA-China Claims (n={len(lengths)})")
plt.legend()
plt.tight_layout()
plt.savefig("results/token_length_distribution.png", dpi=150)
print("\nSaved results/token_length_distribution.png")