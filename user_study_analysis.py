"""
Analyse user study results: paired Wilcoxon signed-rank test.
Run: python3 user_study_analysis.py
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv("user_study_results.csv")
print(f"Total responses: {len(df)}")
print(f"Unique participants: {df['participant_id'].nunique()}")

agg = df.groupby(["participant_id", "condition"])[
    ["trust", "comprehension", "willing_to_act"]].mean().reset_index()

pivot_trust = agg.pivot(index="participant_id", columns="condition", values="trust").dropna()
pivot_comp  = agg.pivot(index="participant_id", columns="condition", values="comprehension").dropna()
pivot_act   = agg.pivot(index="participant_id", columns="condition", values="willing_to_act").dropna()


def cohens_d_paired(a, b):
    diff = b - a
    return float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0


def report(name, p):
    a, b = p["A"].values, p["B"].values
    if len(a) < 6:
        return {"n": len(a), "note": "insufficient sample (need at least 6 paired)"}
    try:
        stat, pval = wilcoxon(b, a, alternative="greater")
    except ValueError as e:
        return {"n": len(a), "note": f"wilcoxon failed: {e}"}
    return {
        "n": int(len(a)),
        "mean_no_shap_A": round(float(np.mean(a)), 3),
        "mean_with_shap_B": round(float(np.mean(b)), 3),
        "mean_diff_B_minus_A": round(float(np.mean(b - a)), 3),
        "wilcoxon_statistic": round(float(stat), 3),
        "p_value_one_sided": round(float(pval), 4),
        "cohens_d_paired": round(cohens_d_paired(a, b), 3),
        "significant_at_0.05": bool(pval < 0.05),
    }


summary = {
    "trust":         report("trust", pivot_trust),
    "comprehension": report("comprehension", pivot_comp),
    "willing_to_act": report("willing_to_act", pivot_act),
    "n_participants": int(df["participant_id"].nunique()),
    "n_responses": int(len(df)),
}

import os
os.makedirs("results", exist_ok=True)
with open("results/user_study_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== USER STUDY SUMMARY ===")
print(json.dumps(summary, indent=2))
print("\nSaved to results/user_study_summary.json")