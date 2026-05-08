"""
STEP 4 (additional): 5-FOLD CROSS-VALIDATION ON BERT
Run: python3 4_kfold_evaluation.py
"""
import os, json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (f1_score, accuracy_score,
                             precision_score, recall_score)
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          TrainingArguments, Trainer)
from torch.utils.data import Dataset

MODEL_NAME = "bert-base-uncased"
df = pd.read_csv("data/dataset.csv")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)


class ClaimsDataset(Dataset):
    def __init__(self, df_):
        enc = tokenizer(list(df_["text"]), padding="max_length",
                        truncation=True, max_length=128, return_tensors="pt")
        self.encodings = enc
        self.labels = torch.tensor(df_["label_binary"].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.encodings["input_ids"][i],
                "attention_mask": self.encodings["attention_mask"][i],
                "labels": self.labels[i]}


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    return {"f1_macro": f1_score(p.label_ids, preds, average="macro")}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_val_idx, test_idx) in enumerate(
        skf.split(df, df["label_binary"]), 1):
    print(f"\n========== FOLD {fold}/5 ==========")
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.15, random_state=42,
        stratify=train_val_df["label_binary"])

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        id2label={0: "False", 1: "True"},
        label2id={"False": 0, "True": 1})

    args = TrainingArguments(
        output_dir=f"model/fold_{fold}",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=200,
        report_to="none",
        fp16=torch.cuda.is_available(),
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=ClaimsDataset(train_df),
        eval_dataset=ClaimsDataset(val_df),
        compute_metrics=compute_metrics)

    trainer.train()

    preds_out = trainer.predict(ClaimsDataset(test_df))
    preds = np.argmax(preds_out.predictions, axis=-1)
    y = test_df["label_binary"].values

    fold_result = {
        "fold": fold,
        "accuracy":  round(accuracy_score(y, preds), 4),
        "precision": round(precision_score(y, preds, average="macro"), 4),
        "recall":    round(recall_score(y, preds, average="macro"), 4),
        "f1_macro":  round(f1_score(y, preds, average="macro"), 4),
        "test_size": len(test_df),
    }
    fold_results.append(fold_result)
    print(fold_result)

f1s = [r["f1_macro"] for r in fold_results]
summary = {
    "fold_results": fold_results,
    "mean_f1_macro": round(float(np.mean(f1s)), 4),
    "std_f1_macro":  round(float(np.std(f1s, ddof=1)), 4),
    "n_folds": 5,
}
os.makedirs("results", exist_ok=True)
with open("results/kfold_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n==== 5-FOLD SUMMARY ====")
print(f"Mean macro-F1: {summary['mean_f1_macro']:.4f}")
print(f"Std  macro-F1: {summary['std_f1_macro']:.4f}")
print(f"Per-fold F1: {[r['f1_macro'] for r in fold_results]}")