"""
STEP 6: Back-translation augmentation + BERT retraining.
Run: python3 6_augment_and_retrain.py
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, accuracy_score,
                             precision_score, recall_score)
from transformers import (MarianMTModel, MarianTokenizer,
                          BertTokenizerFast,
                          BertForSequenceClassification,
                          TrainingArguments, Trainer)
from torch.utils.data import Dataset

device = ("cuda" if torch.cuda.is_available()
          else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")


def make_translator(model_name):
    tok = MarianTokenizer.from_pretrained(model_name)
    mdl = MarianMTModel.from_pretrained(model_name).to(device).eval()

    def translate(texts, batch_size=8):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=128).to(device)
            with torch.no_grad():
                ids = mdl.generate(**enc, max_length=128, num_beams=4)
            out.extend(tok.batch_decode(ids, skip_special_tokens=True))
        return out
    return translate


print("Loading translation models (this downloads ~1.5GB on first run)...")
en_to_fr = make_translator("Helsinki-NLP/opus-mt-en-fr")
fr_to_en = make_translator("Helsinki-NLP/opus-mt-fr-en")
en_to_de = make_translator("Helsinki-NLP/opus-mt-en-de")
de_to_en = make_translator("Helsinki-NLP/opus-mt-de-en")

df = pd.read_csv("data/dataset.csv")
train_df, temp = train_test_split(df, test_size=0.30, random_state=42,
                                  stratify=df["label_binary"])
val_df, test_df = train_test_split(temp, test_size=0.50, random_state=42,
                                   stratify=temp["label_binary"])

print(f"\nOriginal training set: {len(train_df)} claims")
print("Back-translating via French...")
fr_round = fr_to_en(en_to_fr(train_df["text"].tolist()))
print("Back-translating via German...")
de_round = de_to_en(en_to_de(train_df["text"].tolist()))

aug_df = pd.concat([
    train_df,
    pd.DataFrame({
        "text": fr_round,
        "label_binary": train_df["label_binary"].values,
        "source": "aug_fr"
    }),
    pd.DataFrame({
        "text": de_round,
        "label_binary": train_df["label_binary"].values,
        "source": "aug_de"
    }),
], ignore_index=True)

aug_df = aug_df.drop_duplicates(subset="text")
aug_df = aug_df[aug_df["text"].str.len() > 20]
aug_df = aug_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Augmented training set: {len(aug_df)} claims "
      f"(+{len(aug_df) - len(train_df)} synthetic)")

os.makedirs("data", exist_ok=True)
aug_df.to_csv("data/train_augmented.csv", index=False)

print("\n=== SAMPLE AUGMENTATIONS (inspect for quality) ===")
sample = aug_df[aug_df["source"].isin(["aug_fr", "aug_de"])].sample(5, random_state=1)
for _, r in sample.iterrows():
    print(f"  [{r['source']}] {r['text'][:120]}")

MODEL_NAME = "bert-base-uncased"
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


print("\nRetraining BERT on augmented data...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2,
    id2label={0: "False", 1: "True"},
    label2id={"False": 0, "True": 1})

args = TrainingArguments(
    output_dir="model/bert_factchecker_augmented",
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
    logging_steps=100,
    report_to="none",
    fp16=torch.cuda.is_available(),
    save_total_limit=1)

trainer = Trainer(
    model=model, args=args,
    train_dataset=ClaimsDataset(aug_df),
    eval_dataset=ClaimsDataset(val_df),
    compute_metrics=compute_metrics)

trainer.train()

preds = np.argmax(trainer.predict(ClaimsDataset(test_df)).predictions, axis=-1)
y = test_df["label_binary"].values

results = {
    "accuracy":  round(accuracy_score(y, preds), 4),
    "precision": round(precision_score(y, preds, average="macro"), 4),
    "recall":    round(recall_score(y, preds, average="macro"), 4),
    "f1_macro":  round(f1_score(y, preds, average="macro"), 4),
    "train_size_before": int(len(train_df)),
    "train_size_after":  int(len(aug_df)),
    "test_size": int(len(test_df)),
}

os.makedirs("results", exist_ok=True)
with open("results/augmented_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== AUGMENTED BERT TEST RESULTS ===")
for k, v in results.items():
    print(f"  {k}: {v}")
print("\nSaved to results/augmented_metrics.json")