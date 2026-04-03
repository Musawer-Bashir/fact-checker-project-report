"""
STEP 2: TRAIN THE BERT MODEL

Fine-tunes bert-base-uncased on your geopolitical claims dataset.
Saves the model + tokeniser to model/bert_factchecker/
Also saves evaluation results to results/metrics.json

Run: python 2_train_model.py
Requires: pip install transformers torch datasets scikit-learn pandas
Time: ~15-30 mins on GPU, ~2-3 hours on CPU, 10-12 minutes using macbook air m4
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          TrainingArguments, Trainer)
import torch
from torch.utils.data import Dataset

os.makedirs("model/bert_factchecker", exist_ok=True)
os.makedirs("results", exist_ok=True)

#  Load data 
print("Loading dataset...")
df = pd.read_csv("data/dataset.csv")
print(f"Total claims: {len(df)}")
print(df["label_binary"].value_counts())

#  Split: 70% train / 15% val / 15% test
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42,
                                     stratify=df["label_binary"])
val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=42,
                                     stratify=temp_df["label_binary"])

# Save test set for later evaluation
test_df.to_csv("data/test_set.csv", index=False)
print(f"\nSplit: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

#  Tokeniser 
MODEL_NAME = "bert-base-uncased"
tokenizer  = BertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenise(texts, max_length=128):
    return tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

#  PyTorch Dataset 
class ClaimsDataset(Dataset):
    def __init__(self, df):
        self.encodings = tokenise(df["text"])
        self.labels    = torch.tensor(df["label_binary"].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

train_dataset = ClaimsDataset(train_df)
val_dataset   = ClaimsDataset(val_df)
test_dataset  = ClaimsDataset(test_df)

#  Model 
print(f"\nLoading {MODEL_NAME}...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "False", 1: "True"},
    label2id={"False": 0, "True": 1}
)

#  Metrics function for Trainer 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1_macro":  f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall":    recall_score(labels, preds, average="macro"),
    }

#  Training Arguments 
# These follow Devlin et al. (2019) recommendations for fine-tuning BERT
training_args = TrainingArguments(
    output_dir="model/bert_factchecker",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",  # disable wandb
    fp16=torch.cuda.is_available(),
)

#  Train 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\nStarting training...")
trainer.train()

#  Evaluate on test set 
print("\nEvaluating on test set...")
test_results = trainer.evaluate(test_dataset)

# Detailed metrics with confusion matrix
test_preds_output = trainer.predict(test_dataset)
preds = np.argmax(test_preds_output.predictions, axis=-1)
true_labels = test_df["label_binary"].values

acc  = accuracy_score(true_labels, preds)
prec = precision_score(true_labels, preds, average="macro")
rec  = recall_score(true_labels, preds, average="macro")
f1   = f1_score(true_labels, preds, average="macro")
cm   = confusion_matrix(true_labels, preds).tolist()

metrics = {
    "accuracy":  round(acc, 4),
    "precision": round(prec, 4),
    "recall":    round(rec, 4),
    "f1_macro":  round(f1, 4),
    "confusion_matrix": cm,
    "test_size": len(test_df),
    "train_size": len(train_df),
    "val_size": len(val_df),
}

print("\n=== TEST SET RESULTS ===")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1 Macro:  {f1:.4f}")
print(f"  Confusion Matrix:\n    {cm[0]}\n    {cm[1]}")
print("  (rows=actual, cols=predicted; 0=False, 1=True)")

# Save metrics for report
with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\nMetrics saved to results/metrics.json")

#  Save model and tokeniser 
model.save_pretrained("model/bert_factchecker")
tokenizer.save_pretrained("model/bert_factchecker")
print("Model saved to model/bert_factchecker/")

#  Baseline comparison 
print("\n=== BASELINE COMPARISON ===")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

X_train = train_df["text"].tolist()
y_train = train_df["label_binary"].tolist()
X_test  = test_df["text"].tolist()
y_test  = test_df["label_binary"].tolist()

# Majority class baseline
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_preds = dummy.predict(X_test)
dummy_f1 = f1_score(y_test, dummy_preds, average="macro")
print(f"  Majority Class Baseline F1:  {dummy_f1:.4f}")

# TF-IDF + Logistic Regression
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
lr    = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(tfidf.fit_transform(X_train), y_train)
lr_preds = lr.predict(tfidf.transform(X_test))
lr_f1 = f1_score(y_test, lr_preds, average="macro")
print(f"  TF-IDF + LR Baseline F1:     {lr_f1:.4f}")
print(f"  Fine-tuned BERT F1:          {f1:.4f}")
print(f"  Improvement over TF-IDF:     +{(f1 - lr_f1):.4f}")

baselines = {
    "majority_class_f1": round(dummy_f1, 4),
    "tfidf_lr_f1": round(lr_f1, 4),
    "bert_f1": round(f1, 4),
    "improvement_over_tfidf": round(f1 - lr_f1, 4)
}
with open("results/baselines.json", "w") as f:
    json.dump(baselines, f, indent=2)
print("\nBaseline comparison saved to results/baselines.json")
