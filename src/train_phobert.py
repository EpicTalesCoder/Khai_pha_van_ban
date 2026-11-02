# -*- coding: utf-8 -*-
"""(Điểm cộng) Fine-tune PhoBERT cho spam/ham — chạy tốt nhất trên GPU Colab.

Dùng:
    python src/train_phobert.py --data data/raw/a.csv data/raw/b.csv \
        --text-col comment --label-col label --mapping visfd
"""
import argparse
import os

import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from data_loader import load_many, load_sample
from preprocess import clean_text

MODEL_NAME = "vinai/phobert-base"


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.enc = tokenizer(list(texts), truncation=True, padding="max_length",
                             max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]))
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--mapping", default=None)
    ap.add_argument("--out", default="models/phobert")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    df = load_many(args.data, args.text_col, args.label_col, args.mapping) if args.data else load_sample()
    df["clean"] = df["text"].map(clean_text)

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.15,
                                      stratify=y, random_state=42)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(le.classes_))

    tr_ds = TextDataset(df["clean"].iloc[idx_tr], y[idx_tr], tok)
    te_ds = TextDataset(df["clean"].iloc[idx_te], y[idx_te], tok)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to=[],
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        from sklearn.metrics import f1_score
        return {"f1": f1_score(labels, preds, average="macro")}

    trainer = Trainer(model=model, args=targs, train_dataset=tr_ds,
                      eval_dataset=te_ds, compute_metrics=compute_metrics)
    trainer.train()

    preds = np.argmax(trainer.predict(te_ds).predictions, axis=-1)
    print(classification_report(y[idx_te], preds, target_names=le.classes_, digits=3))

    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    joblib.dump(le, os.path.join(args.out, "label_encoder.joblib"))
    print(f"[phobert] đã lưu model vào {args.out}/")


if __name__ == "__main__":
    main()
