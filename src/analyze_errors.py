# -*- coding: utf-8 -*-
"""Phân tích lỗi model baseline: nhầm lẫn, feature quan trọng, confusion matrix.

Dùng (sau khi đã chạy train_baseline.py):
    python src/analyze_errors.py --data data/raw/*.csv --text-col comment --label-col label
"""
import argparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from data_loader import load_many, load_sample
from preprocess import clean_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--mapping", default=None)
    ap.add_argument("--model", default="models/baseline_best.joblib")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    df = load_many(args.data, args.text_col, args.label_col, args.mapping) if args.data else load_sample()
    df["clean"] = df["text"].map(clean_text)
    X_tr, X_te, y_tr, y_te = train_test_split(df["clean"], df["label"], test_size=0.15,
                                              stratify=df["label"], random_state=42)
    model = joblib.load(args.model)
    y_pred = model.predict(X_te)
    print(classification_report(y_te, y_pred, digits=3))

    # Confusion matrix
    labels = sorted(df["label"].unique())
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Dự đoán"); plt.ylabel("Thực tế")
    plt.tight_layout()
    plt.savefig(f"{args.out}/confusion_matrix.png", dpi=150)
    print(f"[errors] confusion matrix -> {args.out}/confusion_matrix.png")

    # Các bình luận bị nhầm (kèm độ tin cậy nếu model hỗ trợ)
    wrong = pd.DataFrame({"text": X_te, "thuc_te": y_te, "du_doan": y_pred}).reset_index(drop=True)
    wrong = wrong[wrong["thuc_te"] != wrong["du_doan"]]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_te).max(axis=1)
        # proba tính trên X_te nguyên thứ tự; sau reset_index, index == vị trí trong X_te
        wrong = wrong.assign(confidence=proba[wrong.index])
    print(f"\n[errors] {len(wrong)} bình luận bị nhầm (20 dòng đầu):")
    print(wrong.head(20).to_string(max_colwidth=80))

    # Feature mạnh nhất theo từng lớp (nếu pipeline có LogisticRegression)
    try:
        clf = model.named_steps["clf"]
        names = model.named_steps["tfidf"].get_feature_names_out()
        if hasattr(clf, "coef_"):
            coefs = clf.coef_ if clf.coef_.ndim == 2 and clf.coef_.shape[0] > 1 else [clf.coef_[0]]
            classes = clf.classes_ if len(coefs) > 1 else [f"'{clf.classes_[1]}' (đối lập '{clf.classes_[0]}')"]
            for i, cls in enumerate(classes):
                top = names[coefs[i].argsort()[-10:][::-1]]
                print(f"\n[top-10 '{cls}'] {', '.join(top)}")
    except (AttributeError, KeyError):
        print("[errors] model hiện tại không xem được coef, bỏ qua top-features")


if __name__ == "__main__":
    main()
