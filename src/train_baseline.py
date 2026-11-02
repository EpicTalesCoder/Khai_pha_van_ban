# -*- coding: utf-8 -*-
"""Baseline: TF-IDF (word 1-2gram + char 2-5gram) -> NB / LogReg / LinearSVC.

Dùng:
    python src/train_baseline.py --data "data/raw/*.csv" --text-col text --label-col label --mapping auto
`--mapping auto` tự nhận diện schema nhãn từng file (ViSFD aspect/chữ, ViHSD số/chữ),
nên trộn nhiều bộ dữ liệu khác format trong cùng 1 thư mục vẫn chạy được.
Không có --data thì chạy với dữ liệu mẫu trong sample_data/.
"""
import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from data_loader import load_many, load_sample
from preprocess import clean_text

MODELS = {
    "naive_bayes": MultinomialNB(),
    "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "svm": LinearSVC(class_weight="balanced"),
}


def make_features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50000,
                                 sublinear_tf=True)),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", help="đường dẫn CSV (cần >=2 bộ theo yêu cầu môn)")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--mapping", default=None, help="key trong LABEL_MAPS (visfd/vihsd)")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    df = load_many(args.data, args.text_col, args.label_col, args.mapping) if args.data else load_sample()
    df["clean"] = df["text"].map(clean_text)

    X, y = df["clean"], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = {}
    for name, clf in MODELS.items():
        pipe = Pipeline([("tfidf", make_features()), ("clf", clf)])
        cv_score = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro").mean()
        scores[name] = cv_score
        print(f"[baseline] {name:12s} f1_macro CV5 = {cv_score:.4f}")

    best = max(scores, key=scores.get)
    print(f"[baseline] chọn model tốt nhất: {best}")

    final = Pipeline([("tfidf", make_features()), ("clf", MODELS[best])])
    # Voting 3 model — đặt cạnh mô hình tốt nhất để so sánh trong báo cáo
    voter = VotingClassifier(
        [(n, Pipeline([("tfidf", make_features()), ("clf", m)])) for n, m in MODELS.items()],
        voting="hard",  # LinearSVC không có predict_proba nên dùng hard voting
    )
    for label, est in [("best", final), ("voting", voter)]:
        est.fit(X_tr, y_tr)
        print(f"\n=== {label} ({best if label == 'best' else 'ensemble'}) trên hold-out 15% ===")
        print(classification_report(y_te, est.predict(X_te), digits=3))

    os.makedirs(args.out, exist_ok=True)
    joblib.dump(final, os.path.join(args.out, "baseline_best.joblib"))
    pd.DataFrame([scores]).T.rename(columns={0: "f1_macro_cv5"}).to_csv(
        os.path.join(args.out, "baseline_scores.csv"))
    print(f"[baseline] đã lưu model vào {args.out}/baseline_best.joblib")


if __name__ == "__main__":
    main()
