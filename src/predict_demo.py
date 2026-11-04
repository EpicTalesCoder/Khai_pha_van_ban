# -*- coding: utf-8 -*-
"""Test model cuối cùng với 1 bình luận — in log TỪNG BƯỚC của pipeline.

Dùng:
    python src/predict_demo.py --comment "Ad ơi ib mình xin sdt freeship nè ;)) 🔥"
    python src/predict_demo.py                # không có --comment -> chạy 3 ví dụ mẫu

In ra: bỏ URL -> bỏ emoji -> viết thường -> teencode (ko -> không) -> TF-IDF
-> dự đoán Baseline; nếu có models/phobert thì thêm bước BPE + dự đoán PhoBERT.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib  # noqa: E402

from preprocess import MULTISPACE_RE, NOISE_RE, URL_RE, normalize_teencode  # noqa: E402

LABEL_VI = {
    "toxic": "TOXIC (spam / xúc phạm)",
    "spam": "SPAM",
    "vohai": "VÔ HẠI (bình luận bình thường)",
    "ham": "HAM",
}


def vi(label):
    return LABEL_VI.get(str(label), str(label))


def demo_one(raw, tfidf_pipe=None, pho=None):
    """pho = (tokenizer, model, label_encoder) hoặc None."""
    print("=" * 70)
    print("INPUT :", repr(raw))

    # ── Bước 1: bỏ URL ──
    s1 = URL_RE.sub(" ", raw).strip()
    print("Bước 1 · bỏ URL            :", repr(s1) if s1 else "(không có URL)")

    # ── Bước 2: bỏ emoji / ký tự lạ ──
    s2 = NOISE_RE.sub(" ", s1).strip()
    print("Bước 2 · bỏ emoji/ký tự lạ :", repr(s2) if s2 != s1.strip() else "(không có)")

    # ── Bước 3: viết thường ──
    s3 = s2.lower()
    print("Bước 3 · viết thường      :", repr(s3))

    # ── Bước 4: chuẩn hoá teencode ──
    s4 = normalize_teencode(s3)
    changed = "" if s4 == s3 else "   <-- ĐÃ ĐỔI"
    print("Bước 4 · teencode          :", repr(s4) + changed)

    # ── Bước 5: dọn khoảng trắng ──
    clean = MULTISPACE_RE.sub(" ", s4).strip()
    print("Bước 5 · dọn khoảng trắng  :", repr(clean))

    if not clean:
        print(">>> Chuỗi rỗng sau làm sạch — không dự đoán được.")
        return

    # ── Bước 6: Baseline TF-IDF ──
    if tfidf_pipe is None:
        print("Bước 6 · Baseline          : thiếu models/baseline_best.joblib (chưa train?)")
    else:
        X = tfidf_pipe.named_steps["tfidf"].transform([clean])
        print(f"Bước 6 · TF-IDF vector     : {X.shape[1]} chiều, {X.nnz} feature khác 0")
        pred = tfidf_pipe.named_steps["clf"].predict(X)[0]
        print(f"Bước 7 · Baseline dự đoán  : {vi(pred)}")

    # ── Bước 8: PhoBERT (nếu đã fine-tune) ──
    if pho is None:
        print("Bước 8 · PhoBERT           : thiếu models/phobert/ (bỏ qua)")
    else:
        tok, model, le = pho
        enc = tok(clean, return_tensors="pt", truncation=True, max_length=128)
        bpe = tok.convert_ids_to_tokens(enc["input_ids"][0])
        print(f"Bước 8 · PhoBERT BPE       : {' '.join(bpe)}")
        import torch
        with torch.no_grad():
            logits = model(**enc).logits
        proba = torch.softmax(logits, dim=-1)[0]
        idx = int(proba.argmax())
        label = str(le.inverse_transform([idx])[0])
        probs = ", ".join(f"{vi(c)}={float(p):.2%}" for c, p in zip(le.classes_, proba))
        print(f"Bước 9 · PhoBERT dự đoán   : {vi(label)}   ({probs})")

    print("=" * 70)


def load_models(base="models"):
    tfidf_pipe = None
    p = os.path.join(base, "baseline_best.joblib")
    if os.path.exists(p):
        tfidf_pipe = joblib.load(p)
        print(f"[load] Baseline OK: {p}")
    else:
        print(f"[load] KHÔNG thấy {p} — chạy lại src/train_baseline.py")

    pho = None
    d = os.path.join(base, "phobert")
    if os.path.exists(os.path.join(d, "label_encoder.joblib")):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(d)
        model = AutoModelForSequenceClassification.from_pretrained(d)
        model.eval()
        le = joblib.load(os.path.join(d, "label_encoder.joblib"))
        pho = (tok, model, le)
        print(f"[load] PhoBERT OK: {d}/")
    else:
        print(f"[load] KHÔNG thấy {d}/ — bỏ qua PhoBERT")
    return tfidf_pipe, pho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comment", help="bình luận cần test (bỏ qua -> chạy ví dụ mẫu)")
    ap.add_argument("--models", default="models")
    args = ap.parse_args()

    tfidf_pipe, pho = load_models(args.models)

    if args.comment:
        demo_one(args.comment, tfidf_pipe, pho)
    else:
        print("\n[mode] Không có --comment -> chạy 3 ví dụ mẫu:\n")
        for c in [
            "Ad ơi INBOX mình xin số SDT freEship nè http://spam.vn ;)) 🔥🔥",
            "ko hiểu j hết, thầy dạy tệ quá",
            "Thầy dạy dễ hiểu quá, em cảm ơn thầy nhé!",
        ]:
            demo_one(c, tfidf_pipe, pho)


if __name__ == "__main__":
    main()
