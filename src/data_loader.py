# -*- coding: utf-8 -*-
"""Nạp & ghép dữ liệu bình luận, map nhãn về spam/ham.

Mỗi bộ dữ liệu là 1 CSV/TSV có cột văn bản + cột nhãn (xem data/README.md
để biết nơi tải). Trả về DataFrame chuẩn 2 cột: `text`, `label`.

Nhãn gốc lộn xộn giữa các bản copy trên Kaggle/GitHub:
- ViSFD bản phổ biến trên Kaggle (tranlemylinh/uit-visfd): label là chuỗi
  aspect `{BATTERY#Negative};{GENERAL#Positive};` (aspect-based sentiment)
- ViSFD bản khác: positive/negative (chữ)
- ViHSD bản gốc: CLEAN/HATE/OFFENSIVE (chữ)
- ViHSD bản Kaggle (cthng123/hate-speech-detection-vietnamese): 0/1/2 (số)
→ `--mapping auto` tự nhận diện theo GIÁ TRỊ nhãn của từng file, nên trộn
  nhiều bộ dữ liệu khác schema vào cùng 1 thư mục vẫn chạy được.

Quy ước đề tài: positive/clean → ham; negative/hate/offensive → spam.
Nhãn mixed (vừa positive vừa negative trong 1 dòng), neutral hoặc không
nhận diện được → bỏ dòng (có in số dòng bị bỏ).
"""
import glob
import os
import re

import pandas as pd

# Cột văn bản/nhãn hay gặp — load_csv tự "sniff" nếu cột chỉ định không tồn tại
TEXT_CANDIDATES = ("comment", "cmt_col", "text", "sentence")
LABEL_CANDIDATES = ("label", "labels", "sentiment")


def _map_visfd(raw: str):
    """ViSFD: chuỗi aspect `{SCREEN#Negative};...` hoặc chữ positive/negative.

    Trả về 'ham'/'spam'; mixed (vừa positive vừa negative) và neutral → None.
    """
    s = str(raw).strip().lower()
    aspects = set(re.findall(r"#\s*(positive|negative|neutral)", s))
    if aspects:
        if aspects == {"positive"}:
            return "ham"
        if aspects == {"negative"}:
            return "spam"
        return None  # mixed / chỉ neutral → không xếp được 1 nhãn
    return {"positive": "ham", "negative": "spam"}.get(s)


def _map_vihsd(raw: str):
    """ViHSD: chữ clean/hate/offensive hoặc số 0/1/2 (0=clean, 1=hate, 2=offensive)."""
    s = str(raw).strip().lower()
    num = {"0": "ham", "0.0": "ham", "1": "spam", "1.0": "spam",
           "2": "spam", "2.0": "spam"}
    return num.get(s) or {"clean": "ham", "hate": "spam", "offensive": "spam"}.get(s)


def _map_identity(raw: str):
    s = str(raw).strip().lower()
    return s if s in ("spam", "ham") else None


def _auto_mapper(raw_series: pd.Series):
    """Chọn mapper theo giá trị nhãn thực tế trong file (không phụ thuộc tên file)."""
    vals = set(raw_series.dropna().astype(str).str.strip().str.lower().unique()[:200])
    if any("#" in v for v in vals) or vals & {"positive", "negative", "neutral"}:
        return _map_visfd, "visfd"
    if vals & {"0", "0.0", "1", "1.0", "2", "2.0", "clean", "hate", "offensive"}:
        return _map_vihsd, "vihsd"
    if vals & {"spam", "ham"}:
        return _map_identity, "spam/ham"
    raise ValueError(
        "Không nhận diện được bộ dữ liệu từ giá trị nhãn: " + str(sorted(vals)[:10])
    )


# Mapping cũ (key → cách map). 'auto' mới: tự nhận diện theo giá trị nhãn.
LABEL_MAPS = {
    "auto": None,  # sentinel, xử lý riêng trong load_csv
    # UIT-ViSFD (feedback SV, nhãn eng): positive/negative
    "visfd": _map_visfd,
    # ViHSD (nhãn tiếng Việt hoặc số)
    "vihsd": _map_vihsd,
}


def load_csv(path: str, text_col: str = "text", label_col: str = "label",
             mapping: str | None = None) -> pd.DataFrame:
    """Nạp 1 file CSV/TSV. `mapping` = 'auto' | key trong LABEL_MAPS | None (giữ nguyên)."""
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    if text_col not in df.columns:
        text_col = next((c for c in TEXT_CANDIDATES if c in df.columns), None)
    if label_col not in df.columns:
        label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if text_col is None or label_col is None:
        raise ValueError(
            f"{path}: không tìm thấy cột text/label. Cột có sẵn: {list(df.columns)}"
        )

    out = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"].str.len() > 0]
    raw = out["label"].astype(str).str.strip()

    if not mapping or mapping == "none":
        out["label"] = raw.str.lower()
    elif mapping == "auto":
        fn, detected = _auto_mapper(raw)
        out["label"] = raw.map(fn)
        dropped = int(out["label"].isna().sum())
        print(f"[data_loader] {os.path.basename(path)}: nhận diện = {detected}")
        if dropped:
            print(f"[data_loader]   bỏ {dropped}/{len(out)} dòng nhãn không map được (mixed/neutral/không rõ)")
    else:
        fn = LABEL_MAPS.get(mapping)
        if fn is None:
            raise ValueError(f"mapping '{mapping}' không tồn tại. Có sẵn: {list(LABEL_MAPS)}")
        out["label"] = raw.map(fn)
        dropped = int(out["label"].isna().sum())
        if dropped:
            print(f"[data_loader] {os.path.basename(path)}: bỏ {dropped}/{len(out)} dòng nhãn không map được")

    return (out.dropna(subset=["label"])
               .drop_duplicates(subset="text")
               .reset_index(drop=True))


def load_many(patterns: list[str], text_col: str = "text", label_col: str = "label",
              mapping: str | None = None) -> pd.DataFrame:
    """Ghép nhiều bộ dữ liệu qua glob pattern (đề bài yêu cầu tối thiểu 2 bộ).

    Với mapping='auto' mỗi file tự nhận diện nhãn riêng → trộn ViSFD + ViHSD được.
    """
    paths = [p for pat in patterns for p in sorted(glob.glob(pat))]
    if len(paths) == 0:
        raise FileNotFoundError(f"Không tìm thấy file nào khớp: {patterns}")
    frames = [load_csv(p, text_col, label_col, mapping) for p in paths]
    print(f"[data_loader] nạp {len(paths)} file: {[os.path.basename(p) for p in paths]}")
    for f in frames:
        print(f"  - {len(f):>6} dòng | nhãn: {f['label'].value_counts().to_dict()}")
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset="text")
    print(f"[data_loader] tổng cộng sau dedupe: {len(combined)} dòng | "
          f"nhãn: {combined['label'].value_counts().to_dict()}")
    return combined


def load_sample() -> pd.DataFrame:
    """Dữ liệu mẫu định dạng để chạy thử pipeline."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return load_csv(os.path.join(here, "sample_data", "ptit_comments_sample.csv"), "text", "label")
