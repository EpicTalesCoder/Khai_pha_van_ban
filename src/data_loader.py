# -*- coding: utf-8 -*-
"""Nạp & ghép dữ liệu bình luận, map nhãn về spam/ham.

Mỗi bộ dữ liệu là 1 CSV/TSV có cột văn bản + cột nhãn (xem data/README.md
để biết nơi tải). Trả về DataFrame chuẩn 2 cột: `text`, `label`.
"""
import glob
import os

import pandas as pd

# Nhãn gốc -> nhãn đề tài. Bộ dữ liệu nào có nhãn khác thì thêm mapping tại đây.
LABEL_MAPS = {
    # UIT-ViSFD (feedback SV, nhãn eng): positive/negative
    "visfd": {"positive": "ham", "negative": "spam"},
    # ViHSD (nhãn tiếng Việt)
    "vihsd": {"clean": "ham", "offensive": "spam", "hate": "spam"},
}


def load_csv(path: str, text_col: str, label_col: str, mapping: str | None = None) -> pd.DataFrame:
    """Nạp 1 file CSV/TSV. `mapping` là key trong LABEL_MAPS (hoặc None để giữ nguyên)."""
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"{path}: thiếu cột '{text_col}'/'{label_col}'. Cột có sẵn: {list(df.columns)}"
        )
    out = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    out["label"] = out["label"].astype(str).str.strip().str.lower()
    if mapping:
        out["label"] = out["label"].map(LABEL_MAPS[mapping])
        if out["label"].isna().any():
            raise ValueError(f"{path}: có nhãn không nằm trong LABEL_MAPS['{mapping}']")
    return out.dropna().drop_duplicates(subset="text").reset_index(drop=True)


def load_many(patterns: list[str], text_col: str, label_col: str, mapping: str | None = None) -> pd.DataFrame:
    """Ghép nhiều bộ dữ liệu qua glob pattern (đề bài yêu cầu tối thiểu 2 bộ)."""
    paths = [p for pat in patterns for p in glob.glob(pat)]
    if len(paths) == 0:
        raise FileNotFoundError(f"Không tìm thấy file nào khớp: {patterns}")
    frames = [load_csv(p, text_col, label_col, mapping) for p in paths]
    print(f"[data_loader] nạp {len(paths)} file: {[os.path.basename(p) for p in paths]}")
    for f in frames:
        print(f"  - {len(f):>6} dòng | nhãn: {f['label'].value_counts().to_dict()}")
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="text")


def load_sample() -> pd.DataFrame:
    """Dữ liệu mẫu định dạng để chạy thử pipeline."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return load_csv(os.path.join(here, "sample_data", "ptit_comments_sample.csv"), "text", "label")
