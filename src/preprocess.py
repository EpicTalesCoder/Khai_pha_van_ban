# -*- coding: utf-8 -*-
"""Tiền xử lý bình luận tiếng Việt cho bài toán spam/ham."""
import re

# Teencode/từ viết tắt phổ biến trong bình luận của sinh viên
TEENCODE = {
    "ko": "không", "k": "không", "kh": "không", "kg": "không",
    "dc": "được", "dk": "được", "dık": "được",
    "bn": "bao nhiêu", "tn": "thế nào", "ntn": "như thế nào",
    "e": "em", "m": "mày", "bnh": "bình", "vs": "với", "z": "vậy",
    "j": "gì", "jk": "joke", "ad": "admin", "mn": "mọi người",
    "ib": "inbox", "ibm": "inbox mình", "sdt": "số điện thoại",
    "ptit": "học viện công nghệ bưu chính viễn thông",
    "sp": "sản phẩm", "km": "khuyến mãi", "freeship": "miễn phí giao hàng",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# Giữ chữ/số/khoảng trắng và dấu câu cơ bản; loại emoji & ký tự lạ
NOISE_RE = re.compile(r"[^0-9A-Za-zÀ-ỹà-ỹ\s,\.!\?]")
MULTISPACE_RE = re.compile(r"\s+")

try:
    from underthesea import word_tokenize
except ImportError:  # chưa cài thì fallback tách theo khoảng trắng
    word_tokenize = None


def normalize_teencode(text: str) -> str:
    tokens = text.split()
    return " ".join(TEENCODE.get(t.lower(), t) for t in tokens)


def clean_text(text: str, tokenize: bool = False) -> str:
    """Làm sạch 1 bình luận: bỏ URL/emoji, teencode, viết thường, (tuỳ chọn) tách từ."""
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = NOISE_RE.sub(" ", text)
    text = text.lower().strip()
    text = normalize_teencode(text)
    text = MULTISPACE_RE.sub(" ", text)
    if tokenize and word_tokenize is not None:
        text = word_tokenize(text, format="text")
    return text


if __name__ == "__main__":
    demos = [
        "Ad ơi INBOX mình xin số SDT freEship nè http://spam.vn  ;)) 🔥🔥",
        "Thầy dạy dễ hiểu quá, em cảm ơn thầy nhé!",
    ]
    for d in demos:
        print(clean_text(d), "| tokenized:", clean_text(d, tokenize=True))
