# Dữ liệu — tối thiểu 2 bộ (yêu cầu môn học)

Đặt file tải về vào `data/raw/` rồi chạy:

```bash
python src/train_baseline.py --data "data/raw/*.csv" --mapping auto
```

(`--mapping auto` tự nhận diện nhãn của từng file; chỉ cần `--text-col/--label-col`
khi cột của file không nằm trong các tên phổ biến — xem phần dưới.)

## Gợi ý 2 bộ dữ liệu tiếng Việt (bình luận) — ĐÃ XÁC MINH tải được từ Kaggle 11/2026

1. **UIT-ViSFD** — feedback SV (bản Kaggle: `kaggle datasets download tranlemylinh/uit-visfd --unzip`,
   7786 train / 1112 dev / 2224 test; cột `comment`,`label` dạng aspect `{BATTERY#Negative};...`).
   Dùng `--mapping auto`: positive→ham, negative→spam, dòng mixed/neutral bị bỏ.
2. **ViHSD** — hate speech VN (`kaggle datasets download cthng123/hate-speech-detection-vietnamese --unzip`,
   5166 train; cột `cmt_col`,`labels` dạng số 0/1/2 = clean/hate/offensive).
   Dùng `--mapping auto`: 0/clean→ham, 1-2/hate/offensive→spam.

Không cần chỉ `--text-col/--label-col`: loader tự dò cột (comment/cmt_col/text/sentence
và label/labels/sentiment). Trộn 2 bộ: gộp file vào `data/raw/` rồi chạy
`--data "data/raw/*.csv" --mapping auto` — mỗi file tự nhận diện schema riêng.

## Muốn dùng đúng bình luận "SV PTIT"

Bộ public sẵn cho PTIT hầu như không có → tự thu thập để có phần "tự tìm kiếm" của nhóm:
dùng selenium (máy đã có sẵn tool ở repo Automation) cào bình luận từ group/fanpage PTIT,
gán nhãn tay ít nhất vài trăm dòng (spam = quảng cáo/ra đình/lừa đảo; ham = trao đổi bình thường),
lưu CSV 2 cột `text,label` theo mẫu trong `sample_data/`.
