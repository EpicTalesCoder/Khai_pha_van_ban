# Dữ liệu — tối thiểu 2 bộ (yêu cầu môn học)

Đặt file tải về vào `data/raw/` rồi chạy:

```bash
python src/train_baseline.py --data "data/raw/<file1>.csv" "data/raw/<file2>.csv" \
    --text-col <cột văn bản> --label-col <cột nhãn> --mapping <visfd|vihsd>
```

## Gợi ý 2 bộ dữ liệu tiếng Việt (bình luận)

1. **UIT-ViSFD** — Vietnamese Student Feedback Dataset (~8.5k bình luận SV về dịch vụ/giảng dạy, nhãn positive/negative).
   Tìm: "UIT-ViSFD dataset" trên GitHub/HuggingFace/Kaggle. Map nhãn: `visfd` (positive→ham, negative→spam).
2. **ViHSD / ViCTSD** — bộ bình luận tiếng Việt gán nhãn CLEAN/OFFENSIVE(/HATE) (~33k câu).
   Tìm: "ViHSD dataset" hoặc "ViCTSD dataset". Map nhãn: `vihsd`.

## Muốn dùng đúng bình luận "SV PTIT"

Bộ public sẵn cho PTIT hầu như không có → tự thu thập để có phần "tự tìm kiếm" của nhóm:
dùng selenium (máy đã có sẵn tool ở repo Automation) cào bình luận từ group/fanpage PTIT,
gán nhãn tay ít nhất vài trăm dòng (spam = quảng cáo/ra đình/lừa đảo; ham = trao đổi bình thường),
lưu CSV 2 cột `text,label` theo mẫu trong `sample_data/`.
