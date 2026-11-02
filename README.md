# Khai_pha_van_ban — Phân loại spam/ham bình luận SV PTIT

Đề tài: **Phân loại spam/ham các bình luận của SV PTIT** (chung đề tài với môn Học máy — xem repo `Hoc_may`).

## Cấu trúc

```
├── data/README.md            # hướng dẫn tải 2 bộ dữ liệu (tự tìm kiếm theo yêu cầu môn)
├── sample_data/              # dữ liệu mẫu định dạng để test pipeline ngay
├── src/
│   ├── preprocess.py         # làm sạch + chuẩn hoá teencode + tokenize tiếng Việt
│   ├── data_loader.py        # nạp/ghép CSV, map nhãn về spam/ham
│   ├── train_baseline.py     # TF-IDF (word+char) + NB/LogReg/SVM, CV 5-fold
│   ├── analyze_errors.py     # phân tích lỗi: nhầm lẫn, feature quan trọng nhất
│   └── train_phobert.py      # (điểm cộng) fine-tune PhoBERT trên GPU
├── notebooks/train_colab.ipynb       # mở trên Google Colab, chạy cả pipeline trên GPU
└── colab/vscode_ssh_setup.ipynb      # mở SSH từ Colab để nối với VS Code/code-server
```

## Cách chạy nhanh (chạy thử pipeline với dữ liệu mẫu)

```bash
pip install -r requirements.txt
python src/train_baseline.py --data sample_data/ptit_comments_sample.csv
python src/analyze_errors.py
```

## Chạy thật trên Colab (GPU miễn phí)

1. Mở `notebooks/train_colab.ipynb` trên Colab → Run all.
2. Notebook tự clone repo, cài deps, chờ bạn đặt 2 file CSV vào `data/raw/`, rồi train baseline + PhoBERT.

## VS Code (code-server) ↔ Colab

Code-server (VS Code web) **không hỗ trợ** Remote-SSH. 2 cách nối:

- **Cách chính — sync qua GitHub** (ổn định, không cần mạng ra-vào): sửa code trong code-server → push → Colab `git pull` → chạy GPU → commit/push ngược kết quả. Notebook `train_colab.ipynb` đã sẵn sàng.
- **Cách nâng cao — SSH tunnel**: chạy `colab/vscode_ssh_setup.ipynb` trên Colab để mở SSH (ngrok), rồi từ **terminal của code-server** `ssh` thẳng vào máy Colab để chạy lệnh. Cần token miễn phí tại dashboard.ngrok.com.

Chi tiết từng bước xem trong notebook `colab/vscode_ssh_setup.ipynb`.
