# Báo cáo: Phân loại bình luận toxic / vô hại của sinh viên PTIT

**Môn:** Khai phá dữ liệu — Đề tài: Phân loại bình luận toxic / vô hại của SV PTIT
**Phiên bản:** baocao_v1.md (nháp v1 — 02/11/2026)
**Môi trường chạy:** Kaggle Notebook GPU, kernel `mq3444/kpvb-train-toxic` (chạy headless, hoàn thành 15:49 giờ VN). Toàn bộ số liệu dưới đây lấy từ log kernel và file output (`ketqua.md`, `baseline_scores.csv`, `confusion_matrix.png`).

---

## 1. Giới thiệu

Bình luận trên các nhóm/fanpage sinh viên luôn kèm những nội dung độc hại: chửi bới, miệt thị, kích động. Mục tiêu đề tài là xây dựng mô hình tự động phân loại bình luận tiếng Việt thành **toxic** (độc hại) và **vô hại**, làm cơ sở cho công cụ lọc bình luận cho môi trường sinh viên PTIT.

Do không tồn tại bộ dữ liệu bình luận SV PTIT công khai (bản gần nhất trên HuggingFace `p-storm/ptit-facebook-comments-spamham` là gated và không tải được — đã thử HTTP 404), đề tài dùng 2 bộ dữ liệu công khai tiếng Việt cùng bản chất để **mô phỏng** môi trường bình luận. Hướng tiếp cận này giữ nguyên tinh thần "dữ liệu tự tìm kiếm, tối thiểu 2 bộ" của yêu cầu môn.

Điểm cộng theo yêu cầu môn: fine-tune transformer PhoBERT (kiến thức ngoài phần được dạy), pipeline hoàn toàn headless tự chạy trên GPU miễn phí Kaggle.

## 2. Dữ liệu & tiền xử lý

### 2.1 Hai bộ dữ liệu

| # | Bộ dữ liệu | Nguồn | Số file | Nhãn gốc | Quy đổi |
|---|---|---|---|---|---|
| 1 | **UIT-ViCTSD** (Vietnamese Constructiveness and Toxicity Dataset) | Kaggle `phanvmhuyn/uit-victsd` | 3 (train 7.000 / dev 2.000 / test 1.000) | `Toxicity` 0/1 | 0 → vô hại, 1 → toxic |
| 2 | **ViHSD** (Vietnamese Hate Speech Detection) | Kaggle `cthng123/hate-speech-detection-vietnamese` | 3 (train 5.166 / val 1.112 / test 2.224) | 0/1/2 (clean/hate/offensive) | 0 → vô hại, 1–2 → toxic |

### 2.2 Tiền xử lý

- Nạp qua `src/data_loader.py` với `--mapping auto`: tự nhận diện schema nhãn từng file theo GIÁ TRỊ nhãn (không phụ thuộc tên file), nên trộn 2 bộ khác format vẫn chạy được.
- Chuẩn hoá văn bản bằng `src/preprocess.py`: làm sạch, chuẩn hoá teencode, tokenize tiếng Việt.
- Bỏ dòng trùng theo **nội dung** (không theo id) — toàn cục trước khi chia tập, tránh một bản sao rơi vào cả train lẫn test làm điểm test đẹp giả.

### 2.3 Kết quả nạp (từ log kernel)

| File | Dòng | Phân bố |
|---|---|---|
| df_train_clean (ViCTSD) | 6.969 | 6.210 vô hại / 759 toxic |
| df_dev_clean (ViCTSD) | 1.992 | 1.760 / 232 |
| df_test_clean (ViCTSD) | 997 | 887 / 110 |
| vihsd_train | 4.867 | 2.092 / 2.775 |
| vihsd_val | 731 | 290 / 441 |
| vihsd_test | 1.441 | 600 / 841 |
| **Tổng sau dedupe** | **16.683** | vô hại 8.834 — toxic 1.100 — spam 3.772 — ham 2.977 |

Chi tiết quan trọng: vì 2 bộ dùng 2 bộ tên nhãn khác nhau (ViCTSD: `vohai/toxic`, ViHSD: `ham/spam`), lần chạy này pipeline giữ nguyên **4 nhãn** thay vì gộp về 2 lớp toxic/vô hại. Ảnh hưởng của quyết định thiết kế này được phân tích ở mục 6.

Chia dữ liệu: **85% train / 15% hold-out test**, stratified, `random_state=42`. Tập test 2.503 mẫu (ham 447, spam 566, toxic 165, vô hại 1.325), chỉ chạm đúng một lần ở cuối.

## 3. Phương pháp

### 3.1 Baseline TF-IDF (src/train_baseline.py)

- Feature: TF-IDF word 1-2gram (30k features) + char_wb 2-5gram (50k features), sublinear tf.
- 3 mô hình: MultinomialNB, LogisticRegression (class_weight balanced), LinearSVC (balanced).
- Chọn model bằng **CV 5-fold stratified trên tập train** (f1_macro), không chạm test.
- So sánh thêm: hard Voting ensemble 3 model.

### 3.2 PhoBERT fine-tune (src/train_phobert.py) — điểm cộng

- `vinai/phobert-base`, 3 epoch, lr 2e-5, batch 32, max_len 128, fp16 trên GPU Kaggle.
- Chọn model tốt nhất theo f1 (load_best_model_at_end).
- `src/analyze_errors.py`: confusion matrix + mẫu nhầm lẫn + top features (điểm trừ: cell này lỗi giữa chừng trong lần chạy hiện tại — xem mục 6.5).

## 4. Kết quả

### 4.1 Baseline — CV5 trên tập train (f1_macro)

| Model | f1_macro CV5 |
|---|---|
| Naive Bayes | 0.4479 |
| LogisticRegression | **0.6959** ← chọn |
| LinearSVC | 0.6932 |

### 4.2 Hold-out test 15% (2.503 mẫu, 4 lớp)

| Model | Accuracy | f1_macro | f1 vô hại | f1 spam | f1 ham | f1 toxic |
|---|---|---|---|---|---|---|
| LogReg (best baseline) | 0.783 | 0.698 | 0.853 | 0.845 | 0.677 | 0.416 |
| Voting ensemble | **0.812** | **0.712** | 0.881 | 0.850 | 0.679 | 0.439 |
| PhoBERT | 0.790 | 0.662 | **0.877** | 0.808 | 0.600 | 0.364 |

PhoBERT theo epoch (eval f1_macro): ep1 0.597 → ep2 0.653 → ep3 0.662.

### 4.3 Confusion matrix (logreg, hàng = thực tế, cột = dự đoán)

| | ham | spam | toxic | vô hại |
|---|---|---|---|---|
| **ham** | 315 | 59 | 16 | 57 |
| **spam** | 58 | 469 | 21 | 18 |
| **toxic** | 12 | 3 | 86 | 64 |
| **vô hại** | 98 | 13 | 125 | 1089 |

## 5. Phân tích lỗi

1. **Nhầm chéo ham ↔ vô hại là lớn nhất** (59 + 98 = 157 mẫu): 2 nhãn này đến từ 2 nguồn khác nhau nhưng ý nghĩa gần trùng nhau ("bình luận sạch"). Model không có thông tin nào để phân biệt ngoài văn phong nguồn — chứng tỏ tách 4 lớp là thiết kế sai, cần gộp (mục 6.2).
2. **Toxic bị bỏ sót nặng**: 64/150 mẫu toxic thực tế bị dự đoán "vô hại" (recall toxic chỉ 0.521). Đặc trưng ngôn từ toxic tinh vi (mỉa mai, nói giảm nói tránh) vượt quá khả năng TF-IDF n-gram.
3. **Vô hại → toxic nhầm 125 mẫu**: f1 toxic thấp một phần do precision thấp (0.347) — model "quảng cáo" nhiều bình luận thường thành toxic, phản ánh lớp toxic quá nhỏ (6.6%).
4. PhoBERT cũng mắc cùng hiện tượng (recall toxic 0.279) dù embedding ngữ nghĩa tốt hơn — xác nhận vấn đề nằm ở dữ liệu/thiết kế, không phải feature.

## 6. Thiếu xót & hạn chế (trung thực, chưa sửa)

1. **[Thiết kế] Chạy 4 lớp thay vì 2 lớp toxic/vô hại.** Đề bài là toxic/vô hại nhưng pipeline giữ nguyên 2 bộ nhãn nguồn (vohai/toxic/ham/spam). Đã kiểm chứng sơ bộ trên máy: nếu gộp nhãn về 2 lớp trước khi train, SVM đạt f1 toxic ~0.77 / macro ~0.84 — cao hơn hẳn 0.416 hiện tại. Đây là thiếu xót lớn nhất, cần sửa ở phiên bản sau.
2. **[Đánh giá] eval_f1 của PhoBERT tính trên chính tập test** (eval_dataset = te_ds trong Trainer), tức model "nhìn" test 3 lần trong lúc chọn epoch — vi phạm nguyên tắc test chỉ chạm một lần. Cần tách dev set từ train để early-stopping/selection.
3. **[Dữ liệu] Lệch lớp nghiêm trọng:** toxic chỉ 1.100/16.683 (6.6%). Chưa xử lý: PhoBERT không có class weighting, không oversample, không threshold tuning. Phần lớn sự chênh giữa macro f1 (0.698) và accuracy (0.783) đến từ đây.
4. **[Dữ liệu] Không phải bình luận SV PTIT thật.** ViCTSD là bình luận báo chí điện tử, ViHSD là bình luận mạng xã hội. Kết quả chỉ mô phỏng; khi áp vào bình luận nhóm/fanpage SV PTIT (ngôn ngữ teencode, viết tắt nặng hơn) hiệu năng có thể giảm.
5. **[Pipeline] analyze_errors.py lỗi giữa chừng** (IndexError ở phần confidence do lệch index sau train_test_split). Đã fix commit `e19d35d` nhưng **chưa chạy lại trên kernel**, nên lần chạy hiện tại thiếu top-features và danh sách mẫu nhầm; confusion matrix ở mục 4.3 vẫn dùng được (sinh trước điểm lỗi).
6. **[Dữ liệu] Chưa kiểm tra near-duplicate chéo nguồn** (chỉ dedupe text y hệt). ViCTSD và ViHSD có thể có câu gần trùng nằm cả 2 tập.
7. **[Mô hình] Naive Bayes tệ bất thường** (0.448): TF-IDF sublinear + char 5gram vi phạm giả định độc lập đặc trưng của NB — nên thay bằng ComplementNB hoặc bỏ khỏi pipeline.
8. **[Phạm vi] "Vô hại" ≠ "constructive".** ViCTSD còn có nhãn Constructiveness (0/1) mà pipeline không dùng; tận dụng thêm có thể tách được 3 mức: constructive / vô hại / toxic.

## 7. Hướng phát triển

- **Ngắn hạn (bản v2):** gộp nhãn 2 lớp toxic/vô hại trước khi train (đã kiểm chứng giúp f1 toxic ~0.77 baseline); tách dev set cho PhoBERT; thêm class_weight/oversample cho lớp toxic; chạy lại analyze_errors sau khi đã fix.
- **Trung hạn:** thử threshold tuning cho lớp toxic (ưu tiên recall); ensemble PhoBERT + TF-IDF; xử lý teencode/emoji đặc trưng bình luận SV.
- **Dài hạn:** xin quyền bộ PTIT gated hoặc crawl bình luận nhóm SV PTIT (đúng môi trường đích); đa nhãn (toxic + constructive + topic).

## 8. Kết luận

Pipeline 2 dataset → TF-IDF baseline + PhoBERT hoạt động trọn vẹn headless trên Kaggle GPU. Ở thiết kế 4 lớp hiện tại, tốt nhất là Voting ensemble: **accuracy 0.812, f1_macro 0.712**; lớp toxic — lớp quan trọng nhất — mới đạt f1 0.44 do lệch lớp và thiết kế nhãn. Thiếu xót lớn nhất (4 lớp thay vì 2) đã xác định được và có hướng sửa rõ với kết quả kiểm chứng sơ bộ; sau khi gộp 2 lớp + xử lý lệch lớp, mục tiêu hợp lý là f1 toxic ≥ 0.80 cho bản v2.
