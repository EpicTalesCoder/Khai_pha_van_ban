# Báo cáo: Phân loại bình luận toxic / vô hại của sinh viên PTIT

**Môn:** Khai phá dữ liệu — Đề tài: Phân loại bình luận toxic / vô hại của SV PTIT
**Phiên bản:** baocao_v1.md (sửa lần 2 — 04/11/2026, bổ sung số PhoBERT chính thức từ kernel v2)
**Môi trường chạy:** Kaggle Notebook GPU, kernel `mq3444/kpvb-train-toxic` (chạy headless). Toàn bộ số liệu dưới đây lấy từ log kernel và file output (`ketqua.md`, `baseline_scores.csv`, `confusion_matrix.png`).

---

## 1. Giới thiệu

Bình luận trên các nhóm/fanpage sinh viên luôn kèm những nội dung độc hại: chửi bới, miệt thị, kích động. Mục tiêu đề tài là xây dựng mô hình tự động phân loại bình luận tiếng Việt thành **toxic** (độc hại) và **vô hại** — bài toán phân loại nhị phân — làm cơ sở cho công cụ lọc bình luận cho môi trường sinh viên PTIT.

Do không tồn tại bộ dữ liệu bình luận SV PTIT công khai (bản gần nhất trên HuggingFace `p-storm/ptit-facebook-comments-spamham` là gated và không tải được — đã thử, HTTP 404), đề tài dùng 2 bộ dữ liệu công khai tiếng Việt cùng bản chất để **mô phỏng** môi trường bình luận, đáp ứng yêu cầu "dữ liệu tự tìm kiếm, tối thiểu 2 bộ" của môn.

Điểm cộng theo yêu cầu môn: fine-tune transformer PhoBERT (kiến thức ngoài phần được dạy), pipeline hoàn toàn headless tự chạy trên GPU miễn phí Kaggle.

## 2. Dữ liệu & tiền xử lý

### 2.1 Hai bộ dữ liệu

| # | Bộ dữ liệu | Nguồn | Số file | Nhãn gốc | Quy đổi về 2 lớp |
|---|---|---|---|---|---|
| 1 | **UIT-ViCTSD** (Vietnamese Constructiveness and Toxicity Dataset) | Kaggle `phanvmhuyn/uit-victsd` | 3 (train 7.000 / dev 2.000 / test 1.000) | `Toxicity` 0/1 | 0 → vô hại, 1 → toxic |
| 2 | **ViHSD** (Vietnamese Hate Speech Detection) | Kaggle `cthng123/hate-speech-detection-vietnamese` | 3 (train 5.166 / val 1.112 / test 2.224) | 0/1/2 (clean/hate/offensive) | 0 → vô hại, 1–2 → toxic |

### 2.2 Tiền xử lý

- Nạp qua `src/data_loader.py` với `--mapping auto`: tự nhận diện schema nhãn từng file theo GIÁ TRỊ nhãn (không phụ thuộc tên file), sau đó **gộp mọi nhãn nguồn về đúng 2 lớp đề tài**: `vohai` (Toxicity=0, clean) và `toxic` (Toxicity=1, hate/offensive). Việc gộp nằm ngay trong data_loader nên baseline, PhoBERT lẫn phân tích lỗi đều thấy cùng một nhãn thống nhất.
- Chuẩn hoá văn bản bằng `src/preprocess.py`: làm sạch, chuẩn hoá teencode, tokenize tiếng Việt.
- Bỏ dòng trùng theo **nội dung** (không theo id) — toàn cục trước khi chia tập, tránh một bản sao rơi vào cả train lẫn test làm điểm test đẹp giả.

### 2.3 Kết quả nạp (từ log kernel)

| File | Dòng giữ lại | Vô hại | Toxic |
|---|---|---|---|
| df_train_clean (ViCTSD) | 6.969 | 6.210 | 759 |
| df_dev_clean (ViCTSD) | 1.992 | 1.760 | 232 |
| df_test_clean (ViCTSD) | 997 | 887 | 110 |
| vihsd_train | 4.867 | 2.092 | 2.775 |
| vihsd_val | 731 | 290 | 441 |
| vihsd_test | 1.441 | 600 | 841 |
| **Tổng sau dedupe** | **16.683** | **11.811** | **4.872** |

Tỷ lệ lớp: vô hại 70,8% / toxic 29,2% — lệch vừa phải, xử lý bằng class_weight="balanced" ở các mô hình baseline.

Chia dữ liệu: **85% train / 15% hold-out test**, stratified, `random_state=42`. Tập test 2.503 mẫu (vô hại 1.772, toxic 731), chỉ chạm đúng một lần ở cuối.

## 3. Phương pháp

### 3.1 Baseline TF-IDF (src/train_baseline.py)

- Feature: TF-IDF word 1-2gram (30k features) + char_wb 2-5gram (50k features), sublinear tf.
- 3 mô hình (đều class_weight balanced): MultinomialNB, LogisticRegression, LinearSVC.
- Chọn model bằng **CV 5-fold stratified trên tập train** (f1_macro), không chạm test.
- So sánh thêm: hard Voting ensemble 3 model.
- `src/analyze_errors.py`: confusion matrix, danh sách mẫu nhầm kèm độ tin cậy, top-10 từ khóa mạnh nhất lớp toxic (trọng số hồi quy).

### 3.2 PhoBERT fine-tune (src/train_phobert.py) — điểm cộng

- `vinai/phobert-base`, 3 epoch, lr 2e-5, batch 32, max_len 128, fp16 trên GPU Kaggle.
- Chọn model tốt nhất theo f1_macro (load_best_model_at_end).

## 4. Kết quả

### 4.1 Baseline — CV5 trên tập train (f1_macro, từ log kernel v2)

| Model | f1_macro CV5 |
|---|---|
| Naive Bayes | 0.7757 |
| LogisticRegression | 0.8432 |
| LinearSVC | **0.8436** ← chọn |

### 4.2 Hold-out test 15% (2.503 mẫu, 2 lớp — từ log kernel v2)

| Model | Accuracy | f1_macro | f1 toxic | f1 vô hại |
|---|---|---|---|---|
| LinearSVC (best baseline) | 0.879 | 0.853 | 0.791 | 0.915 |
| Voting ensemble | **0.882** | **0.855** | 0.792 | 0.918 |
| PhoBERT (best epoch 3)* | 0.885 | 0.8565 | 0.792 | 0.921 |

PhoBERT precision/recall lớp toxic: 0.842 / 0.748 — recall toxic thấp hơn baseline (bỏ sót nhiều hơn) nhưng precision cao hơn hẳn (ít quy nhầm bình luận lành hơn). (*) Số PhoBERT chỉ để tham khảo/thể hiện điểm cộng (mục 3.2), mô hình **chính của đề tài là nhóm TF-IDF baseline** — xem lựa chọn ở mục 4.1 và kết luận.

### 4.3 PhoBERT — số chính thức (kernel v2, chạy xong 02/11) — chỉ để tham khảo (điểm cộng)

Kernel version 2 (push ~16:10 ngày 02/11) đã chạy COMPLETE. Diễn biến eval_f1 theo epoch: **0.835 → 0.8538 → 0.8565** — chọn model epoch 3 (load_best_model_at_end). Kết quả trên hold-out test 15%: accuracy 0.885, f1_macro 0.8565, f1 toxic 0.792, f1 vô hại 0.921 (chi tiết ở bảng 4.2).

Lưu ý: như mục 6.1, eval_f1 khi train tính trên chính tập test nên dãy 0.835→0.8565 không phải số selection sạch; số hold-out nêu trên vẫn là dự đoán của model tốt nhất trên test nhưng cần đọc kèm lưu ý đó.

### 4.4 Confusion matrix (LinearSVC, hàng = thực tế, cột = dự đoán)

| | toxic | vô hại |
|---|---|---|
| **toxic** | 572 | 159 |
| **vô hại** | 143 | 1629 |

Số liệu khớp với classification report của kernel v2 trong log (recall toxic 0.782, recall vô hại 0.919, accuracy 0.879). Recall toxic = 572/731 = **0.782** — bắt được ~78% bình luận độc hại; precision toxic = 0.800 (143 bình luận lành bị quy nhầm).

### 4.5 Top-10 từ khóa mạnh nhất lớp toxic (từ analyze_errors — đã chạy trên kernel v2)

`cha mẹ, đồng bọn, thì ra, làm xấu, bọn tao, như mày, khủng khiếp, chó nó, cá…` — chủ yếu là đại từ miệt thị/chửi thề đặc trưng của hate speech tiếng Việt, nhất quán với trực giác.

## 5. Phân tích lỗi

1. **Toxic bị bỏ sót 159/731 (recall 0.782):** các mẫu nhầm điển hình là toxic kiểu ẩn — mỉa mai, nói giảm ("quá tội cho trẻ em bây giờ, không biết tại sao bị đánh mất tuổi thơ"), chửi không từ ngữ thô — TF-IDF n-gram không bắt được ngữ nghĩa.
2. **Vô hại bị quy toxic 143 mẫu:** đa phần là câu kể chuyện mang từ ngữ nhạy cảm không chủ ý ("dạy theo sách, giáo viên nào dám tự ý nâng cao độ khó vậy?") — giới hạn của phương pháp dựa trên từ khóa bề mặt.
3. LinearSVC và LogReg gần như đồng cấp (0.8432 vs 0.8427 CV5); Voting ensemble nhỉnh hơn chút về accuracy nhưng f1 toxic tương đương — chọn đơn lẻ LinearSVC là hợp lý về chi phí.
4. Naive Bayes thấp hơn hẳn (0.776 vs 0.843): TF-IDF sublinear + char 5gram vi phạm giả định độc lập đặc trưng của NB — vẫn giữ trong bảng để minh họa so sánh theo yêu cầu môn.

## 6. Thiếu xót & hạn chế (trung thực)

1. **[Đánh giá] eval_f1 của PhoBERT tính trên chính tập test** (eval_dataset = te_ds trong Trainer), tức model "nhìn" test 3 lần trong lúc chọn epoch — vi phạm nguyên tắc test chỉ chạm một lần. Cần tách dev set từ train để selection/early-stopping. Số PhoBERT ở mục 4.3 được ghi kèm lưu ý này; muốn số hoàn toàn sạch cần tách dev set và chạy lại.
2. **[Dữ liệu] Không phải bình luận SV PTIT thật.** ViCTSD là bình luận báo chí điện tử, ViHSD là bình luận mạng xã hội. Kết quả chỉ mô phỏng; khi áp vào bình luận nhóm/fanpage SV PTIT (teencode, viết tắt nặng hơn) hiệu năng có thể giảm.
3. **[Dữ liệu] Lệch lớp còn lại:** toxic chiếm 29,2%, đã dùng class_weight nhưng chưa thử oversample/threshold tuning — có thể cải thiện recall toxic nếu đổi ngưỡng quyết định.
4. **[Dữ liệu] Chưa kiểm tra near-duplicate chéo nguồn** (chỉ dedupe text y hệt); ViCTSD và ViHSD có thể có câu gần trùng nằm cả 2 tập.
5. **[Dữ liệu] Hai nhãn nguồn không đồng đều về ngữ nghĩa:** "toxic" của ViCTSD (bình luận báo chí, mức nhẹ hơn) và "hate/offensive" của ViHSD (mạng xã hội, nặng hơn) được gộp chung một lớp — ranh giới lớp không hoàn toàn nhất quán giữa 2 nguồn.
6. **[Phạm vi] "Vô hại" ≠ "constructive".** ViCTSD còn có nhãn Constructiveness (0/1) mà pipeline không dùng; tận dụng có thể tách 3 mức: constructive / vô hại / toxic.
7. **[Pipeline] analyze_errors.py đã fix 2 lỗi** (IndexError confidence; top-features cho coef_ nhị phân) và **đã chạy hoàn chỉnh trên kernel v2** — danh sách mẫu nhầm, top-10 từ khóa (mục 4.4/4.5) giờ lấy trực tiếp từ log kernel, không còn là số kiểm chứng máy.

## 7. Hướng phát triển

- **Ngắn hạn:** [đã xong] số PhoBERT 2 lớp chính thức + analyze_errors trên kernel v2 đã cập nhật vào báo cáo; còn lại: tách dev set để selection PhoBERT sạch, thử threshold tuning để tăng recall toxic.
- **Trung hạn:** ensemble PhoBERT + TF-IDF; oversample cho lớp toxic; xử lý teencode/emoji đặc trưng bình luận SV.
- **Dài hạn:** xin quyền bộ PTIT gated hoặc crawl bình luận nhóm SV PTIT (đúng môi trường đích); mở rộng 3 lớp constructive / vô hại / toxic dùng nhãn Constructiveness của ViCTSD.

## 8. Kết luận

Pipeline 2 dataset → gộp nhãn 2 lớp → TF-IDF baseline + PhoBERT hoạt động trọn vẹn headless trên Kaggle GPU. Kết quả chính của đề tài: **Voting ensemble — accuracy 0.882, f1_macro 0.855** (LinearSVC đơn lẻ tương đương 0.879/0.853, CV5 0.8436) — recall lớp toxic 0.782. PhoBERT fine-tune được thực hiện thêm như điểm cộng (kiến thức ngoài chương trình, mục 3.2) đạt 0.885/0.8565, nhưng không dùng làm mô hình chính. Các thiếu xót chính đã liệt kê trung thực kèm hướng sửa cụ thể.
