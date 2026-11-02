# Kết quả train Kaggle — 2026-11-02 08:08

## Baseline TF-IDF (f1_macro CV5)
```
,f1_macro_cv5
naive_bayes,0.8361465477293091
logreg,0.8917247305307546
svm,0.8925397308596444
```

## PhoBERT
Đã train xong (model quá nặng, không kèm output).

## PhoBERT (test hold-out 15% — đọc từ log kernel kpvb-train)
```
              precision    recall  f1-score   support
         ham      0.933     0.881     0.906       885
        spam      0.899     0.943     0.920       987
    accuracy                          0.914      1872
   macro avg      0.916     0.912     0.913      1872
weighted avg      0.915     0.914     0.914      1872
```
eval_f1 theo epoch: ep1-2 = 0.9053, ep3 = 0.9134 (load_best_model_at_end).
Model lưu trong kernel tại models/phobert/.
