# SESSION_NOTES: expA02_feature_fraction_bynode

## 実験概要

**目的**: per-tree sampling（colsample_bytree）から per-node sampling（colsample_bynode）に変更し、重要特徴量の選択機会を増やす。

**仮説**: `colsample_bytree=0.8` では除外された特徴が1ツリーで一切使われない。`colsample_bynode=0.7` にすることで、各ノードで独立にサンプリングするため、重要特徴が1ノードで外れても別ノードで使われる機会がある。Random Forest的な多様性とGBDT的な特徴活用を両立できる。

**ベース実験**: expA01_raw_keypoints (旧データOOF F1=0.8041 / 新データOOF F1=0.8585, 40特徴量)

---

## 変更点

| 項目 | expA01 | expA02 |
|---|---|---|
| 特徴量数 | 40 | 40（同一） |
| colsample_bytree | 0.8 | **1.0** （無効化） |
| colsample_bynode | なし | **0.7** （per-node 70%） |
| num_leaves | 20 | 20（同一） |
| その他HPs | 同一 | 同一 |

---

## 結果

### 学習結果（train.py）

**旧データセット（927件）:**
```
OOF Macro F1: 0.8009  Mean: 0.8008 +/- 0.0304
```

**新データセット（803件）:**
```
Fold 0: Macro F1 = 0.8561
Fold 1: Macro F1 = 0.8445
Fold 2: Macro F1 = 0.8688
Fold 3: Macro F1 = 0.8542
Fold 4: Macro F1 = 0.8550
Overall OOF Macro F1: 0.8558
Mean: 0.8557 +/- 0.0078
```

expA01との比較（新データセット）:

| 指標 | expA01 (案1) | expA02 (案2) | 差分 |
|---|---|---|---|
| OOF Macro F1 | **0.8585** | 0.8558 | -0.003 |
| Fold std | 0.0099 | 0.0078 | expA02の方が安定 |

### 特徴量重要度 Top 10（gain）- 新データセット

```
bbox_aspect_ratio       428.6
left_hip_knee_vert      285.2
raw_right_hip_x         276.6
raw_left_hip_x          265.6
right_knee_x_offset     242.0
left_knee_x_offset      223.0
angle_l_hip_knee_ankle  210.8
angle_r_hip_knee_ankle  201.8
angle_r_shoulder_hip_knee 173.4
right_hip_knee_vert     170.0
```

---

## タスク管理

- [x] expA02フォルダ・srcファイル作成
- [x] train.py: colsample_bytree→1.0, colsample_bynode→0.7
- [x] 旧データで学習（OOF F1=0.8009、expA01旧比-0.003）
- [x] 新データセットに更新・再学習（OOF F1=0.8558、expA01新比-0.003）
- [x] report_draft.md更新
