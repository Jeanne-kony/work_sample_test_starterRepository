# SESSION_NOTES: expA01_raw_keypoints

## 実験概要

**目的**: 下半身6キーポイント（インデックス11-16）のbbox正規化生座標を直接特徴量として追加し、手作り特徴量では捉えられないパターンをLightGBMに学習させる。

**仮説**: 既存の22特徴量はすべて「設計者が定義した幾何学的特徴」であり、非定型姿勢（片足伸ばし座り・足を組む等）では特徴量の計算が崩れる可能性がある。生座標を与えることで、モデルが自力で判別パターンを発見できる。

**ベース実験**: expA00_baseline (OOF Macro F1 = 0.7915)

---

## 変更点

| 項目 | expA00 | expA01 |
|---|---|---|
| 特徴量数 | 22 | 40 (+18) |
| 追加特徴量 | なし | 下半身6KP × (x_norm, y_norm, conf) |
| num_leaves | 31 | 20 (overfitting対策) |
| その他HPs | 同一 | 同一 |

### 追加した特徴量（18次元）

キーポイント11-16それぞれについて:
- `raw_{name}_x`: bbox正規化x座標（信頼度<0.3のときNaN）
- `raw_{name}_y`: bbox正規化y座標（信頼度<0.3のときNaN）
- `raw_{name}_conf`: 信頼度スコア（常に有効）

NaN処理: LightGBMはNaNを自動で欠損として扱うため追加実装不要。

---

## 結果

### 学習結果（train.py）

**旧データセット（927件）:**
```
Fold 0: Macro F1 = 0.7945 / Fold 1: 0.8433 / Fold 2: 0.7994 / Fold 3: 0.7880 / Fold 4: 0.7945
Overall OOF Macro F1: 0.8041  Mean: 0.8040 +/- 0.0200
```

**新データセット（803件）:**
```
Fold 0: Macro F1 = 0.8625
Fold 1: Macro F1 = 0.8445
Fold 2: Macro F1 = 0.8691
Fold 3: Macro F1 = 0.8673
Fold 4: Macro F1 = 0.8488
Overall OOF Macro F1: 0.8585
Mean: 0.8584 +/- 0.0099
```

### 特徴量重要度 Top 10（gain）- 新データセット

```
bbox_aspect_ratio       422.4
raw_right_hip_x         297.4
left_hip_knee_vert      289.8
raw_left_hip_x          262.0
right_knee_x_offset     240.2
left_knee_x_offset      214.2
angle_r_hip_knee_ankle  212.0
angle_l_hip_knee_ankle  211.6
angle_r_shoulder_hip_knee 180.4
raw_left_hip_conf       173.6
```

---

## 考察

- **生座標の有効性**: `raw_*` 系特徴量が重要度上位に入るかどうかで、手作り特徴量では捉えられていなかったパターンの有無を確認できる
- **NaN比率**: 下半身KP低信頼度（<0.3）のサンプルが多い場合、raw_x/yの欠損率が高くなる。これはLightGBMが欠損方向のsplitを活用できることを意味する
- **過学習リスク**: 40次元への増加でnum_leaves=20に下げた。fold間のスコア分散が拡大した場合は正則化を強化する

---

## タスク管理

- [x] expA01フォルダ・srcファイル作成
- [x] features.py: 18特徴量追加
- [x] train.py: パス・num_leaves修正
- [x] predict.py: パス修正
- [x] evaluate.py: パス修正
- [x] train.py実行（OOF F1=0.8041）
- [x] evaluate.py実行
- [x] report_draft.md更新
