# SESSION_NOTES: expA00_baseline

## セッション情報
- **日付**: 2026-03-14
- **作業フォルダ**: workspace/expA00_baseline
- **目標**: sitting/standing分類のベースラインモデル構築

## 仮説
- bbox aspect ratio と hip-knee垂直距離が最も判別力が高い (EDAでCohen's d > 0.86)
- 幾何学的特徴量 + 信頼度情報で Macro F1 0.85+ を目指す

## 試したアプローチと結果

| アプローチ | 変更点 | CV (Macro F1) | LB | 備考 |
|-----------|--------|-----|-----|------|
| ルールベース v1 | スコアリング方式、5特徴量 | 0.7172 (全データ) | - | aspect ratioのif/elif順バグあり |
| ルールベース v2 | バグ修正 + 8特徴量 | 0.7300 (全データ) | - | standingの誤分類が多い |
| LightGBM 5-fold | 22特徴量, StratifiedKFold | **0.7915** | - | OOF評価。Fold std=0.022 |

## ファイル構成
- `src/features.py` - 特徴量抽出 (22特徴量)
- `src/train.py` - LightGBM学習 + StratifiedKFold CV
- `src/predict.py` - **sitting_prediction関数** (LightGBM 5-fold アンサンブル)
- `src/evaluate.py` - 評価スクリプト
- `results/lgbm_fold{0-4}.pkl` - 学習済みモデル

## 重要な知見
- **bbox aspect ratioが特徴量重要度1位** (gain=511)、続いてknee x-offset (370)、hip-knee-ankle角度 (330)
- hip-knee垂直距離はCohen's dでは最強(1.17)だが、LightGBMの重要度では中位（角度特徴量との共線性）
- 下半身keypoint信頼度は単独では弱いが、モデルに含めると補助的に効く
- ルールベース(0.73) vs LightGBM(0.79): 約6ポイント差。非線形な特徴量組み合わせが効いている

## 性能変化の記録

| 実験 | 変更内容 | 結果 (Macro F1) | 改善幅 |
|------|---------|------|--------|
| rule_v1 | 初期ルールベース | 0.7172 | baseline |
| rule_v2 | バグ修正+特徴量追加 | 0.7300 | +0.013 |
| lgbm_baseline | LightGBM 22feat | 0.7915 | +0.062 |

## コマンド履歴
```bash
python3 workspace/expA00_baseline/src/train.py      # LightGBM学習+CV
python3 workspace/expA00_baseline/src/evaluate.py    # 評価
```

## 次のステップ
- [ ] 閾値最適化 (0.5以外の閾値でMacro F1改善の余地)
- [ ] 特徴量追加: 全17 keypoint の相対座標を直接入力
- [ ] 特徴量追加: keypoint間のユークリッド距離
- [ ] confidence score の17次元をそのまま特徴量に
- [ ] XGBoost / CatBoost との比較
- [ ] 爆発案: keypoint confidence scoreだけで分類
