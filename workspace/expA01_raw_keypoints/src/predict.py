"""推論モジュール: sitting_prediction 関数を提供

LightGBM 5-fold アンサンブルモデルによる sitting/standing 判定。
expA01: 下半身6KPの生座標を追加した40次元特徴量を使用。
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_features

# モデル読み込み（モジュールロード時に1回だけ）
_MODELS = None
_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _load_models():
    """学習済みLightGBMモデルを遅延読み込み"""
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    _MODELS = []
    for fold in range(5):
        model_path = _RESULTS_DIR / f"lgbm_fold{fold}.pkl"
        with open(model_path, "rb") as f:
            _MODELS.append(pickle.load(f))
    return _MODELS


def sitting_prediction(data: dict) -> str:
    """人物が座っているか立っているかを判定する

    LightGBM 5-foldアンサンブルモデルを使用。
    各foldのモデルの予測確率を平均し、0.5を閾値として判定する。
    特徴量はbbox正規化keypoint位置、関節角度、信頼度スコア、
    下半身6KPの生座標など40次元のベクトルで構成される。

    Args:
        data: 以下のキーを持つdict
            - bbox: {"x": float, "y": float, "w": float, "h": float}
              人物を囲む矩形の左上頂点座標と幅・高さ（ピクセル）
            - bbox_confidence: float
              矩形推定の信頼度
            - keypoints: 17要素のリスト、各要素は[x, y]の2要素リスト
              COCO形式の関節点座標
            - keypoint_scores: 17要素のリスト
              各関節点の推定信頼度

    Returns:
        "sitting" - 座っていると判定された場合
        "standing" - 立っていると判定された場合
    """
    models = _load_models()
    feats = extract_features(data)
    feature_names = list(feats.keys())
    X = np.array([[feats[k] for k in feature_names]], dtype=float)

    # 5-foldモデルの予測確率を平均
    proba_sum = 0.0
    for model in models:
        proba = model.predict_proba(X)[0, 1]  # standing確率
        proba_sum += proba
    avg_proba = proba_sum / len(models)

    # 0.5を閾値として判定（0=sitting, 1=standing）
    if avg_proba < 0.5:
        return "sitting"
    else:
        return "standing"
