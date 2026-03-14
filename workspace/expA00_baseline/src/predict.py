"""推論モジュール: sitting_prediction 関数を提供

LightGBM 5-fold アンサンブルモデルによる sitting/standing 判定。
CV Macro F1: 0.7915
"""

import os
import math
import pickle
import numpy as np
from pathlib import Path

# モデル読み込み（モジュールロード時に1回だけ）
_MODELS = None
_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# COCO 17 keypoint indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


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


def _angle_between(p1, p2, p3):
    """3点 p1-p2-p3 のp2における角度(度)を計算"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def _extract_features(data: dict) -> np.ndarray:
    """入力dictから特徴量ベクトルを抽出する

    学習時と同じ順序・同じ計算で22個の特徴量を生成。
    特徴量はbbox正規化したkeypoint位置、関節角度、
    keypoint信頼度などで構成される。
    """
    bbox = data["bbox"]
    kps = np.array(data["keypoints"])  # (17, 2)
    scores = np.array(data["keypoint_scores"])  # (17,)
    bx, by, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

    # bbox正規化キーポイント
    rel_kps = np.zeros_like(kps)
    rel_kps[:, 0] = (kps[:, 0] - bx) / (bw + 1e-8)
    rel_kps[:, 1] = (kps[:, 1] - by) / (bh + 1e-8)

    features = []

    # 1. bbox aspect ratio
    features.append(bh / (bw + 1e-8))

    # 2-3. hip-knee垂直距離
    features.append(rel_kps[LEFT_KNEE, 1] - rel_kps[LEFT_HIP, 1])
    features.append(rel_kps[RIGHT_KNEE, 1] - rel_kps[RIGHT_HIP, 1])

    # 4-5. hip-ankle垂直距離
    features.append(rel_kps[LEFT_ANKLE, 1] - rel_kps[LEFT_HIP, 1])
    features.append(rel_kps[RIGHT_ANKLE, 1] - rel_kps[RIGHT_HIP, 1])

    # 6-9. 関節角度
    features.append(_angle_between(kps[LEFT_SHOULDER], kps[LEFT_HIP], kps[LEFT_KNEE]))
    features.append(_angle_between(kps[RIGHT_SHOULDER], kps[RIGHT_HIP], kps[RIGHT_KNEE]))
    features.append(_angle_between(kps[LEFT_HIP], kps[LEFT_KNEE], kps[LEFT_ANKLE]))
    features.append(_angle_between(kps[RIGHT_HIP], kps[RIGHT_KNEE], kps[RIGHT_ANKLE]))

    # 10-11. hip相対y座標
    features.append(rel_kps[LEFT_HIP, 1])
    features.append(rel_kps[RIGHT_HIP, 1])

    # 12-13. knee相対y座標
    features.append(rel_kps[LEFT_KNEE, 1])
    features.append(rel_kps[RIGHT_KNEE, 1])

    # 14-15. 下半身keypoint信頼度
    lower_body = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
    features.append(float(np.mean(scores[lower_body])))
    features.append(float(np.min(scores[lower_body])))

    # 16-19. 個別keypoint信頼度
    features.append(scores[LEFT_KNEE])
    features.append(scores[RIGHT_KNEE])
    features.append(scores[LEFT_ANKLE])
    features.append(scores[RIGHT_ANKLE])

    # 20. torso/leg比率
    upper_y = (rel_kps[LEFT_SHOULDER, 1] + rel_kps[RIGHT_SHOULDER, 1]) / 2
    hip_y = (rel_kps[LEFT_HIP, 1] + rel_kps[RIGHT_HIP, 1]) / 2
    ankle_y = (rel_kps[LEFT_ANKLE, 1] + rel_kps[RIGHT_ANKLE, 1]) / 2
    torso_len = hip_y - upper_y
    leg_len = ankle_y - hip_y
    features.append(torso_len / (leg_len + 1e-8))

    # 21-22. knee x方向オフセット
    features.append(rel_kps[LEFT_KNEE, 0] - rel_kps[LEFT_HIP, 0])
    features.append(rel_kps[RIGHT_KNEE, 0] - rel_kps[RIGHT_HIP, 0])

    return np.array(features).reshape(1, -1)


def sitting_prediction(data: dict) -> str:
    """人物が座っているか立っているかを判定する

    LightGBM 5-foldアンサンブルモデルを使用。
    各foldのモデルの予測確率を平均し、0.5を閾値として判定する。
    特徴量はbbox正規化keypoint位置、関節角度、信頼度スコアなど
    22次元のベクトルで構成される。

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
    X = _extract_features(data)

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
