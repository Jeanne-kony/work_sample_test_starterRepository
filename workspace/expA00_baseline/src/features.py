"""特徴量抽出モジュール: keypointsとbboxからsitting/standing判別用の特徴量を計算"""

import math
import numpy as np


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


def _angle_between(p1, p2, p3):
    """3点 p1-p2-p3 のp2における角度(度)を計算"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def extract_features(data: dict) -> dict:
    """入力dictから特徴量dictを抽出する

    Args:
        data: bbox, bbox_confidence, keypoints, keypoint_scores を含むdict

    Returns:
        特徴量名をkey、値をvalueとするdict
    """
    bbox = data["bbox"]
    kps = np.array(data["keypoints"])  # (17, 2)
    scores = np.array(data["keypoint_scores"])  # (17,)
    bx, by, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

    # bbox正規化キーポイント
    rel_kps = np.zeros_like(kps)
    rel_kps[:, 0] = (kps[:, 0] - bx) / (bw + 1e-8)
    rel_kps[:, 1] = (kps[:, 1] - by) / (bh + 1e-8)

    features = {}

    # 1. bbox aspect ratio (最も単純で強力)
    features["bbox_aspect_ratio"] = bh / (bw + 1e-8)

    # 2. hip-knee垂直距離 (最強シグナル d>1.1)
    features["left_hip_knee_vert"] = rel_kps[LEFT_KNEE, 1] - rel_kps[LEFT_HIP, 1]
    features["right_hip_knee_vert"] = rel_kps[RIGHT_KNEE, 1] - rel_kps[RIGHT_HIP, 1]

    # 3. hip-ankle垂直距離
    features["left_hip_ankle_vert"] = rel_kps[LEFT_ANKLE, 1] - rel_kps[LEFT_HIP, 1]
    features["right_hip_ankle_vert"] = rel_kps[RIGHT_ANKLE, 1] - rel_kps[RIGHT_HIP, 1]

    # 4. 関節角度
    features["angle_l_shoulder_hip_knee"] = _angle_between(
        kps[LEFT_SHOULDER], kps[LEFT_HIP], kps[LEFT_KNEE]
    )
    features["angle_r_shoulder_hip_knee"] = _angle_between(
        kps[RIGHT_SHOULDER], kps[RIGHT_HIP], kps[RIGHT_KNEE]
    )
    features["angle_l_hip_knee_ankle"] = _angle_between(
        kps[LEFT_HIP], kps[LEFT_KNEE], kps[LEFT_ANKLE]
    )
    features["angle_r_hip_knee_ankle"] = _angle_between(
        kps[RIGHT_HIP], kps[RIGHT_KNEE], kps[RIGHT_ANKLE]
    )

    # 5. hip相対y座標
    features["rel_left_hip_y"] = rel_kps[LEFT_HIP, 1]
    features["rel_right_hip_y"] = rel_kps[RIGHT_HIP, 1]

    # 6. knee相対y座標
    features["rel_left_knee_y"] = rel_kps[LEFT_KNEE, 1]
    features["rel_right_knee_y"] = rel_kps[RIGHT_KNEE, 1]

    # 7. 下半身キーポイントの信頼度
    lower_body_indices = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
    features["mean_conf_lower"] = float(np.mean(scores[lower_body_indices]))
    features["min_conf_lower"] = float(np.min(scores[lower_body_indices]))
    features["conf_left_knee"] = scores[LEFT_KNEE]
    features["conf_right_knee"] = scores[RIGHT_KNEE]
    features["conf_left_ankle"] = scores[LEFT_ANKLE]
    features["conf_right_ankle"] = scores[RIGHT_ANKLE]

    # 8. 上半身-下半身の高さ比
    upper_y = np.mean([rel_kps[LEFT_SHOULDER, 1], rel_kps[RIGHT_SHOULDER, 1]])
    hip_y = np.mean([rel_kps[LEFT_HIP, 1], rel_kps[RIGHT_HIP, 1]])
    ankle_y = np.mean([rel_kps[LEFT_ANKLE, 1], rel_kps[RIGHT_ANKLE, 1]])
    torso_len = hip_y - upper_y
    leg_len = ankle_y - hip_y
    features["torso_leg_ratio"] = torso_len / (leg_len + 1e-8)

    # 9. knee水平位置（sitting時は前方に出やすい）
    features["left_knee_x_offset"] = rel_kps[LEFT_KNEE, 0] - rel_kps[LEFT_HIP, 0]
    features["right_knee_x_offset"] = rel_kps[RIGHT_KNEE, 0] - rel_kps[RIGHT_HIP, 0]

    return features
