"""学習スクリプト: LightGBM + StratifiedKFold(5) で sitting/standing 分類

expA02: expA01の40次元特徴量をベースに、per-tree sampling から per-node sampling に変更。
  colsample_bytree: 0.8 → 1.0 (per-tree sampling を無効化)
  colsample_bynode: (未設定) → 0.7 (per-node: 40特徴 × 70% ≈ 28特徴/split)
重要特徴が1ノードで外れても別ノードで使われるため、選択機会が増える。
"""

import json
import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_features

SEED = 42
N_FOLDS = 5
DATA_DIR = Path("C:/Users/Hidekazu/Downloads/dataset_resselect/dataset_reselect")
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_data():
    """JSONファイルを読み込み、特徴量とラベルを返す"""
    X_list = []
    y_list = []
    filenames = []

    for label_str, label_int in [("sitting", 0), ("standing", 1)]:
        folder = DATA_DIR / f"output_jsons_{label_str}"
        for fp in sorted(folder.glob("*.json")):
            with open(fp) as f:
                data = json.load(f)
            feats = extract_features(data)
            X_list.append(feats)
            y_list.append(label_int)
            filenames.append(fp.name)

    feature_names = list(X_list[0].keys())
    X = np.array([[row[k] for k in feature_names] for row in X_list], dtype=float)
    y = np.array(y_list)
    return X, y, feature_names, filenames


def train_and_evaluate():
    np.random.seed(SEED)
    X, y, feature_names, filenames = load_data()
    print(f"データ: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"クラス分布: sitting={np.sum(y == 0)}, standing={np.sum(y == 1)}")
    print(f"特徴量: {feature_names}\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(y))
    fold_scores = []
    models = []

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": SEED,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 20,
        "max_depth": 6,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 1.0,      # per-tree sampling を無効化
        "colsample_bynode": 0.7,      # per-node: 40特徴 × 70% ≈ 28特徴/split
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(0)],
        )

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        fold_f1 = f1_score(y_val, val_pred, average="macro")
        fold_scores.append(fold_f1)
        models.append(model)
        print(f"Fold {fold}: Macro F1 = {fold_f1:.4f}")

    # 全体OOF結果
    overall_f1 = f1_score(y, oof_preds, average="macro")
    print(f"\n=== Overall OOF Macro F1: {overall_f1:.4f} ===")
    print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"Mean: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}\n")

    print("Classification Report:")
    print(classification_report(y, oof_preds, target_names=["sitting", "standing"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y, oof_preds))

    # Feature importance
    print("\nFeature Importance (gain):")
    avg_importance = np.mean(
        [m.feature_importances_ for m in models], axis=0
    )
    sorted_idx = np.argsort(avg_importance)[::-1]
    for i in sorted_idx:
        print(f"  {feature_names[i]:35s} {avg_importance[i]:.1f}")

    # モデル保存
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for fold, model in enumerate(models):
        model_path = RESULTS_DIR / f"lgbm_fold{fold}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    print(f"\nモデル保存先: {RESULTS_DIR}")

    return overall_f1, fold_scores, models, feature_names


if __name__ == "__main__":
    train_and_evaluate()
