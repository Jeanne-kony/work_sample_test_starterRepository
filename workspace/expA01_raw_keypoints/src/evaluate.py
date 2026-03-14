"""評価スクリプト: sitting_prediction関数の性能をMacro F1で評価"""

import json
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from predict import sitting_prediction

SEED = 42
DATA_DIR = Path("C:/Users/Hidekazu/Downloads/dataset_resselect/dataset_reselect")


def evaluate():
    """全データに対してsitting_predictionを実行し、Macro F1を計算"""
    y_true = []
    y_pred = []
    errors = []

    for label_str in ["sitting", "standing"]:
        folder = DATA_DIR / f"output_jsons_{label_str}"
        for fp in sorted(folder.glob("*.json")):
            with open(fp) as f:
                data = json.load(f)
            pred = sitting_prediction(data)
            y_true.append(label_str)
            y_pred.append(pred)
            if pred != label_str:
                errors.append((fp.name, label_str, pred))

    macro_f1 = f1_score(y_true, y_pred, average="macro", pos_label=None)
    print(f"=== Macro F1: {macro_f1:.4f} ===\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix (rows=true, cols=pred):")
    print("           sitting  standing")
    cm = confusion_matrix(y_true, y_pred, labels=["sitting", "standing"])
    print(f"sitting    {cm[0][0]:7d}  {cm[0][1]:8d}")
    print(f"standing   {cm[1][0]:7d}  {cm[1][1]:8d}")

    print(f"\n誤分類数: {len(errors)} / {len(y_true)}")
    if errors:
        print("\n誤分類サンプル (先頭20件):")
        for fname, true, pred in errors[:20]:
            print(f"  {fname}: true={true}, pred={pred}")

    return macro_f1


if __name__ == "__main__":
    evaluate()
