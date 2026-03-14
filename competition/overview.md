# Competition Overview: Sitting vs Standing Pose Classification

Generated: 2026-03-14

## Dataset Summary

| Item | Value |
|---|---|
| Total samples | 927 |
| Sitting | 440 (47.5%) |
| Standing | 487 (52.5%) |
| Class imbalance ratio | ~1.11:1 (mild, negligible) |
| Features per sample | 17 keypoints x (x,y,score) + bbox + derived = 108 columns |
| Missing values | None |
| Data format | JSON per image, COCO 17-keypoint format |

## Target Variable

- Binary classification: `sitting` (0) vs `standing` (1)
- Class distribution is well-balanced (~47.5% / 52.5%). No special handling needed.

---

## Key Findings

### 1. Most Discriminative Features (Cohen's d ranking)

| Rank | Feature | Cohen's d | Sitting mean | Standing mean |
|---|---|---|---|---|
| 1 | right_hip_knee_vert_dist (relative) | **1.17** | 0.056 | 0.192 |
| 2 | left_hip_knee_vert_dist (relative) | **1.16** | 0.058 | 0.194 |
| 3 | bbox_aspect_ratio (h/w) | **0.86** | 1.33 | 1.80 |
| 4 | angle_left_shoulder_hip_knee | **0.85** | 112.5 deg | 145.4 deg |
| 5 | angle_right_shoulder_hip_knee | **0.80** | 112.5 deg | 145.4 deg |
| 6 | left/right_hip_ankle_span | **0.76** | 0.199 | 0.311 |
| 7 | angle_right_hip_knee_ankle | **0.65** | 102.8 deg | 131.5 deg |
| 8 | angle_left_hip_knee_ankle | **0.58** | 102.5 deg | 128.8 deg |
| 9 | rel_left/right_hip_y | **0.51** | 0.792 | 0.693 |

**Key insight**: The hip-to-knee vertical distance (relative to bbox) is the single strongest signal with d > 1.1. In standing posture, the knee is much further below the hip than in sitting.

### 2. Angle Analysis

- **Hip-Knee-Ankle angle**: sitting ~102 deg vs standing ~129 deg. Standing posture has a more extended knee (larger angle toward 180 deg = straight leg).
- **Shoulder-Hip-Knee angle**: sitting ~112 deg vs standing ~145 deg. The largest angular difference (33 deg gap). In sitting, the torso-thigh-shank chain is more folded.
- Both left and right sides agree well — bilateral symmetry is consistent.
- Angles range from near 0 to 180 degrees in both classes, indicating high variance. Many samples have partially occluded lower-body keypoints.

### 3. Bbox Aspect Ratio

- Sitting: mean h/w = **1.33** (more square, person occupies wider relative bounding box)
- Standing: mean h/w = **1.80** (taller and narrower)
- This is a strong signal (d = 0.86) and completely independent of keypoint confidence.

### 4. Relative Hip Position (normalized by bbox)

- In sitting, the hip y-coordinate is at ~79% of bbox height from top.
- In standing, the hip is at ~69% of bbox height — higher in the frame relative to the body extent.
- This reflects that in standing, legs extend fully below the hip.

---

## Data Quality Issues

### Low Keypoint Confidence (Lower Body)

| Threshold | Affected samples | Pct |
|---|---|---|
| < 0.3 (any lower-body KP) | 580 / 927 | **62.6%** |
| < 0.05 (any lower-body KP) | 408 / 927 | **44.0%** |

- Lower-body keypoints (knees, ankles, hips) frequently have very low confidence.
- This is a critical issue: many sitting samples likely have partially visible lower limbs (e.g., seated behind a desk or table).
- The `knee_ratio` feature was unstable (numerical instability from division by near-zero ankle-hip span), so it should not be used directly.

### Occlusion pattern

- In sitting class: ankle and knee keypoints often score near 0 (occluded by chair/table).
- The mean lower-body confidence is similar between classes (sitting: 0.582, standing: 0.595), suggesting confidence alone is not a clean discriminator.
- **Recommendation**: treat low-confidence keypoints as missing; impute or mask them during model training.

### Bbox sizes

- Sitting: bbox_w mean = 529 px, bbox_h = 639 px
- Standing: bbox_w mean = 356 px, bbox_h = 577 px
- Sitting images tend to capture wider crops (including chairs/environment), while standing crops are more body-focused.

---

## Fold Design Recommendation

- **StratifiedKFold (k=5)** is appropriate here.
- No time-series or group structure was detected from filenames.
- Class balance is mild but use stratified split for correctness.

---

## Feature Engineering Ideas

### Solid Approaches

1. **Hip-knee vertical distance** (normalized by bbox): highest discriminative power (d > 1.1). Already computed — use as primary feature.
2. **Bbox aspect ratio** (h/w): strong, noise-free proxy.
3. **Shoulder-Hip-Knee angle** (left + right average): d = 0.82 after averaging bilateral.
4. **Hip-ankle span** (relative): strong proxy for leg extension.
5. **Confidence-weighted angles**: down-weight angle features when keypoint score < threshold (e.g., 0.3).
6. **Body part occlusion indicator**: binary flag for whether ankle/knee are confidently detected. Heavily occluded lower body is strong evidence of sitting.
7. **Symmetry features**: left-right difference in angles or positions (detects unusual poses or occlusion asymmetry).

### Feature Selection for Simple Classifiers

A logistic regression or gradient boosting model with these ~10 features should perform strongly:

```
[bbox_aspect_ratio,
 angle_left_shoulder_hip_knee, angle_right_shoulder_hip_knee,
 angle_left_hip_knee_ankle, angle_right_hip_knee_ankle,
 left_hip_knee_vert_dist, right_hip_knee_vert_dist,
 left_hip_ankle_span, right_hip_ankle_span,
 rel_left_hip_y, rel_right_hip_y,
 mean_conf_lower]
```

### Explosive Approaches

1. **Treat as a geometric/topological problem**: instead of angles, compute the convex hull area ratio of upper-body keypoints vs lower-body keypoints. Sitting collapses the lower hull.
2. **Use keypoint confidence scores as features themselves**: the occlusion pattern (which keypoints are invisible) is structurally different between sitting and standing — train a model purely on the 17 confidence scores.
3. **Graph Neural Network on skeleton**: represent each sample as a graph (17 nodes, COCO skeleton edges) and apply a GNN classifier. Captures multi-hop structural relationships that angle features miss.
4. **Self-supervised pose pretraining**: treat all 927 samples as unlabeled, pretrain an autoencoder on the (17, 2) keypoint sequences, then fine-tune on labels. Might capture pose manifold structure not reflected in handcrafted features.
5. **Borrow from action recognition**: compute motion-inspired features even for static poses (treat left/right body halves as "temporal" frames to create pseudo-temporal sequences and use 1D-Conv).

---

## Visualizations

All saved to `/Users/estyle-155/Documents/work_sample_test_starterRepository/competition/`:

| File | Description |
|---|---|
| `class_distribution.png` | Bar chart of sitting (440) vs standing (487) |
| `bbox_aspect_ratio.png` | Overlapping histograms by class — clear separation |
| `keypoint_relative_positions.png` | Relative y-positions of hip, knee, ankle by class |
| `keypoint_confidence_distributions.png` | Mean confidence per keypoint for each class |
| `discriminative_scatter.png` | Scatter plot of top discriminative features |
| `angle_distributions.png` | Angle histograms for all 4 joint angles |
| `skeleton_samples.png` | 4 sample skeletons from each class (normalized to bbox) |
| `correlation_heatmap.png` | Correlation matrix of key features with label |
| `features_df.csv` | Full feature dataframe (927 rows x 108 cols) |

---

## Recommended Modeling Pipeline

1. **Baseline**: Gradient Boosting (XGBoost / LightGBM) on the 10-feature set above. Expected accuracy: ~85-90% based on effect sizes.
2. **Step 2**: Add confidence-weighted features and occlusion flags.
3. **Step 3**: Cross-validate with StratifiedKFold(5), track AUC + accuracy.
4. **Ensemble**: Blend GBM with a simple MLP on full normalized keypoint coordinates.
