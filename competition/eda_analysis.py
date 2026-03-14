import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ── Keypoint names (COCO 17) ──────────────────────────────────────────────────
KP_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
SITTING_DIR  = "/Users/estyle-155/Documents/work_sample_test_starterRepository/datasets/output_jsons_sitting/"
STANDING_DIR = "/Users/estyle-155/Documents/work_sample_test_starterRepository/datasets/output_jsons_standing/"
OUT_DIR      = "/Users/estyle-155/Documents/work_sample_test_starterRepository/competition/"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Helper: compute angle at vertex B given three 2-D points ──────────────────
def angle_at_B(A, B, C):
    """Angle in degrees at point B in the triangle A-B-C."""
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    cos_a = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


# ── Load all JSON files ───────────────────────────────────────────────────────
def load_dir(directory, label):
    records = []
    for fname in os.listdir(directory):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(directory, fname)
        with open(fpath) as f:
            d = json.load(f)

        bbox = d['bbox']
        kps  = np.array(d['keypoints'])        # (17, 2)
        scs  = np.array(d['keypoint_scores'])  # (17,)

        row = {
            'file': fname,
            'label': label,
            'bbox_x': bbox['x'],
            'bbox_y': bbox['y'],
            'bbox_w': bbox['w'],
            'bbox_h': bbox['h'],
            'bbox_confidence': d['bbox_confidence'],
            'bbox_aspect_ratio': bbox['h'] / (bbox['w'] + 1e-8),
        }

        # Raw keypoints and scores
        for i, name in enumerate(KP_NAMES):
            row[f'kp_{name}_x'] = kps[i, 0]
            row[f'kp_{name}_y'] = kps[i, 1]
            row[f'sc_{name}']   = scs[i]

        # ── Relative keypoints (normalize by bbox) ────────────────────────────
        rel_kps = (kps - np.array([bbox['x'], bbox['y']])) / np.array([bbox['w'] + 1e-8, bbox['h'] + 1e-8])
        for i, name in enumerate(KP_NAMES):
            row[f'rel_{name}_x'] = rel_kps[i, 0]
            row[f'rel_{name}_y'] = rel_kps[i, 1]

        # ── Angle features ────────────────────────────────────────────────────
        # Left: hip(11) – knee(13) – ankle(15)
        row['angle_left_hip_knee_ankle']  = angle_at_B(kps[11], kps[13], kps[15])
        # Right: hip(12) – knee(14) – ankle(16)
        row['angle_right_hip_knee_ankle'] = angle_at_B(kps[12], kps[14], kps[16])
        # Left: shoulder(5) – hip(11) – knee(13)
        row['angle_left_shoulder_hip_knee']  = angle_at_B(kps[5],  kps[11], kps[13])
        # Right: shoulder(6) – hip(12) – knee(14)
        row['angle_right_shoulder_hip_knee'] = angle_at_B(kps[6],  kps[12], kps[14])

        # ── Vertical position ratios ──────────────────────────────────────────
        # knee_y relative to hip_y and ankle_y (all relative coords, y increases downward)
        # In standing: knee is between hip and ankle linearly
        # In sitting:  knee might be at similar height as hip
        left_hip_y    = rel_kps[11, 1]
        left_knee_y   = rel_kps[13, 1]
        left_ankle_y  = rel_kps[15, 1]
        right_hip_y   = rel_kps[12, 1]
        right_knee_y  = rel_kps[14, 1]
        right_ankle_y = rel_kps[16, 1]

        # Ratio: how far is knee between hip and ankle (0=at hip, 1=at ankle)
        row['left_knee_ratio']  = (left_knee_y  - left_hip_y)  / (left_ankle_y  - left_hip_y  + 1e-8)
        row['right_knee_ratio'] = (right_knee_y - right_hip_y) / (right_ankle_y - right_hip_y + 1e-8)

        # Hip-knee vertical distance (relative)
        row['left_hip_knee_vert_dist']  = left_knee_y  - left_hip_y
        row['right_hip_knee_vert_dist'] = right_knee_y - right_hip_y

        # Hip-ankle vertical span
        row['left_hip_ankle_span']  = left_ankle_y  - left_hip_y
        row['right_hip_ankle_span'] = right_ankle_y - right_hip_y

        # Mean confidence (upper body vs lower body)
        row['mean_conf_upper'] = scs[:11].mean()
        row['mean_conf_lower'] = scs[11:].mean()
        row['mean_conf_all']   = scs.mean()
        row['min_conf_lower']  = scs[11:].min()

        records.append(row)
    return records


print("Loading data...")
records = load_dir(SITTING_DIR, 'sitting') + load_dir(STANDING_DIR, 'standing')
df = pd.DataFrame(records)
df['label_int'] = (df['label'] == 'standing').astype(int)

print(f"\n=== DATASET SHAPE ===")
print(f"Total samples: {len(df)}")
print(f"Sitting:  {(df['label']=='sitting').sum()}")
print(f"Standing: {(df['label']=='standing').sum()}")
print(f"Features: {df.shape[1]}")

# ── Missing / NaN check ───────────────────────────────────────────────────────
print(f"\n=== MISSING VALUES ===")
missing = df.isnull().sum()
print(f"Columns with NaN: {(missing > 0).sum()}")
if (missing > 0).sum():
    print(missing[missing > 0])

# ── Basic statistics by class ─────────────────────────────────────────────────
key_features = [
    'bbox_aspect_ratio',
    'bbox_confidence',
    'angle_left_hip_knee_ankle', 'angle_right_hip_knee_ankle',
    'angle_left_shoulder_hip_knee', 'angle_right_shoulder_hip_knee',
    'left_knee_ratio', 'right_knee_ratio',
    'left_hip_knee_vert_dist', 'right_hip_knee_vert_dist',
    'left_hip_ankle_span', 'right_hip_ankle_span',
    'mean_conf_upper', 'mean_conf_lower', 'mean_conf_all', 'min_conf_lower',
    'rel_left_hip_y', 'rel_right_hip_y',
    'rel_left_knee_y', 'rel_right_knee_y',
    'rel_left_ankle_y', 'rel_right_ankle_y',
]

print("\n=== KEY FEATURE STATISTICS BY CLASS ===")
stats = df.groupby('label')[key_features].agg(['mean', 'std', 'min', 'max'])
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(stats.T.to_string())

# ── Discriminative power (effect size / separation) ──────────────────────────
print("\n=== DISCRIMINATIVE POWER (Cohen's d) ===")
sit = df[df['label'] == 'sitting']
std = df[df['label'] == 'standing']
rows = []
for feat in key_features:
    m1, s1 = sit[feat].mean(), sit[feat].std()
    m2, s2 = std[feat].mean(), std[feat].std()
    pooled_std = np.sqrt((s1**2 + s2**2) / 2 + 1e-12)
    d = abs(m1 - m2) / pooled_std
    rows.append({'feature': feat, 'sitting_mean': m1, 'standing_mean': m2, 'cohens_d': d})
disc = pd.DataFrame(rows).sort_values('cohens_d', ascending=False)
print(disc.to_string(index=False))

# ── Data quality: low confidence samples ─────────────────────────────────────
print("\n=== DATA QUALITY: LOW CONFIDENCE KEYPOINTS ===")
lower_kp_cols = [f'sc_{name}' for name in KP_NAMES[11:]]
low_conf_mask = df[lower_kp_cols].min(axis=1) < 0.3
print(f"Samples with at least one lower-body keypoint confidence < 0.3: {low_conf_mask.sum()} ({100*low_conf_mask.mean():.1f}%)")
print(f"  Sitting:  {low_conf_mask[df['label']=='sitting'].sum()}")
print(f"  Standing: {low_conf_mask[df['label']=='standing'].sum()}")

low_conf_mask_005 = df[lower_kp_cols].min(axis=1) < 0.05
print(f"Samples with at least one lower-body keypoint confidence < 0.05: {low_conf_mask_005.sum()} ({100*low_conf_mask_005.mean():.1f}%)")

# ── bbox stats ────────────────────────────────────────────────────────────────
print("\n=== BBOX STATISTICS ===")
for cls in ['sitting', 'standing']:
    sub = df[df['label'] == cls]
    print(f"\n{cls.upper()}:")
    print(f"  bbox_w:  mean={sub['bbox_w'].mean():.1f}, std={sub['bbox_w'].std():.1f}")
    print(f"  bbox_h:  mean={sub['bbox_h'].mean():.1f}, std={sub['bbox_h'].std():.1f}")
    print(f"  aspect_ratio (h/w): mean={sub['bbox_aspect_ratio'].mean():.3f}, std={sub['bbox_aspect_ratio'].std():.3f}")
    print(f"  bbox_confidence: mean={sub['bbox_confidence'].mean():.3f}, min={sub['bbox_confidence'].min():.3f}")

# ── Angle outlier check ───────────────────────────────────────────────────────
print("\n=== ANGLE OUTLIER CHECK ===")
angle_cols = ['angle_left_hip_knee_ankle', 'angle_right_hip_knee_ankle',
              'angle_left_shoulder_hip_knee', 'angle_right_shoulder_hip_knee']
for c in angle_cols:
    print(f"{c}: min={df[c].min():.1f}, max={df[c].max():.1f}, mean={df[c].mean():.1f}")

# Save dataframe for later use
df.to_csv(os.path.join(OUT_DIR, 'features_df.csv'), index=False)
print(f"\nDataFrame saved to {OUT_DIR}features_df.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 1: Class distribution
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['label'].value_counts()
bars = ax.bar(counts.index, counts.values, color=['#2196F3', '#FF5722'], width=0.5, edgecolor='k')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, str(val),
            ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_title('Class Distribution', fontsize=14)
ax.set_ylabel('Count')
ax.set_ylim(0, max(counts.values) * 1.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'class_distribution.png'), dpi=120)
plt.close()
print("Saved: class_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 2: Bbox aspect ratio distribution by class
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4))
for cls, color in [('sitting', '#2196F3'), ('standing', '#FF5722')]:
    sub = df[df['label'] == cls]['bbox_aspect_ratio']
    ax.hist(sub, bins=40, alpha=0.6, color=color, label=cls, density=True, edgecolor='none')
    ax.axvline(sub.mean(), color=color, linewidth=2, linestyle='--', label=f'{cls} mean={sub.mean():.2f}')
ax.set_title('Bbox Aspect Ratio (h/w) Distribution by Class', fontsize=13)
ax.set_xlabel('Aspect Ratio (h/w)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'bbox_aspect_ratio.png'), dpi=120)
plt.close()
print("Saved: bbox_aspect_ratio.png")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 3: Key relative keypoint positions (y-axis, lower body)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
lower_kp_pairs = [
    ('rel_left_hip_y',   'Hip Y (relative)'),
    ('rel_left_knee_y',  'Knee Y (relative)'),
    ('rel_left_ankle_y', 'Ankle Y (relative)'),
]
for ax, (col, title) in zip(axes, lower_kp_pairs):
    for cls, color in [('sitting', '#2196F3'), ('standing', '#FF5722')]:
        sub = df[df['label'] == cls][col]
        ax.hist(sub, bins=30, alpha=0.6, color=color, label=cls, density=True)
        ax.axvline(sub.mean(), color=color, linewidth=2, linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('Relative position (0=top, 1=bottom of bbox)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle('Left Side Keypoint Relative Vertical Positions', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'keypoint_relative_positions.png'), dpi=120, bbox_inches='tight')
plt.close()
print("Saved: keypoint_relative_positions.png")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 4: Keypoint confidence score distributions
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
for ax_idx, cls in enumerate(['sitting', 'standing']):
    sub = df[df['label'] == cls]
    means = [sub[f'sc_{name}'].mean() for name in KP_NAMES]
    stds  = [sub[f'sc_{name}'].std()  for name in KP_NAMES]
    x = np.arange(len(KP_NAMES))
    axes[ax_idx].bar(x, means, yerr=stds, capsize=3, color='#2196F3' if cls=='sitting' else '#FF5722',
                     alpha=0.7, edgecolor='k', linewidth=0.5)
    axes[ax_idx].set_xticks(x)
    axes[ax_idx].set_xticklabels(KP_NAMES, rotation=45, ha='right', fontsize=8)
    axes[ax_idx].set_title(f'Keypoint Confidence Scores — {cls.capitalize()}')
    axes[ax_idx].set_ylabel('Mean Confidence')
    axes[ax_idx].set_ylim(0, 1.1)
    axes[ax_idx].axhline(0.3, color='red', linestyle='--', alpha=0.5, label='threshold 0.3')
    axes[ax_idx].legend(fontsize=8)
    axes[ax_idx].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'keypoint_confidence_distributions.png'), dpi=120)
plt.close()
print("Saved: keypoint_confidence_distributions.png")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 5: Scatter plot — top discriminative features
# ═══════════════════════════════════════════════════════════════════════════════
top2 = disc.iloc[:2]['feature'].tolist()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (feat1, feat2) in zip(axes, [(top2[0], top2[1]), ('bbox_aspect_ratio', top2[0])]):
    for cls, color, marker in [('sitting', '#2196F3', 'o'), ('standing', '#FF5722', 's')]:
        sub = df[df['label'] == cls]
        ax.scatter(sub[feat1], sub[feat2], alpha=0.3, s=15, color=color, label=cls, marker=marker)
    ax.set_xlabel(feat1, fontsize=9)
    ax.set_ylabel(feat2, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(f'{feat1}\nvs {feat2}', fontsize=9)

fig.suptitle('Scatter Plot of Top Discriminative Features', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'discriminative_scatter.png'), dpi=120)
plt.close()
print("Saved: discriminative_scatter.png")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 6: Angle distributions (4 angles, 2 classes)
# ═══════════════════════════════════════════════════════════════════════════════
angle_display = [
    ('angle_left_hip_knee_ankle',    'L Hip-Knee-Ankle'),
    ('angle_right_hip_knee_ankle',   'R Hip-Knee-Ankle'),
    ('angle_left_shoulder_hip_knee', 'L Shoulder-Hip-Knee'),
    ('angle_right_shoulder_hip_knee','R Shoulder-Hip-Knee'),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (col, title) in zip(axes.flat, angle_display):
    for cls, color in [('sitting', '#2196F3'), ('standing', '#FF5722')]:
        sub = df[df['label'] == cls][col]
        ax.hist(sub, bins=35, alpha=0.6, color=color, label=f'{cls} μ={sub.mean():.1f}°', density=True)
        ax.axvline(sub.mean(), color=color, linewidth=2, linestyle='--')
    ax.set_title(f'Angle: {title}', fontsize=11)
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle('Joint Angle Distributions by Class', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'angle_distributions.png'), dpi=120)
plt.close()
print("Saved: angle_distributions.png")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 7: Skeleton visualization for sample images
# ═══════════════════════════════════════════════════════════════════════════════
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),          # face
    (5,6),(5,7),(7,9),(6,8),(8,10),   # upper body
    (5,11),(6,12),(11,12),            # torso
    (11,13),(13,15),(12,14),(14,16),  # legs
]

def draw_skeleton(ax, kps, scores, color, title, bbox):
    # Normalize to bbox
    bx, by, bw, bh = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    rel = (np.array(kps) - np.array([bx, by])) / np.array([bw, bh])
    # Flip y for display (image coords: y increases downward)
    for i, j in SKELETON:
        if scores[i] > 0.1 and scores[j] > 0.1:
            ax.plot([rel[i,0], rel[j,0]], [rel[i,1], rel[j,1]],
                    color=color, linewidth=2, alpha=0.7)
    for i in range(17):
        if scores[i] > 0.1:
            ax.scatter(rel[i,0], rel[i,1], s=40, color=color,
                       zorder=5, edgecolors='k', linewidths=0.5)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1.1, -0.1)  # y-axis flipped (image space)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.2)

n_samples = 4
np.random.seed(42)
fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 7))

for row_idx, (cls_dir, cls_label, color) in enumerate([
    (SITTING_DIR,  'Sitting',  '#2196F3'),
    (STANDING_DIR, 'Standing', '#FF5722'),
]):
    files = [f for f in os.listdir(cls_dir) if f.endswith('.json')]
    chosen = np.random.choice(files, n_samples, replace=False)
    for col_idx, fname in enumerate(chosen):
        with open(os.path.join(cls_dir, fname)) as f:
            d = json.load(f)
        draw_skeleton(
            axes[row_idx, col_idx],
            d['keypoints'], d['keypoint_scores'],
            color, f"{cls_label}\n{fname[:12]}",
            d['bbox']
        )

fig.suptitle('Skeleton Visualizations (Normalized to BBox)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'skeleton_samples.png'), dpi=120)
plt.close()
print("Saved: skeleton_samples.png")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 8: Correlation heatmap of key features
# ═══════════════════════════════════════════════════════════════════════════════
corr_feats = [
    'label_int', 'bbox_aspect_ratio',
    'angle_left_hip_knee_ankle', 'angle_right_hip_knee_ankle',
    'angle_left_shoulder_hip_knee', 'angle_right_shoulder_hip_knee',
    'left_knee_ratio', 'right_knee_ratio',
    'left_hip_ankle_span', 'right_hip_ankle_span',
    'mean_conf_lower',
]
corr = df[corr_feats].corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_feats)))
ax.set_yticks(range(len(corr_feats)))
short_names = ['label', 'aspect_ratio', 'L_hka', 'R_hka', 'L_shk', 'R_shk',
               'L_knee_ratio', 'R_knee_ratio', 'L_ha_span', 'R_ha_span', 'conf_lower']
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(short_names, fontsize=9)
for i in range(len(corr_feats)):
    for j in range(len(corr_feats)):
        ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=7,
                color='black' if abs(corr.values[i,j]) < 0.5 else 'white')
plt.colorbar(im, ax=ax)
ax.set_title('Feature Correlation Heatmap (label_int=1 means standing)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'correlation_heatmap.png'), dpi=120)
plt.close()
print("Saved: correlation_heatmap.png")

print("\n=== ALL VISUALIZATIONS COMPLETE ===")
print(f"Output directory: {OUT_DIR}")
EOF
