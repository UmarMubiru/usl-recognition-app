"""
EDA – Dataset 2: DATASET_ug_sign_language
==========================================
Structure:
  • videos/           – 90 mp4 videos (0001.mp4 … 0090.mp4)
  • sign_annotations.csv – video_ID, sign_word
  • Keypoints/        – 90 _keypoints.npy files  (one per video)
  • features/         – 90 _keypoints.npy + 90 .mp4.npy files

Outputs saved to:  outputs/eda_dataset2/
"""

from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DS2_ROOT = ROOT / "DATASET_ug_sign_language"
VIDEOS_DIR = DS2_ROOT / "videos"
KP_DIR = DS2_ROOT / "Keypoints"
FEAT_DIR = DS2_ROOT / "features"
CSV_PATH = DS2_ROOT / "sign_annotations.csv"
OUT_DIR = ROOT / "outputs" / "eda_dataset2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(name: str, dpi: int = 150) -> None:
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=dpi, bbox_inches="tight")
    plt.close()


def video_meta(path: Path) -> Dict[str, float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0
    cap.release()
    duration = frames / fps if fps > 0 else 0.0
    return {"fps": fps, "frame_count": frames, "width": w, "height": h,
            "duration_sec": duration}


def safe_load_npy(path: Path) -> np.ndarray:
    try:
        return np.load(str(path), allow_pickle=True)
    except Exception:
        return np.array([])


def pool_sequence(arr: np.ndarray) -> np.ndarray:
    """Flatten temporal sequence to a fixed-length feature vector via mean/std/min/max pooling."""
    if arr.ndim == 0 or arr.size == 0:
        return np.zeros(4)
    flat = arr.reshape(arr.shape[0], -1).astype(float) if arr.ndim >= 2 else arr.reshape(1, -1).astype(float)
    return np.concatenate([flat.mean(0), flat.std(0), flat.min(0), flat.max(0)])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load annotations and video metadata
# ─────────────────────────────────────────────────────────────────────────────
print("Loading annotations …")
ann = pd.read_csv(CSV_PATH)
ann.columns = ann.columns.str.strip()
ann["sign_word"] = ann["sign_word"].str.strip()
print(f"  Annotation rows : {len(ann)}")
print(f"  Unique sign words: {ann['sign_word'].nunique()}")
print(f"  Sample:\n{ann.head(5).to_string(index=False)}")

print("\nExtracting video metadata …")
meta_rows: List[Dict] = []
for _, row in ann.iterrows():
    vid_path = VIDEOS_DIR / row["video_ID"]
    meta = video_meta(vid_path) if vid_path.exists() else {"fps": 0, "frame_count": 0, "width": 0, "height": 0, "duration_sec": 0}
    meta_rows.append({"video_ID": row["video_ID"], "sign_word": row["sign_word"], **meta})

df = pd.DataFrame(meta_rows)
df.to_csv(OUT_DIR / "dataset2_inventory.csv", index=False)
print(f"  Saved → dataset2_inventory.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Summary statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Summary Statistics ─────────────────────────────────")
print(df[["fps", "frame_count", "duration_sec", "width", "height"]].describe().round(2).to_string())

word_summary = df.groupby("sign_word").agg(
    video_count=("video_ID", "count"),
    mean_duration=("duration_sec", "mean"),
    mean_fps=("fps", "mean"),
    mean_frames=("frame_count", "mean"),
).round(2).reset_index()
word_summary.to_csv(OUT_DIR / "dataset2_word_summary.csv", index=False)
print("\nTop 10 words by video count:")
print(word_summary.sort_values("video_count", ascending=False).head(10).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plot 1 – Videos per sign word (full bar chart)
# ─────────────────────────────────────────────────────────────────────────────
word_counts = df["sign_word"].value_counts().sort_values(ascending=False)
n_words = len(word_counts)

fig, ax = plt.subplots(figsize=(max(14, n_words * 0.25), 6))
bars = ax.bar(range(n_words), word_counts.values,
              color=sns.color_palette("tab20", n_colors=n_words))
ax.set_xticks(range(n_words))
ax.set_xticklabels(word_counts.index, rotation=90, fontsize=7)
ax.set_title("Dataset 2 – Videos per Sign Word", fontsize=13, fontweight="bold")
ax.set_xlabel("Sign Word")
ax.set_ylabel("Number of Videos")
save_fig("01_videos_per_sign_word.png")
print("\nSaved → 01_videos_per_sign_word.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Plot 2 – Word frequency distribution (histogram of counts per word)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(word_counts.values, bins=range(1, word_counts.max() + 2),
             color="#4C72B0", ax=ax, discrete=True)
ax.set_title("Dataset 2 – Histogram of Videos-per-Word Counts", fontsize=13, fontweight="bold")
ax.set_xlabel("Videos per Sign Word")
ax.set_ylabel("Number of Sign Words")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
save_fig("02_word_frequency_histogram.png")
print("Saved → 02_word_frequency_histogram.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plot 3 – Duration distribution
# ─────────────────────────────────────────────────────────────────────────────
valid_dur = df[df["duration_sec"] > 0]["duration_sec"]

fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(valid_dur, bins=25, kde=True, color="#DD8452", ax=ax)
ax.axvline(valid_dur.mean(), color="darkblue", linestyle="--",
           label=f"Mean {valid_dur.mean():.2f}s")
ax.axvline(valid_dur.median(), color="green", linestyle="--",
           label=f"Median {valid_dur.median():.2f}s")
ax.set_title("Dataset 2 – Video Duration Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Duration (seconds)")
ax.set_ylabel("Count")
ax.legend()
save_fig("03_duration_distribution.png")
print("Saved → 03_duration_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plot 4 – FPS distribution
# ─────────────────────────────────────────────────────────────────────────────
valid_fps = df[df["fps"] > 0]["fps"].round(2)
fps_counts = valid_fps.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(fps_counts.index.astype(str), fps_counts.values, color="#55A868")
ax.set_title("Dataset 2 – FPS Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Frames per Second")
ax.set_ylabel("Number of Videos")
ax.tick_params(axis="x", rotation=45)
for x, v in zip(fps_counts.index, fps_counts.values):
    ax.text(str(x), v + 0.1, str(v), ha="center", fontsize=8)
save_fig("04_fps_distribution.png")
print("Saved → 04_fps_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Plot 5 – Frame count distribution
# ─────────────────────────────────────────────────────────────────────────────
valid_fc = df[df["frame_count"] > 0]["frame_count"]

fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(valid_fc, bins=25, kde=True, color="#C44E52", ax=ax)
ax.axvline(valid_fc.mean(), color="navy", linestyle="--",
           label=f"Mean {valid_fc.mean():.0f} frames")
ax.set_title("Dataset 2 – Frame Count Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Total Frames")
ax.set_ylabel("Count")
ax.legend()
save_fig("05_frame_count_distribution.png")
print("Saved → 05_frame_count_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Keypoints analysis – load from Keypoints/ folder
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading keypoint files …")
kp_rows: List[Dict] = []

for kp_file in sorted(KP_DIR.glob("*_keypoints.npy")):
    vid_id_stem = kp_file.stem.replace("_keypoints", "")
    arr = safe_load_npy(kp_file)
    row_kp = {
        "video_stem": vid_id_stem,
        "kp_shape": str(arr.shape),
        "kp_ndim": arr.ndim,
        "kp_total_values": int(arr.size),
        "kp_frames": int(arr.shape[0]) if arr.ndim >= 1 and arr.size > 0 else 0,
        "kp_features_per_frame": int(np.prod(arr.shape[1:])) if arr.ndim >= 2 and arr.size > 0 else 0,
        "kp_mean": float(arr.mean()) if arr.size > 0 else 0.0,
        "kp_std": float(arr.std()) if arr.size > 0 else 0.0,
        "kp_min": float(arr.min()) if arr.size > 0 else 0.0,
        "kp_max": float(arr.max()) if arr.size > 0 else 0.0,
    }
    kp_rows.append(row_kp)

kp_df = pd.DataFrame(kp_rows)

# Merge with annotations (match video_stem to video_ID stem)
ann_copy = ann.copy()
ann_copy["video_stem"] = ann_copy["video_ID"].str.replace(".mp4", "", regex=False)
kp_df = kp_df.merge(ann_copy[["video_stem", "sign_word"]], on="video_stem", how="left")
kp_df.to_csv(OUT_DIR / "dataset2_keypoints_stats.csv", index=False)
print(f"  Loaded {len(kp_df)} keypoint files")
print(f"  Shapes found: {kp_df['kp_shape'].value_counts().to_dict()}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Plot 6 – Keypoints: frames per video (histogram)
# ─────────────────────────────────────────────────────────────────────────────
valid_kp = kp_df[kp_df["kp_frames"] > 0]

if not valid_kp.empty:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(valid_kp["kp_frames"], bins=20, kde=True, color="#8172B2", ax=ax)
    ax.axvline(valid_kp["kp_frames"].mean(), color="red", linestyle="--",
               label=f"Mean {valid_kp['kp_frames'].mean():.0f}")
    ax.set_title("Dataset 2 – Keypoint Sequence Length (frames per video)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Frames in Keypoint Array")
    ax.set_ylabel("Count")
    ax.legend()
    save_fig("06_keypoint_frames_histogram.png")
    print("Saved → 06_keypoint_frames_histogram.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Plot 7 – Keypoints: mean value distribution per video (value range check)
# ─────────────────────────────────────────────────────────────────────────────
if not valid_kp.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(valid_kp["kp_mean"], bins=20, kde=True, color="#4878CF", ax=axes[0])
    axes[0].set_title("Mean Keypoint Value per Video")
    axes[0].set_xlabel("Mean Value")

    sns.histplot(valid_kp["kp_std"], bins=20, kde=True, color="#6ACC65", ax=axes[1])
    axes[1].set_title("Std of Keypoint Values per Video")
    axes[1].set_xlabel("Std Value")

    plt.suptitle("Dataset 2 – Keypoint Feature Statistics", fontsize=13, fontweight="bold")
    save_fig("07_keypoint_value_stats.png")
    print("Saved → 07_keypoint_value_stats.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Feature (.mp4.npy) analysis – load from features/ folder
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading .mp4.npy feature files …")
feat_rows: List[Dict] = []
feat_vectors: List[np.ndarray] = []
feat_labels: List[str] = []

for feat_file in sorted(FEAT_DIR.glob("*.mp4.npy")):
    vid_id = feat_file.name.replace(".npy", "")   # e.g. "0001.mp4"
    arr = safe_load_npy(feat_file)

    # Build flat feature vector via temporal pooling for PCA
    flat = None
    if arr.ndim >= 2 and arr.size > 0:
        t = arr.shape[0]
        reshaped = arr.reshape(t, -1).astype(float)
        flat = np.concatenate([reshaped.mean(0), reshaped.std(0)])
    elif arr.ndim == 1 and arr.size > 0:
        flat = arr.astype(float)

    f_row = {
        "video_ID": vid_id,
        "feat_shape": str(arr.shape),
        "feat_ndim": arr.ndim,
        "feat_size": int(arr.size),
        "feat_mean": float(arr.mean()) if arr.size > 0 else 0.0,
        "feat_std": float(arr.std()) if arr.size > 0 else 0.0,
        "feat_norm": float(np.linalg.norm(arr.ravel())) if arr.size > 0 else 0.0,
    }
    feat_rows.append(f_row)

    if flat is not None:
        feat_vectors.append(flat)
        matched = ann[ann["video_ID"] == vid_id]["sign_word"]
        feat_labels.append(matched.values[0] if len(matched) > 0 else "unknown")

feat_df = pd.DataFrame(feat_rows)
feat_df = feat_df.merge(ann.rename(columns={"video_ID": "video_ID"}), on="video_ID", how="left")
feat_df.to_csv(OUT_DIR / "dataset2_feature_stats.csv", index=False)
print(f"  Loaded {len(feat_df)} feature files")
print(f"  Feature shapes: {feat_df['feat_shape'].value_counts().to_dict()}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Plot 8 – Feature norm distribution per video
# ─────────────────────────────────────────────────────────────────────────────
valid_feat = feat_df[feat_df["feat_norm"] > 0]

if not valid_feat.empty:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(valid_feat["feat_norm"], bins=25, kde=True, color="#C44E52", ax=ax)
    ax.set_title("Dataset 2 – Feature Vector L2-Norm Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Count")
    save_fig("08_feature_norm_distribution.png")
    print("Saved → 08_feature_norm_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 13. Plot 9 – PCA of pooled feature vectors (coloured by sign word)
# ─────────────────────────────────────────────────────────────────────────────
if len(feat_vectors) >= 5:
    print("\nRunning PCA on feature vectors …")
    # Pad/truncate to the same length
    min_len = min(v.shape[0] for v in feat_vectors)
    X = np.vstack([v[:min_len] for v in feat_vectors])

    # Remove zero-variance columns
    var_mask = X.std(axis=0) > 0
    X = X[:, var_mask]

    if X.shape[1] >= 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=min(2, X_scaled.shape[1]), random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        le = LabelEncoder()
        y = le.fit_transform(feat_labels)
        n_classes = len(le.classes_)

        fig, ax = plt.subplots(figsize=(10, 8))
        palette = sns.color_palette("tab20", n_colors=n_classes)
        for idx, label in enumerate(le.classes_):
            mask = y == idx
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1] if X_pca.shape[1] > 1 else np.zeros(mask.sum()),
                       label=label, color=palette[idx], s=60, alpha=0.8)
        ax.set_title(f"Dataset 2 – PCA of Feature Vectors\n"
                     f"(Explained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%"
                     + (f", PC2={pca.explained_variance_ratio_[1]*100:.1f}%)" if pca.n_components_ > 1 else ")"),
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2" if pca.n_components_ > 1 else "")
        if n_classes <= 30:
            ax.legend(fontsize=6, ncol=3, loc="best")
        save_fig("09_pca_feature_vectors.png")
        print("Saved → 09_pca_feature_vectors.png")


# ─────────────────────────────────────────────────────────────────────────────
# 14. Plot 10 – Correlation heatmap of numeric metadata
# ─────────────────────────────────────────────────────────────────────────────
numeric_cols = ["fps", "frame_count", "duration_sec", "width", "height"]
corr = df[numeric_cols].dropna().corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            linewidths=0.5, vmin=-1, vmax=1)
ax.set_title("Dataset 2 – Correlation Matrix (Video Metadata)", fontsize=13, fontweight="bold")
save_fig("10_metadata_correlation_heatmap.png")
print("Saved → 10_metadata_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 15. Plot 11 – Box plot: duration grouped by sign word (top 20 most frequent)
# ─────────────────────────────────────────────────────────────────────────────
top20 = word_counts.head(20).index.tolist()
df_top = df[df["sign_word"].isin(top20) & (df["duration_sec"] > 0)]

if not df_top.empty:
    fig, ax = plt.subplots(figsize=(14, 6))
    order = df_top.groupby("sign_word")["duration_sec"].median().sort_values(ascending=False).index
    sns.boxplot(data=df_top, x="sign_word", y="duration_sec", order=order,
                hue="sign_word", palette="tab20", ax=ax, linewidth=0.8, legend=False)
    ax.set_title("Dataset 2 – Duration per Sign Word (Top 20 by count)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sign Word")
    ax.set_ylabel("Duration (seconds)")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    save_fig("11_duration_per_sign_word_boxplot.png")
    print("Saved → 11_duration_per_sign_word_boxplot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 16. Plot 12 – Keypoint feature mean per sign word (joint kp+annotation df)
# ─────────────────────────────────────────────────────────────────────────────
kp_valid = kp_df[kp_df["sign_word"].notna() & (kp_df["kp_frames"] > 0)]

if not kp_valid.empty:
    kp_word_stats = kp_valid.groupby("sign_word")[["kp_frames", "kp_mean", "kp_std"]].mean().round(3)
    kp_word_stats = kp_word_stats.sort_values("kp_frames", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].barh(kp_word_stats.index[:20], kp_word_stats["kp_frames"][:20],
                 color="#8172B2")
    axes[0].set_title("Mean Keypoint Frames by Sign Word (top 20)")
    axes[0].set_xlabel("Mean Frames")
    axes[0].invert_yaxis()

    axes[1].barh(kp_word_stats.index[:20], kp_word_stats["kp_mean"][:20],
                 color="#4878CF")
    axes[1].set_title("Mean Keypoint Value by Sign Word (top 20)")
    axes[1].set_xlabel("Mean Keypoint Value")
    axes[1].invert_yaxis()

    plt.suptitle("Dataset 2 – Keypoint Stats per Sign Word", fontsize=13, fontweight="bold")
    save_fig("12_keypoints_per_sign_word.png")
    print("Saved → 12_keypoints_per_sign_word.png")


# ─────────────────────────────────────────────────────────────────────────────
# 17. Additional unsupervised analysis on DS2 (metadata + keypoint stats)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Unsupervised Learning Analysis (DS2) ─────────────────")

unsup_df = kp_df.merge(
    ann_copy[["video_stem", "video_ID", "sign_word"]],
    on=["video_stem", "sign_word"],
    how="left",
)
unsup_df = unsup_df.merge(
    df[["video_ID", "fps", "frame_count", "duration_sec", "width", "height"]],
    on="video_ID",
    how="left",
)

unsup_cols = [
    "fps", "frame_count", "duration_sec", "width", "height",
    "kp_frames", "kp_features_per_frame", "kp_mean", "kp_std", "kp_min", "kp_max",
]
x_unsup = unsup_df[unsup_cols].fillna(0).astype(float).values
x_unsup = StandardScaler().fit_transform(x_unsup)

ks = list(range(2, 11))
inertias = []
sils = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(x_unsup)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(x_unsup, labels))

best_k = ks[int(np.argmax(sils))]
best_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
best_cluster = best_model.fit_predict(x_unsup)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ks, inertias, marker="o")
ax.set_title("Dataset 2 – Unsupervised KMeans Elbow Curve", fontsize=13, fontweight="bold")
ax.set_xlabel("k")
ax.set_ylabel("Inertia")
save_fig("13_unsup_kmeans_elbow.png")
print("Saved → 13_unsup_kmeans_elbow.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ks, sils, marker="o", color="#dd8452")
ax.set_title("Dataset 2 – Unsupervised Silhouette by k", fontsize=13, fontweight="bold")
ax.set_xlabel("k")
ax.set_ylabel("Silhouette Score")
save_fig("14_unsup_silhouette_by_k.png")
print("Saved → 14_unsup_silhouette_by_k.png")

pca_unsup = PCA(n_components=2, random_state=42)
xy = pca_unsup.fit_transform(x_unsup)
fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=best_cluster, palette="tab10", s=55, ax=ax)
ax.set_title(f"Dataset 2 – Unsupervised PCA + KMeans (k={best_k})", fontsize=13, fontweight="bold")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
save_fig("15_unsup_pca_kmeans_scatter.png")
print("Saved → 15_unsup_pca_kmeans_scatter.png")

cluster_word = pd.crosstab(pd.Series(best_cluster, name="cluster"), unsup_df["sign_word"].fillna("unknown"))
cluster_word = cluster_word.loc[:, cluster_word.sum(axis=0).sort_values(ascending=False).index[:20]]
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(cluster_word, cmap="magma", annot=True, fmt="d", ax=ax)
ax.set_title("Dataset 2 – Cluster vs Sign Word (Top 20)", fontsize=13, fontweight="bold")
save_fig("16_unsup_cluster_word_heatmap.png")
print("Saved → 16_unsup_cluster_word_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 17. Class balance check
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Class Balance Analysis ──────────────────────────────")
print(f"  Total videos     : {len(df)}")
print(f"  Unique sign words: {df['sign_word'].nunique()}")
max_c = word_counts.max()
min_c = word_counts.min()
print(f"  Most-represented  : {word_counts.idxmax()} ({max_c} videos)")
print(f"  Least-represented : {word_counts.idxmin()} ({min_c} videos)")
print(f"  Imbalance ratio   : {max_c / min_c:.2f}x")
print(f"  Words with only 1 video: {(word_counts == 1).sum()}")


print("\n✓  Dataset 2 EDA complete.  All figures saved to:", OUT_DIR)
