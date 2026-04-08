"""
EDA – Dataset 1: SIGN LANGUAGE DISEASES FINISHED
================================================
Covers ALL 27 top-level categories:
  • disease folders    (each with 'diagnostic prompts' / 'screening prompts' sub-dirs)
  • ALPHABETS AND NUMBERS (sub-dirs: ALPHABET, NUMBERS)
  • UNIQUE WORDS           (videos directly inside the folder)

Outputs saved to:  outputs/eda_dataset1/
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DS1_ROOT = ROOT / "SIGN LANGUAGE DISEASES FINISHED"
OUT_DIR = ROOT / "outputs" / "eda_dataset1"
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


# ─────────────────────────────────────────────────────────────────────────────
# 1. Collect inventory
# ─────────────────────────────────────────────────────────────────────────────
print("Collecting video inventory from Dataset 1 …")
rows: List[Dict] = []

for cat_dir in sorted(DS1_ROOT.iterdir()):
    if not cat_dir.is_dir():
        continue
    category = cat_dir.name.strip()

    # Find all mp4 files recursively; record the immediate sub-folder name
    for mp4 in sorted(cat_dir.rglob("*.mp4")):
        # sub_category = direct parent relative to category root
        rel_parts = mp4.relative_to(cat_dir).parts
        sub_cat = rel_parts[0] if len(rel_parts) > 1 else "direct"

        meta = video_meta(mp4)
        rows.append({
            "category": category,
            "sub_category": sub_cat,
            "video_name": mp4.name,
            "video_path": str(mp4),
            **meta,
        })

df = pd.DataFrame(rows)
df["is_disease"] = (~df["category"].isin(["ALPHABETS AND NUMBERS", "UNIQUE WORDS"])).astype(int)
print(f"  Total videos : {len(df)}")
print(f"  Categories   : {df['category'].nunique()}")
print(f"  Sub-categories (unique names): {df['sub_category'].nunique()}")

# Save master CSV
df.to_csv(OUT_DIR / "dataset1_inventory.csv", index=False)
print("  Saved → dataset1_inventory.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Summary statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Summary Statistics ─────────────────────────────────")
print(df[["fps", "frame_count", "duration_sec", "width", "height"]].describe().round(2).to_string())

summary = df.groupby("category").agg(
    total_videos=("video_name", "count"),
    mean_duration=("duration_sec", "mean"),
    std_duration=("duration_sec", "std"),
    mean_fps=("fps", "mean"),
    mean_frames=("frame_count", "mean"),
    width=("width", "first"),
    height=("height", "first"),
).round(2).reset_index()
summary.to_csv(OUT_DIR / "dataset1_category_summary.csv", index=False)
print("\nCategory summary:")
print(summary.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plot 1 – Videos per category (sorted bar chart)
# ─────────────────────────────────────────────────────────────────────────────
cat_counts = df["category"].value_counts().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(cat_counts)), cat_counts.values,
              color=sns.color_palette("tab20", n_colors=len(cat_counts)))
ax.set_title("Dataset 1 – Videos per Category (all 27 folders)", fontsize=13, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Number of Videos")
ax.set_xticks(range(len(cat_counts)))
ax.set_xticklabels(cat_counts.index, rotation=45, ha="right", fontsize=8)
for bar, val in zip(bars, cat_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(val), ha="center", va="bottom", fontsize=7)
save_fig("01_videos_per_category.png")
print("\nSaved → 01_videos_per_category.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Plot 2 – Sub-category (prompt type) distribution – pie chart
# ─────────────────────────────────────────────────────────────────────────────
sub_counts = df["sub_category"].value_counts()

fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    sub_counts.values,
    labels=sub_counts.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=sns.color_palette("pastel", n_colors=len(sub_counts)),
)
ax.set_title("Dataset 1 – Sub-Category Distribution\n(all videos)", fontsize=13, fontweight="bold")
save_fig("02_subcategory_pie.png")
print("Saved → 02_subcategory_pie.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plot 3 – Stacked bar: diagnostic vs screening per disease
#    (only categories that have those two sub-types)
# ─────────────────────────────────────────────────────────────────────────────
diag_screen = df[df["sub_category"].str.lower().isin(
    ["diagnostic prompts", "screening prompts"]
)].copy()

if not diag_screen.empty:
    pivot = (
        diag_screen.groupby(["category", "sub_category"])
        .size()
        .unstack(fill_value=0)
    )
    pivot.plot(
        kind="bar", stacked=True, figsize=(14, 6),
        color=["#4C72B0", "#DD8452"],
    )
    plt.title("Dataset 1 – Diagnostic vs Screening Prompts per Disease", fontsize=13, fontweight="bold")
    plt.xlabel("Disease / Category")
    plt.ylabel("Number of Videos")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.legend(title="Prompt Type")
    save_fig("03_diagnostic_vs_screening_stacked.png")
    print("Saved → 03_diagnostic_vs_screening_stacked.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plot 4 – Duration distribution (histogram + KDE)
# ─────────────────────────────────────────────────────────────────────────────
valid_dur = df[df["duration_sec"] > 0]["duration_sec"]

fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(valid_dur, bins=30, kde=True, color="#4C72B0", ax=ax)
ax.axvline(valid_dur.mean(), color="red", linestyle="--", label=f"Mean {valid_dur.mean():.1f}s")
ax.axvline(valid_dur.median(), color="orange", linestyle="--", label=f"Median {valid_dur.median():.1f}s")
ax.set_title("Dataset 1 – Video Duration Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Duration (seconds)")
ax.set_ylabel("Count")
ax.legend()
save_fig("04_duration_distribution.png")
print("Saved → 04_duration_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Plot 5 – FPS distribution
# ─────────────────────────────────────────────────────────────────────────────
valid_fps = df[df["fps"] > 0]["fps"].round(1)

fig, ax = plt.subplots(figsize=(8, 5))
fps_counts = valid_fps.value_counts().sort_index()
ax.bar(fps_counts.index.astype(str), fps_counts.values, color="#55A868")
ax.set_title("Dataset 1 – FPS Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Frames per Second")
ax.set_ylabel("Number of Videos")
ax.tick_params(axis="x", rotation=45)
for x, v in zip(fps_counts.index, fps_counts.values):
    ax.text(str(x), v + 0.2, str(v), ha="center", fontsize=8)
save_fig("05_fps_distribution.png")
print("Saved → 05_fps_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Plot 6 – Frame count distribution
# ─────────────────────────────────────────────────────────────────────────────
valid_fc = df[df["frame_count"] > 0]["frame_count"]

fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(valid_fc, bins=30, kde=True, color="#C44E52", ax=ax)
ax.axvline(valid_fc.mean(), color="navy", linestyle="--", label=f"Mean {valid_fc.mean():.0f} frames")
ax.set_title("Dataset 1 – Frame Count Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Total Frames")
ax.set_ylabel("Count")
ax.legend()
save_fig("06_frame_count_distribution.png")
print("Saved → 06_frame_count_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Plot 7 – Box plot: duration per category
# ─────────────────────────────────────────────────────────────────────────────
valid_df = df[df["duration_sec"] > 0].copy()

fig, ax = plt.subplots(figsize=(16, 6))
order = valid_df.groupby("category")["duration_sec"].median().sort_values(ascending=False).index
sns.boxplot(data=valid_df, x="category", y="duration_sec", order=order, ax=ax,
            hue="category", palette="tab20", linewidth=0.8, legend=False)
ax.set_title("Dataset 1 – Duration per Category (sorted by median)", fontsize=13, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Duration (seconds)")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
save_fig("07_duration_per_category_boxplot.png")
print("Saved → 07_duration_per_category_boxplot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Plot 8 – Resolution scatter (width vs height)
# ─────────────────────────────────────────────────────────────────────────────
res_df = df[(df["width"] > 0) & (df["height"] > 0)].copy()

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(res_df["width"], res_df["height"],
                     c=pd.factorize(res_df["category"])[0],
                     cmap="tab20", alpha=0.6, s=40, edgecolors="none")
ax.set_title("Dataset 1 – Video Resolution (Width × Height)", fontsize=13, fontweight="bold")
ax.set_xlabel("Width (pixels)")
ax.set_ylabel("Height (pixels)")
# Annotate most common resolutions
for (w, h), grp in res_df.groupby(["width", "height"]):
    if len(grp) >= 3:
        ax.annotate(f"{int(w)}×{int(h)} ({len(grp)})",
                    xy=(w, h), fontsize=7, ha="center",
                    xytext=(0, 8), textcoords="offset points")
save_fig("08_resolution_scatter.png")
print("Saved → 08_resolution_scatter.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Plot 9 – Mean duration per category (bar)
# ─────────────────────────────────────────────────────────────────────────────
mean_dur = valid_df.groupby("category")["duration_sec"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(mean_dur)), mean_dur.values,
              color=sns.color_palette("coolwarm", n_colors=len(mean_dur)))
ax.set_title("Dataset 1 – Mean Video Duration per Category", fontsize=13, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Mean Duration (seconds)")
ax.set_xticks(range(len(mean_dur)))
ax.set_xticklabels(mean_dur.index, rotation=45, ha="right", fontsize=8)
for bar, val in zip(bars, mean_dur.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val:.1f}s", ha="center", va="bottom", fontsize=7)
save_fig("09_mean_duration_per_category.png")
print("Saved → 09_mean_duration_per_category.png")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Plot 10 – Correlation heatmap of video-level numeric features
# ─────────────────────────────────────────────────────────────────────────────
numeric_cols = ["fps", "frame_count", "duration_sec", "width", "height"]
corr = df[numeric_cols].dropna().corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            linewidths=0.5, vmin=-1, vmax=1)
ax.set_title("Dataset 1 – Correlation Matrix (Video Metadata)", fontsize=13, fontweight="bold")
save_fig("10_metadata_correlation_heatmap.png")
print("Saved → 10_metadata_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 13. Plot 11 – Heatmap: videos by category × sub_category
# ─────────────────────────────────────────────────────────────────────────────
heat = df.groupby(["category", "sub_category"]).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(heat, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
            linewidths=0.4, linecolor="white")
ax.set_title("Dataset 1 – Video Count: Category × Sub-Category", fontsize=13, fontweight="bold")
ax.set_xlabel("Sub-Category")
ax.set_ylabel("Category")
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0, labelsize=8)
save_fig("11_category_subcategory_heatmap.png")
print("Saved → 11_category_subcategory_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 14. Plot 12 – Duration scatter: duration vs frame_count coloured by fps
# ─────────────────────────────────────────────────────────────────────────────
plot_df = df[(df["duration_sec"] > 0) & (df["frame_count"] > 0) & (df["fps"] > 0)]

fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(plot_df["duration_sec"], plot_df["frame_count"],
                c=plot_df["fps"], cmap="viridis", alpha=0.7, s=40, edgecolors="none")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("FPS")
ax.set_title("Dataset 1 – Duration vs Frame Count (coloured by FPS)", fontsize=13, fontweight="bold")
ax.set_xlabel("Duration (seconds)")
ax.set_ylabel("Total Frame Count")
save_fig("12_duration_vs_frames_scatter.png")
print("Saved → 12_duration_vs_frames_scatter.png")


# ─────────────────────────────────────────────────────────────────────────────
# 15. Class balance check – print imbalance ratio
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Class Balance Analysis ──────────────────────────────")
max_c = cat_counts.max()
min_c = cat_counts.min()
print(f"  Most-represented  : {cat_counts.idxmax()} ({max_c} videos)")
print(f"  Least-represented : {cat_counts.idxmin()} ({min_c} videos)")
print(f"  Imbalance ratio   : {max_c / min_c:.2f}x")
print(f"  Categories with < 10 videos: "
      f"{list(cat_counts[cat_counts < 10].index)}")


# ─────────────────────────────────────────────────────────────────────────────
# 16. Learning-focused analysis on Dataset 1
#    (Unsupervised, Semi-supervised, Weak, Positive-Unlabeled)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Learning-Focused Analysis (DS1) ───────────────────")

learning_cols = ["fps", "frame_count", "duration_sec", "width", "height"]
x = df[learning_cols].fillna(0).astype(float).values
x_scaled = StandardScaler().fit_transform(x)

# Unsupervised: KMeans model selection
ks = list(range(2, 11))
inertias: List[float] = []
sil_scores: List[float] = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(x_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(x_scaled, cluster_ids))

best_k = ks[int(np.argmax(sil_scores))]
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(x_scaled)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ks, inertias, marker="o")
ax.set_title("Dataset 1 – Unsupervised KMeans Elbow Curve", fontsize=13, fontweight="bold")
ax.set_xlabel("k")
ax.set_ylabel("Inertia")
save_fig("13_unsup_kmeans_elbow.png")
print("Saved → 13_unsup_kmeans_elbow.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ks, sil_scores, marker="o", color="#dd8452")
ax.set_title("Dataset 1 – Unsupervised Silhouette by k", fontsize=13, fontweight="bold")
ax.set_xlabel("k")
ax.set_ylabel("Silhouette Score")
save_fig("14_unsup_silhouette_by_k.png")
print("Saved → 14_unsup_silhouette_by_k.png")

pca = PCA(n_components=2, random_state=42)
pca_xy = pca.fit_transform(x_scaled)
fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(x=pca_xy[:, 0], y=pca_xy[:, 1], hue=clusters, palette="tab10", s=45, ax=ax)
ax.set_title(f"Dataset 1 – Unsupervised PCA + KMeans (k={best_k})", fontsize=13, fontweight="bold")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
save_fig("15_unsup_pca_kmeans_scatter.png")
print("Saved → 15_unsup_pca_kmeans_scatter.png")

cluster_cat = pd.crosstab(pd.Series(clusters, name="cluster"), df["category"])
cluster_cat = cluster_cat.loc[:, cluster_cat.sum(axis=0).sort_values(ascending=False).index[:15]]
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(cluster_cat, cmap="YlGnBu", annot=True, fmt="d", ax=ax)
ax.set_title("Dataset 1 – Cluster vs Category (Top 15 Categories)", fontsize=13, fontweight="bold")
save_fig("16_unsup_cluster_category_heatmap.png")
print("Saved → 16_unsup_cluster_category_heatmap.png")

# Semi-supervised: LabelSpreading on DS1 categories (classes with >=10 samples)
semi_df = df[df["category"].map(df["category"].value_counts()) >= 10].copy()
sx = semi_df[learning_cols].fillna(0).astype(float).values
sy = LabelEncoder().fit_transform(semi_df["category"].astype(str).values)

sx_train, sx_test, sy_train, sy_test = train_test_split(
    sx, sy, test_size=0.25, random_state=42, stratify=sy
)
scaler = StandardScaler()
sx_train = scaler.fit_transform(sx_train)
sx_test = scaler.transform(sx_test)

fractions = [0.05, 0.1, 0.2, 0.3, 0.4]
semi_acc: List[float] = []
best_model = None
best_acc = -1.0
for i, frac in enumerate(fractions):
    rng = np.random.default_rng(42 + i)
    partial_labels = np.full(len(sy_train), -1, dtype=int)
    labeled_n = max(1, int(len(sy_train) * frac))
    labeled_idx = rng.choice(len(sy_train), size=labeled_n, replace=False)
    partial_labels[labeled_idx] = sy_train[labeled_idx]

    model = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=50)
    model.fit(sx_train, partial_labels)
    pred = model.predict(sx_test)
    acc = accuracy_score(sy_test, pred)
    semi_acc.append(acc)

    if acc > best_acc:
        best_acc = acc
        best_model = model

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([int(f * 100) for f in fractions], semi_acc, marker="o")
ax.set_title("Dataset 1 – Semi-supervised Accuracy vs Labeled %", fontsize=13, fontweight="bold")
ax.set_xlabel("Labeled portion of training set (%)")
ax.set_ylabel("Accuracy")
save_fig("17_semisup_accuracy_curve.png")
print("Saved → 17_semisup_accuracy_curve.png")

best_pred = best_model.predict(sx_test)
semi_cm = confusion_matrix(sy_test, best_pred)
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(semi_cm, cmap="Blues", ax=ax)
ax.set_title("Dataset 1 – Semi-supervised Confusion Matrix", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
save_fig("18_semisup_confusion_matrix.png")
print("Saved → 18_semisup_confusion_matrix.png")

# Weak supervision: heuristic labeling for disease vs non-disease
weak_x = df[["duration_sec", "frame_count", "fps"]].fillna(0)
true_bin = df["is_disease"].astype(int).values

votes = np.zeros(len(df), dtype=int)
votes += (weak_x["duration_sec"] > 4.0).astype(int)
votes += (weak_x["frame_count"] > 120).astype(int)
votes += (weak_x["fps"] >= 28.0).astype(int)
weak_pred = (votes >= 2).astype(int)

weak_cm = confusion_matrix(true_bin, weak_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(weak_cm, annot=True, fmt="d", cmap="OrRd", ax=ax)
ax.set_title("Dataset 1 – Weak Supervision Confusion Matrix", fontsize=13, fontweight="bold")
ax.set_xlabel("Weak Label")
ax.set_ylabel("True Label")
save_fig("19_weak_supervision_confusion.png")
print("Saved → 19_weak_supervision_confusion.png")

weak_acc = accuracy_score(true_bin, weak_pred)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(["Weak Label Accuracy"], [weak_acc], color="#dd8452")
ax.set_ylim(0, 1)
ax.set_title("Dataset 1 – Weak Supervision Accuracy", fontsize=13, fontweight="bold")
ax.text(0, weak_acc + 0.02, f"{weak_acc:.3f}", ha="center")
save_fig("20_weak_supervision_accuracy.png")
print("Saved → 20_weak_supervision_accuracy.png")

# Positive-Unlabeled: MALARIA as positive class, others unlabeled
pu_df = df.copy()
pu_df["positive"] = (pu_df["category"] == "MALARIA").astype(int)
px = pu_df[learning_cols].fillna(0).astype(float).values
py = pu_df["positive"].values

px_train, px_test, py_train, py_test = train_test_split(
    px, py, test_size=0.3, random_state=42, stratify=py
)
pu_scaler = StandardScaler()
px_train = pu_scaler.fit_transform(px_train)
px_test = pu_scaler.transform(px_test)

unlabeled = np.full_like(py_train, -1)
unlabeled[py_train == 1] = 1

stage1 = LogisticRegression(max_iter=400)
stage1.fit(px_train, (unlabeled == 1).astype(int))
prob_u = stage1.predict_proba(px_train)[:, 1]

unlabeled_mask = unlabeled == -1
if np.any(unlabeled_mask):
    # Adaptive threshold: take bottom 20% of unlabeled scores as reliable negatives.
    threshold = np.quantile(prob_u[unlabeled_mask], 0.2)
    reliable_negative = unlabeled_mask & (prob_u <= threshold)
else:
    reliable_negative = np.zeros_like(unlabeled, dtype=bool)

pu_labels = np.full_like(py_train, -1)
pu_labels[py_train == 1] = 1
pu_labels[reliable_negative] = 0
used = pu_labels != -1

# Safety fallback: if still single-class, force a small set of lowest-scored unlabeled as negatives.
if len(np.unique(pu_labels[used])) < 2 and np.any(unlabeled_mask):
    unlabeled_indices = np.where(unlabeled_mask)[0]
    sorted_unlabeled = unlabeled_indices[np.argsort(prob_u[unlabeled_indices])]
    fallback_n = max(5, int(0.05 * len(unlabeled_indices)))
    forced_neg = sorted_unlabeled[:fallback_n]
    pu_labels[forced_neg] = 0
    used = pu_labels != -1

stage2 = LogisticRegression(max_iter=400)
stage2.fit(px_train[used], pu_labels[used])
pu_score = stage2.predict_proba(px_test)[:, 1]

precision, recall, _ = precision_recall_curve(py_test, pu_score)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recall, precision, color="#4c72b0")
ax.set_title("Dataset 1 – PU Learning Precision-Recall (MALARIA vs U)", fontsize=13, fontweight="bold")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
save_fig("21_pu_precision_recall.png")
print("Saved → 21_pu_precision_recall.png")

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(pu_score[py_test == 1], color="#55a868", label="Positive", kde=True, stat="density", ax=ax)
sns.histplot(pu_score[py_test == 0], color="#c44e52", label="Unlabeled/Negative", kde=True, stat="density", ax=ax)
ax.set_title("Dataset 1 – PU Learning Score Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted positive probability")
ax.legend()
save_fig("22_pu_score_distribution.png")
print("Saved → 22_pu_score_distribution.png")

print("\n✓  Dataset 1 EDA complete.  All figures saved to:", OUT_DIR)
