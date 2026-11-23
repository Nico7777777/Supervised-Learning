"""Generate publication-ready figures for report.tex based on actual experiment data."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid", font="sans-serif")

df = pd.read_csv("features.csv")
train_df = df[df["split"] == "train"].copy()

# -------------------------------------------------------------
# 1. Class Distribution
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.8))
counts = df["label"].value_counts()
colors = sns.color_palette("mako", len(counts))
bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="#333333", alpha=0.9)
ax.set_xlabel("Total Sample Count", fontweight="bold", fontsize=11)
ax.set_title("Dataset Instance Distribution Across 12 Domestic Sound Classes (N = 1537)", fontweight="bold", fontsize=12)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 2, bar.get_y() + bar.get_height() / 2, f"{int(w)} ({w / len(df) * 100:.1f}%)",
            va="center", fontsize=9, fontweight="bold", color="#222222")
ax.set_xlim(0, 180)
ax.invert_yaxis()
plt.tight_layout()
fig.savefig("figures/class_distribution.png", dpi=180)
plt.close(fig)
print("[FIG] figures/class_distribution.png generated.")

# -------------------------------------------------------------
# 2. Feature Boxplots (4-panel: Centroid, SNR, Active Fraction, ZCR)
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
features_to_plot = [
    ("spectral_centroid", "Spectral Centroid (Hz)", "Spectral Centroid Across Classes", axes[0, 0]),
    ("active_frame_fraction", "Active Frame Fraction (0 - 1)", "Active Frame Fraction (Temporal Continuity)", axes[0, 1]),
    ("approx_snr_db", "Approximate SNR (dB)", "Approximate SNR (P95 - P10 Level)", axes[1, 0]),
    ("zero_crossing_rate", "Zero-Crossing Rate", "Zero-Crossing Rate (ZCR)", axes[1, 1]),
]

for col_name, xlabel, title, ax in features_to_plot:
    order = train_df.groupby("label")[col_name].median().sort_values(ascending=False).index
    sns.boxplot(
        data=train_df,
        y="label",
        x=col_name,
        order=order,
        ax=ax,
        palette="crest",
        fliersize=2,
        linewidth=1.2,
    )
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel(xlabel, fontweight="bold", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(labelsize=9)

plt.tight_layout()
fig.savefig("figures/feature_boxplots.png", dpi=180)
plt.close(fig)
print("[FIG] figures/feature_boxplots.png generated.")

# -------------------------------------------------------------
# 3. Pearson Correlation Heatmap
# -------------------------------------------------------------
scalar_cols = [
    "rms_dbfs", "crest_factor_db", "noise_floor_dbfs_p10", "signal_level_dbfs_p95",
    "approx_snr_db", "active_frame_fraction", "spectral_centroid",
    "spectral_bandwidth", "spectral_rolloff_85", "spectral_flatness", "zero_crossing_rate"
]
corr = train_df[scalar_cols].corr(method="pearson")

fig, ax = plt.subplots(figsize=(9.5, 7.5))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdYlBu_r",
    vmin=-1.0,
    vmax=1.0,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
    ax=ax,
    annot_kws={"size": 8.5, "fontweight": "bold"},
)
ax.set_title("Pearson Correlation Heatmap of Scalar Acoustic Features (Train Split)", fontweight="bold", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
fig.savefig("figures/correlation_matrix.png", dpi=180)
plt.close(fig)
print("[FIG] figures/correlation_matrix.png generated.")

# -------------------------------------------------------------
# 4. 2D PCA Projection of Standardized MFCC Features
# -------------------------------------------------------------
mfcc_cols = [f"mfcc_mean_{i}" for i in range(1, 21)] + [f"mfcc_std_{i}" for i in range(1, 21)]
scaler = StandardScaler()
mfcc_scaled = scaler.fit_transform(train_df[mfcc_cols])

pca = PCA(n_components=2, random_state=42)
mfcc_pca = pca.fit_transform(mfcc_scaled)
var1, var2 = pca.explained_variance_ratio_ * 100

train_df["pca_1"] = mfcc_pca[:, 0]
train_df["pca_2"] = mfcc_pca[:, 1]

fig, ax = plt.subplots(figsize=(10, 7.5))
palette = sns.color_palette("tab20", n_colors=train_df["label"].nunique())
sns.scatterplot(
    data=train_df,
    x="pca_1",
    y="pca_2",
    hue="label",
    palette=palette,
    style="label",
    s=55,
    alpha=0.85,
    ax=ax,
    edgecolor="w",
    linewidth=0.4,
)
ax.set_title(f"2D PCA Projection of Standardized MFCC Features (40 Dimensions)\nPC1 ({var1:.1f}%) and PC2 ({var2:.1f}%) --- Total Explained Variance: {var1 + var2:.1f}%",
             fontweight="bold", fontsize=12)
ax.set_xlabel(f"Principal Component 1 ({var1:.1f}% Variance)", fontweight="bold", fontsize=11)
ax.set_ylabel(f"Principal Component 2 ({var2:.1f}% Variance)", fontweight="bold", fontsize=11)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, title="Sound Class", fontsize=9, title_fontsize=10)
plt.tight_layout()
fig.savefig("figures/pca_projection.png", dpi=180)
plt.close(fig)
print("[FIG] figures/pca_projection.png generated.")

# -------------------------------------------------------------
# 5. Model Comparison Bar Chart
# -------------------------------------------------------------
# Actual experimental values from training run
models = ["SVM (RBF)", "Gradient Boosting", "Random Forest"]
group_cv_acc = [63.96, 64.20, 58.83]
val_acc = [61.69, 60.39, 51.95]
test_acc = [61.69, 59.74, 54.55]
test_f1 = [56.66, 54.11, 50.57]

x = np.arange(len(models))
width = 0.20

fig, ax = plt.subplots(figsize=(9, 5.5))
r1 = ax.bar(x - 1.5 * width, group_cv_acc, width, label="Group CV (Train)", color="#2b5c8f", edgecolor="black")
r2 = ax.bar(x - 0.5 * width, val_acc, width, label="Validation Split", color="#419d78", edgecolor="black")
r3 = ax.bar(x + 0.5 * width, test_acc, width, label="Test Split (Accuracy)", color="#e0a96d", edgecolor="black")
r4 = ax.bar(x + 1.5 * width, test_f1, width, label="Test Split (Macro-F1)", color="#c74b4b", edgecolor="black")

ax.set_ylabel("Score (%)", fontweight="bold", fontsize=11)
ax.set_title("Quantitative Performance Comparison Across Evaluated Model Architectures", fontweight="bold", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight="bold", fontsize=11)
ax.set_ylim(40, 75)
ax.legend(loc="upper right", fontsize=9.5)

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

autolabel(r1)
autolabel(r2)
autolabel(r3)
autolabel(r4)

plt.tight_layout()
fig.savefig("figures/model_comparison.png", dpi=180)
plt.close(fig)
print("[FIG] figures/model_comparison.png generated.")

# -------------------------------------------------------------
# 6. Copy / Format Confusion Matrix for Best Model (SVM)
# -------------------------------------------------------------
import shutil
svm_cm_src = "eval/SVM/SVM_C_10_gamma_scale_kernel_rbf/confusion_matrix_normalized.png"
if os.path.exists(svm_cm_src):
    shutil.copy(svm_cm_src, "figures/confusion_matrix_svm.png")
    print("[FIG] Copied figures/confusion_matrix_svm.png from eval/.")
else:
    print("[WARN] SVM normalized confusion matrix not found at:", svm_cm_src)

# -------------------------------------------------------------
# 7. Multi-Model Confusion Matrix Comparison (3 Models side-by-side)
# -------------------------------------------------------------
cm_paths = [
    ("SVM (Optimal: C=10, rbf)", "eval/SVM/SVM_C_10_gamma_scale_kernel_rbf/confusion_matrix_normalized.png"),
    ("Gradient Boosting (lr=0.1, iter=80)", "eval/Gradient Boosting/GB_learning_rate_0.1_max_iter_80_max_leaf_nodes_15/confusion_matrix_normalized.png"),
    ("Random Forest (depth=15, entropy)", "eval/RandomForest/RF_criterion_entropy_max_depth_15_max_features_log2/confusion_matrix_normalized.png"),
]

fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
for ax, (title, p) in zip(axes, cm_paths):
    if os.path.exists(p):
        img = plt.imread(p)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontweight="bold", fontsize=12)

plt.tight_layout()
fig.savefig("figures/confusion_matrix_all.png", dpi=180)
plt.close(fig)
print("[FIG] figures/confusion_matrix_all.png generated.")
