"""Dataset analysis and feature extraction entry point."""

# utils
import argparse
from itertools import combinations, product
import os
import pickle
import random
import time
from typing import Any, Dict, cast

# data science
import importlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns

# EDA
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

# models
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from __init__ import *
from features import extract_dataset_features, extract_level_features
from spectrograms import *
from spectrograms import plot_continuous_distr

librosa = importlib.import_module("librosa")


def parse_args() -> argparse.Namespace:
    """Parse phase-level output controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-pie-chart-plotting",
        action="store_true",
        help="compute the pie chart but do not write its image",
    )
    parser.add_argument(
        "--skip-stats-plotting",
        action="store_true",
        help="compute metadata statistics plots but do not write their images",
    )
    parser.add_argument(
        "--skip-feature-dumping",
        action="store_true",
        help="compute all features and EDA but do not write features.csv",
    )
    return parser.parse_args()


def read_metadata(*, save_stats_plots: bool) -> dict[str, object]:
    """Read metadata, compute statistics, and optionally save their plots."""
    result: dict[str, object] = {}
    try:
        dataframe = pd.read_csv("metadata.csv")
    except Exception as error:
        result["error"] = error
        return result

    result["dataframe"] = dataframe
    os.makedirs("stats", exist_ok=True)
    feature_columns = [
        column for column in dataframe.columns
        if column not in {"label", "filename"}
    ]
    with open(
        os.path.join("stats", "PandasStatistics.txt"),
        "w",
        encoding="utf-8",
    ) as stats_file:
        stats_file.write("Metadata feature statistics\n")
        stats_file.write("===========================\n\n")
        for column in feature_columns:
            series = dataframe[column]
            stats_file.write(f"Column: {column}\n")
            stats_file.write(f"Data type: {series.dtype}\n")
            if pd.api.types.is_float_dtype(series):
                stats_file.write("Quantiles:\n")
                for quantile, value in series.quantile(
                    [0.0, 0.25, 0.5, 0.75, 1.0]
                ).items():
                    stats_file.write(f"  {quantile:.2f}: {value}\n")
                plot_continuous_distr(
                    series,
                    f"{column}.png",
                    save=save_stats_plots,
                )
            else:
                stats_file.write(
                    "Quantiles: not applicable (not continuous data)\n"
                )
            stats_file.write("\n")

    ordinal = OrdinalEncoder()
    result["shot"] = ordinal.fit_transform(dataframe[["shot"]])
    ohe = OneHotEncoder()
    result["category"] = ohe.fit_transform(dataframe[["category"]])
    return result


def build_split_lists(verbose: bool = False) -> None:
    """Reload complete metadata rows into the train/test/validation lists."""
    load_metadata_splits()
    if verbose:
        for class_name in CLASSES:
            print(
                f"class={class_name}; train={TRAIN_TEST_VAL[class_name][0]}, "
                f"test={TRAIN_TEST_VAL[class_name][1]}, "
                f"val={TRAIN_TEST_VAL[class_name][2]}"
            )


def plot_PCA(
    dataframe: pd.DataFrame,
    *,
    save_path: str | None = None,
) -> tuple[PCA, pd.DataFrame]:
    """Project metadata onto five PCA components and plot all 2D pairs."""
    excluded_columns = {
        "label",
        "filename",
        "category",
        "split",
        "shot",
        "source_id",
        "username",
        "url",
        "orig_file",
        "content_hash",
    }
    feature_columns = [
        column
        for column in dataframe.select_dtypes(include="number").columns
        if column not in excluded_columns
    ]
    if len(feature_columns) < 5:
        raise ValueError("PCA requires at least five numeric metadata features")

    numeric_features = dataframe[feature_columns].dropna()
    if len(numeric_features) < 5:
        raise ValueError("PCA requires at least five complete metadata rows")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)
    pca = PCA(n_components=5)
    components = pca.fit_transform(scaled_features)
    projected = pd.DataFrame(
        components,
        index=numeric_features.index,
        columns=[
            "principal_component_1",
            "principal_component_2",
            "principal_component_3",
            "principal_component_4",
            "principal_component_5",
        ],
    )
    projected["label"] = dataframe.loc[numeric_features.index, "label"]

    component_pairs = list(combinations(range(5), 2))
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), squeeze=False)
    class_labels = sorted(projected["label"].dropna().unique())
    color_map = {
        label: matplotlib.colormaps["tab20"](index % 20)
        for index, label in enumerate(class_labels)
    }
    for ax, (x_component, y_component) in zip(axes.flat, component_pairs):
        for label in class_labels:
            group = projected[projected["label"] == label]
            ax.scatter(
                group.iloc[:, x_component],
                group.iloc[:, y_component],
                color=color_map[label],
                alpha=0.7,
                s=18,
            )
        ax.set_xlabel(f"Principal Component {x_component + 1}")
        ax.set_ylabel(f"Principal Component {y_component + 1}")
        ax.set_title(f"PC{x_component + 1} vs PC{y_component + 1}")
        ax.grid(alpha=0.25)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[label],
            markersize=7,
            label=label,
        )
        for label in class_labels
    ]
    fig.suptitle("Metadata projected onto all PCA component pairs")
    fig.legend(
        handles,
        class_labels,
        title="Class",
        loc="center left",
        bbox_to_anchor=(0.91, 0.5),
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return pca, projected


def generate_sample_outputs() -> None:
    """Compute and save the six representative sample visualizations."""
    rng = random.Random(42)
    examples = [
        (class_name, rng.choice(class_files(class_name)))
        for class_name in [rng.choice(CLASSES) for _ in range(6)]
    ]
    for index, (class_name, filename) in enumerate(examples):
        clip, sample_rate = librosa.load(
            get_entry(class_name, filename), sr=16000, mono=True
        )
        name = f"{class_name}_{filename}"
        values = extract_level_features(
            clip, sample_rate, include_mel_spectrum=True
        )
        level_stats: Dict[str, object] = {
            "rms_dbfs": float(values[0]),
            "crest_factor_db": float(values[1]),
            "noise_floor_dbfs_p10": float(values[2]),
            "signal_level_dbfs_p95": float(values[3]),
            "approx_snr_db": float(values[4]),
            "active_frame_fraction": float(values[5]),
            "spectral_centroid": float(values[6]),
            "spectral_bandwidth": float(values[7]),
            "spectral_rolloff_85": float(values[8]),
            "spectral_flatness": float(values[9]),
            "zero_crossing_rate": float(values[10]),
            "mfcc_means": values[11:31],
            "mfcc_stds": values[31:51],
            "mean_mel_spectrum": values[51:115],
        }
        RMSE_ZCR_plot(clip, name, index)
        SNR_plot(clip, name, index, sr=sample_rate)
        plot_spectrogram_1(
            clip, name, index, level_stats=cast(Dict[str, float], level_stats)
        )
        plot_spectrogram_2(clip, name, index)
        with open(sample_stats(name, index), "w", encoding="utf-8") as stats:
            stats.write(f"file: {get_entry(class_name, filename)}\n")
            stats.write(f"class: {class_name}\n")
            stats.write(f"rms_dbfs: {values[0]:.3f}\n")
            stats.write(f"crest_factor_db: {values[1]:.3f}\n")
            stats.write(f"noise_floor_dbfs_p10: {values[2]:.3f}\n")
            stats.write(f"signal_level_dbfs_p95: {values[3]:.3f}\n")
            stats.write(f"approx_snr_db: {values[4]:.3f}\n")
            stats.write(f"active_frame_fraction: {values[5]:.3f}\n")
            stats.write(f"spectral_descriptors: {values[6:11].tolist()}\n")
            stats.write(f"mfcc_means: {values[11:31].tolist()}\n")
            stats.write(f"mfcc_stds: {values[31:51].tolist()}\n")
            stats.write(f"mean_mel_spectrum: {values[51:115].tolist()}\n")


def main() -> None:
    args = parse_args()
    metadata_result = read_metadata(
        save_stats_plots=not args.skip_stats_plotting
    )
    if "error" in metadata_result:
        raise metadata_result["error"]  # type: ignore[misc]
    dataframe = metadata_result["dataframe"]
    assert isinstance(dataframe, pd.DataFrame)

    counts = {
        class_name: len(class_files(class_name)) for class_name in CLASSES
    }
    plt.figure()
    plt.pie(
        list(counts.values()),
        labels=list(counts),
        rotatelabels=False,
        autopct="%1.1f%%",
        startangle=90,
    )
    if not args.skip_pie_chart_plotting:
        plt.savefig(visual_file("PIE_CHART_DISTRIBUTION.png"))
    plt.close()

    build_split_lists()
    generate_sample_outputs()
    os.makedirs(os.path.join("visuals", "EDA", "train"), exist_ok=True)
    plot_PCA(
        dataframe,
        save_path=os.path.join(
            "visuals", "EDA", "train", "pca_metadata_5_components.png"
        ),
    )
    if os.path.exists("features.csv") and os.path.getsize("features.csv") > 0:
        print("[INFO] Found existing 'features.csv', loading precomputed features...")
        feature_table = pd.read_csv("features.csv")
    else:
        print("[INFO] Extracting audio dataset features (this may take a few minutes)...")
        feature_table = extract_dataset_features(
            dataframe, include_mel_spectrum=True
        )
        if not args.skip_feature_dumping:
            feature_table.to_csv("features.csv", index=False)
    load_metadata_splits(feature_table=feature_table)
    plot_train_feature_boxplots(feature_table)
    plot_feature_boxplots(
        feature_table,
        [
            "rms_dbfs",
            "crest_factor_db",
            "noise_floor_dbfs_p10",
            "signal_level_dbfs_p95",
            "approx_snr_db",
            "active_frame_fraction",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff_85",
            "spectral_flatness",
            "zero_crossing_rate",
        ],
    )
    plot_train_correlation_matrices(feature_table)


# ==============================================================================
# Model Grids and Combinations
# ==============================================================================
svm_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.1, 1],
    "kernel": ["rbf", "linear"],
}
svm_combos = list(product(svm_grid["C"], svm_grid["gamma"], svm_grid["kernel"]))

rf_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [10, 15, 20],
    "max_features": ["sqrt", "log2"],
}
rf_combos = list(product(rf_grid["criterion"], rf_grid["max_depth"], rf_grid["max_features"]))

gbc_grid = {
    "learning_rate": [0.05, 0.1],
    "max_iter": [50, 80],
    "max_leaf_nodes": [15, 31],
}
gbc_combos = list(product(gbc_grid["learning_rate"], gbc_grid["max_iter"], gbc_grid["max_leaf_nodes"]))


def format_svm_name(C: float, gamma: Any, kernel: str) -> str:
    """Format SVM combo name matching 'SVM_C_01_gamma_0.01_kernel_rbf' style."""
    c_str = str(C).replace(".", "") if isinstance(C, float) and C < 1 else str(C)
    return f"SVM_C_{c_str}_gamma_{gamma}_kernel_{kernel}"


def format_rf_name(criterion: str, max_depth: int, max_features: str) -> str:
    """Format Random Forest combo name."""
    return f"RF_criterion_{criterion}_max_depth_{max_depth}_max_features_{max_features}"


def format_gbc_name(learning_rate: float, max_iter: int, max_leaf_nodes: int) -> str:
    """Format Gradient Boosting combo name."""
    return f"GB_learning_rate_{learning_rate}_max_iter_{max_iter}_max_leaf_nodes_{max_leaf_nodes}"


PICKLED_MODELS: dict[str, bytes] = {}


def evaluate_and_log_model(
    model: Any,
    x_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    model_family: str,
    combo_name: str,
    classes: list[str] = CLASSES,
    is_best: bool = False,
    output_root: str = "eval",
    x_val: pd.DataFrame | np.ndarray | None = None,
    y_val: pd.Series | np.ndarray | None = None,
    cv_score: float | None = None,
) -> dict[str, float]:
    """Evaluate model on test and validation sets, generate confusion matrices, write scores.txt, and pickle best models."""
    combo_dir = os.path.join(output_root, model_family, combo_name)
    os.makedirs(combo_dir, exist_ok=True)

    # Optional evaluation on validation set
    val_acc = val_f1 = val_weighted_acc = None
    if x_val is not None and y_val is not None:
        y_val_pred = model.predict(x_val)
        val_acc = float(accuracy_score(y_val, y_val_pred))
        val_weighted_acc = float(balanced_accuracy_score(y_val, y_val_pred))
        val_f1 = float(f1_score(y_val, y_val_pred, average="macro"))

    # Evaluation on test set
    y_pred = model.predict(x_test)

    acc = float(accuracy_score(y_test, y_pred))
    weighted_acc = float(balanced_accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    report = classification_report(y_test, y_pred, zero_division=0)

    # 1. scores.txt -> F1, precision, recall, accuracy, weighted accuracy
    scores_path = os.path.join(combo_dir, "scores.txt")
    with open(scores_path, "w", encoding="utf-8") as f:
        f.write(f"Model Class:             {model_family}\n")
        f.write(f"Hyperparameters:         {combo_name}\n")
        f.write(f"Is Optimal CV Model:      {'YES' if is_best else 'NO'}\n")
        if cv_score is not None:
            f.write(f"Group CV Acc (source_id): {cv_score:.4f}\n")
        if val_acc is not None:
            f.write("=" * 60 + "\n")
            f.write("VALIDATION SET EVALUATION:\n")
            f.write("=" * 60 + "\n")
            f.write(f"Accuracy:                {val_acc:.4f}\n")
            f.write(f"Weighted Accuracy:       {val_weighted_acc:.4f}\n")
            f.write(f"F1 Score (Macro):        {val_f1:.4f}\n\n")
        f.write("=" * 60 + "\n")
        f.write("FINAL EVALUATION METRICS (TEST SET):\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy:                {acc:.4f}\n")
        f.write(f"Weighted Accuracy:       {weighted_acc:.4f}\n")
        f.write(f"F1 Score (Macro):        {f1_macro:.4f}\n")
        f.write(f"F1 Score (Weighted):     {f1_weighted:.4f}\n")
        f.write(f"Precision (Macro):       {prec_macro:.4f}\n")
        f.write(f"Precision (Weighted):    {prec_weighted:.4f}\n")
        f.write(f"Recall (Macro):          {rec_macro:.4f}\n")
        f.write(f"Recall (Weighted):       {rec_weighted:.4f}\n\n")
        f.write("=" * 60 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write("=" * 60 + "\n")
        f.write(report)

    # 2. confusion_matrix.png (Raw Counts)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        cbar_kws={"label": "Sample Count"},
    )
    ax.set_title(f"Confusion Matrix - {model_family}\n{combo_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=10, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(combo_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    # 2b. confusion_matrix_normalized.png (Row-normalized recall per class)
    cm_norm = confusion_matrix(y_test, y_pred, labels=classes, normalize="true")
    fig_norm, ax_norm = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax_norm,
        cbar_kws={"label": "Normalized Recall"},
    )
    ax_norm.set_title(f"Normalized Confusion Matrix - {model_family}\n{combo_name}", fontsize=11, fontweight="bold")
    ax_norm.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
    ax_norm.set_ylabel("True Class", fontsize=10, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig_norm.savefig(os.path.join(combo_dir, "confusion_matrix_normalized.png"), dpi=120)
    plt.close(fig_norm)

    # 3. Dump model to memory using pickle and persist to disk
    if is_best:
        # Memory dump (bytes)
        PICKLED_MODELS[model_family] = pickle.dumps(model)
        # Disk dump in combo directory
        with open(os.path.join(combo_dir, "final_model.pkl"), "wb") as f:
            pickle.dump(model, f)
        # Disk dump in family root directory
        family_root = os.path.join(output_root, model_family)
        with open(os.path.join(family_root, "final_model.pkl"), "wb") as f:
            pickle.dump(model, f)

    return {
        "accuracy": acc,
        "weighted_accuracy": weighted_acc,
        "f1_macro": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "val_accuracy": val_acc if val_acc is not None else 0.0,
    }


def describe_and_plot_variables(
    train_df: pd.DataFrame,
    output_dir: str = "pre_training_analysis",
) -> None:
    """Export descriptive statistics, dataframe head dump, and distribution plots for all final variables."""
    # os.makedirs(output_dir, exist_ok=True)
    dist_dir = os.path.join(output_dir, "distributions")
    mfcc_dist = os.path.join(dist_dir, "mfcc_std")
    mean_mel = os.path.join(dist_dir, "mean_mel")
    mfcc_mean = os.path.join(dist_dir, "mfcc_mean")
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(mfcc_dist, exist_ok=True)
    os.makedirs(mean_mel, exist_ok=True)
    os.makedirs(mfcc_mean, exist_ok=True)
    print(f"\n[ANALYSIS] Exporting pre-training descriptive statistics and distributions to '{output_dir}'...")

    # 1. Save dataframe head to a .txt file
    head_txt_path = os.path.join(output_dir, "dataframe_head.txt")
    with open(head_txt_path, "w", encoding="utf-8") as head_file:
        head_file.write("=" * 80 + "\n")
        head_file.write("PRE-TRAINING FINAL DATAFRAME HEAD (First 10 Rows)\n")
        head_file.write(f"Shape: {train_df.shape[0]} rows x {train_df.shape[1]} columns\n")
        head_file.write("=" * 80 + "\n\n")
        with pd.option_context("display.max_columns", None, "display.width", 1000):
            head_file.write(train_df.head(10).to_string())
        head_file.write("\n\n" + "=" * 80 + "\n")
        head_file.write("CLASS LABEL DISTRIBUTION:\n")
        head_file.write("=" * 80 + "\n")
        head_file.write(train_df["label"].value_counts().to_string())
        head_file.write("\n\n" + "=" * 80 + "\n")
        head_file.write("COLUMNS AND DATA TYPES:\n")
        head_file.write("=" * 80 + "\n")
        for col_name, dtype in train_df.dtypes.items():
            head_file.write(f"{col_name:<35}: {dtype}\n")

    # 2. Compute and save descriptive statistics
    numeric_df = train_df.select_dtypes(include="number")
    stats_df = numeric_df.describe().T
    stats_df["skew"] = numeric_df.skew()
    stats_df["kurtosis"] = numeric_df.kurtosis()

    stats_txt_path = os.path.join(output_dir, "descriptive_statistics.txt")
    with open(stats_txt_path, "w", encoding="utf-8") as stats_file:
        stats_file.write("=" * 100 + "\n")
        stats_file.write("DESCRIPTIVE STATISTICS OF FINAL NUMERIC VARIABLES (After Encoding & Outlier Removal)\n")
        stats_file.write("=" * 100 + "\n\n")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            stats_file.write(stats_df.to_string())

    # 3. Plot individual distributions for every numeric variable
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(
            series,
            bins=35,
            density=True,
            alpha=0.6,
            color="#3498db",
            edgecolor="#2980b9",
        )
        if series.nunique() > 5 and series.std() > 1e-6:
            try:
                kde_series = series.sample(min(len(series), 1000), random_state=42)
                kde = gaussian_kde(kde_series)
                x_vals = np.linspace(series.min(), series.max(), 200)
                ax.plot(x_vals, kde(x_vals), color="#e74c3c", linewidth=2, label="KDE")
            except Exception:
                pass
        mean_val = float(series.mean())
        median_val = float(series.median())
        ax.axvline(mean_val, color="#2ecc71", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color="#f39c12", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.2f}")
        ax.set_title(f"Distribution: {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        safe_col_name = col.replace("/", "_").replace("\\", "_")
        if col.startswith("mfcc_") and "std" in col:
            fig.savefig(os.path.join(mfcc_dist, f"{safe_col_name}.png"), dpi=100)
        elif col.startswith("mfcc_"):
            fig.savefig(os.path.join(mfcc_mean, f"{safe_col_name}.png"), dpi=100)
        elif col.startswith("mean_mel_"):
            fig.savefig(os.path.join(mean_mel, f"{safe_col_name}.png"), dpi=100)
        else:
            fig.savefig(os.path.join(dist_dir, f"{safe_col_name}.png"), dpi=100)
        plt.close(fig)

    # 4. Grouped summary plots
    core_scalars = [
        "rms_dbfs", "crest_factor_db", "noise_floor_dbfs_p10",
        "signal_level_dbfs_p95", "approx_snr_db", "active_frame_fraction",
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff_85",
        "spectral_flatness", "zero_crossing_rate"
    ]
    present_scalars = [c for c in core_scalars if c in numeric_df.columns]
    if present_scalars:
        n_cols = 3
        n_rows = (len(present_scalars) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows))
        for ax, feat in zip(axes.flat, present_scalars):
            vals = numeric_df[feat].dropna()
            ax.hist(vals, bins=30, color="#1abc9c", edgecolor="#16a085", alpha=0.7)
            ax.set_title(feat, fontsize=10, fontweight="bold")
            ax.grid(alpha=0.25)
        for ax in axes.flat[len(present_scalars):]:
            ax.set_visible(False)
        fig.suptitle("Core Audio Features - Post-Processing Distributions", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "01_core_audio_features_summary.png"), dpi=130)
        plt.close(fig)

    meta_cols = [
        c for c in numeric_df.columns
        if not c.startswith("mfcc_") and not c.startswith("mean_mel_") and c not in core_scalars
    ]
    if meta_cols:
        n_cols = 3
        n_rows = (len(meta_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.2 * n_rows), squeeze=False)
        for ax, feat in zip(axes.flat, meta_cols):
            vals = numeric_df[feat].dropna()
            ax.hist(vals, bins=25, color="#9b59b6", edgecolor="#8e44ad", alpha=0.7)
            ax.set_title(feat, fontsize=9, fontweight="bold")
            ax.grid(alpha=0.25)
        for ax in axes.flat[len(meta_cols):]:
            ax.set_visible(False)
        fig.suptitle("Audio Contrast & SAD Features - Distributions", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "02_audio_contrast_sad_summary.png"), dpi=130)
        plt.close(fig)

    print(f"[ANALYSIS] Done. Generated:")
    print(f"  -> {head_txt_path}")
    print(f"  -> {stats_txt_path}")
    print(f"  -> {len(numeric_df.columns)} distribution plots in '{dist_dir}'")
    print(f"  -> Overview charts in '{output_dir}'")


def update_best_params(
    mean_score: float,
    best_cv_score: float,
    best_params: dict[str, Any] | None,
    current_params: dict[str, Any],
) -> tuple[bool, float, dict[str, Any]]:
    """Check if mean_score improves upon best_cv_score, returning (is_new_best, best_cv_score, best_params)."""
    if mean_score > best_cv_score:
        return True, mean_score, current_params
    return False, best_cv_score, best_params if best_params is not None else {}


# Strictly exclude all non-audio metadata columns from the feature matrix
# per the project specification:
# "Nu folosiți coloanele din metadata.csv ca trăsături: orig_duration_s, pad_s, peak_dbfs_before,
# shot etc. Ele descriu modul în care a fost construit setul, nu sunetul, și nu vor avea același
# înțeles pentru înregistrările voastre. Modelul trebuie să decidă doar pe baza semnalului audio."
METADATA_COLUMNS = {
    "label",
    "filename",
    "split",
    "source_id",
    "username",
    "url",
    "orig_file",
    "content_hash",
    "category",
    "shot",
    "orig_duration_s",
    "window_start_s",
    "pad_s",
    "peak_dbfs_before",
    "pad_noise_rms",
    "approx_snr_db_raw",
    "spectral_centroid_raw",
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature matrix strictly using audio signal features and audio-derived SAD features.
    
    All columns from metadata.csv (orig_duration_s, pad_s, peak_dbfs_before, shot, category, etc.)
    are excluded per the project requirement.
    """
    drop_cols = [
        c for c in df.columns
        if c in METADATA_COLUMNS or c.startswith("category_") or c.endswith("_raw")
    ]
    feats_df = df.drop(columns=drop_cols, errors="ignore").copy()

    # Improvement 3: Sound Activity Detection (SAD) active contrast and silence de-biasing features
    if "active_frame_fraction" in feats_df.columns:
        active_frac = np.clip(feats_df["active_frame_fraction"].astype(float), 0.01, 1.0)

        # 1. Active SNR: Effective SNR scaled by active sound duration
        if "signal_level_dbfs_p95" in feats_df.columns and "noise_floor_dbfs_p10" in feats_df.columns:
            feats_df["sad_active_snr"] = (
                feats_df["signal_level_dbfs_p95"].astype(float) - feats_df["noise_floor_dbfs_p10"].astype(float)
            ) / active_frac

        # 2. Impulsiveness ratio: separates short transient clicks from continuous sounds
        if "crest_factor_db" in feats_df.columns:
            feats_df["sad_impulse_ratio"] = feats_df["crest_factor_db"].astype(float) / active_frac

        # 3. Active Energy Density: log-energy concentrated in the active sound region
        if "rms_dbfs" in feats_df.columns:
            feats_df["sad_active_energy_density"] = (
                feats_df["rms_dbfs"].astype(float) - 10.0 * np.log10(active_frac)
            )

        # 4. Silence-compensated spectral centroid: prevents silence from pulling centroid to zero
        if "spectral_centroid" in feats_df.columns:
            feats_df["sad_active_centroid"] = feats_df["spectral_centroid"].astype(float) / np.sqrt(active_frac)

    return feats_df


def training():
    global svm_grid, rf_grid, gbc_grid, svm_combos, rf_combos, gbc_combos
    global TRAIN, TEST, VAL, PICKLED_MODELS

    train_df = pd.DataFrame(TRAIN).sample(frac=1, random_state=42)
    x_train = prepare_features(train_df)
    y_train = train_df["label"]
    groups_train = train_df["source_id"].values

    test_df = pd.DataFrame(TEST).sample(frac=1, random_state=42)
    x_test = prepare_features(test_df)
    y_test = test_df["label"]

    valid_df = pd.DataFrame(VAL).sample(frac=1, random_state=42)
    x_val = prepare_features(valid_df)
    y_val = valid_df["label"]

    # Combined train + validation set for final evaluation fits
    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    x_train_valid = prepare_features(train_valid_df)
    y_train_valid = train_valid_df["label"]

    # Strict audit: verify ZERO metadata columns exist in any feature matrix
    raw_meta_cols = set(pd.read_csv("metadata.csv", nrows=1).columns)
    forbidden_features = [
        c for c in x_train.columns
        if c in raw_meta_cols or c.startswith("category_") or c.endswith("_raw")
    ]
    assert len(forbidden_features) == 0, (
        f"[ERROR] Forbidden metadata columns detected in feature matrix: {forbidden_features}"
    )
    print(f"\n[AUDIT] Confirmed: {x_train.shape[1]} features are 100% audio-derived.")
    print(f"[AUDIT] Exactly 0 metadata columns from 'metadata.csv' are used.")

    # ================= PRE-TRAINING DESCRIPTIVE ANALYSIS =================
    analysis_df = pd.concat([x_train, y_train], axis=1)
    describe_and_plot_variables(analysis_df, output_dir="pre_training_analysis")

    # Option B: 5-Fold Stratified Group CV on train grouped by source_id
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    os.makedirs("eval", exist_ok=True)

    # ==============================================================================
    # FIRST MODEL: SVM
    # ==============================================================================
    print("\n" + "=" * 36 + " FIRST MODEL: SVM " + "=" * 36)
    total_svm_combos = len(svm_combos)
    print(f"Starting 5-Fold Stratified Group CV Search (grouped by source_id, {total_svm_combos} combinations)...")
    print("-" * 90)

    best_params_svm, best_cv_score_svm = None, -1.0
    svm_cv_scores: dict[str, float] = {}
    t_start = time.time()

    for idx, (C_, gam, ker) in enumerate(svm_combos, start=1):
        t_iter = time.time()

        # StandardScaler is vital for SVM to prevent ill-conditioned quadratic solver
        model_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=C_, kernel=ker, gamma=gam, random_state=42)),
        ])

        scores = cross_val_score(
            model_pipeline, x_train, y_train,
            cv=sgkf, groups=groups_train, scoring="accuracy", n_jobs=-1,
        )
        mean_score = float(scores.mean())
        std_score = float(scores.std())
        elapsed = time.time() - t_iter
        combo_key = format_svm_name(C_, gam, ker)
        svm_cv_scores[combo_key] = mean_score

        is_new_best, best_cv_score_svm, best_params_svm = update_best_params(
            mean_score, best_cv_score_svm, best_params_svm,
            {"C": C_, "gamma": gam, "kernel": ker},
        )

        status_flag = " <-- [NEW BEST]" if is_new_best else ""
        print(
            f"[{idx:02d}/{total_svm_combos}] C={C_:<5} gamma={str(gam):<6} kernel={ker:<7} | "
            f"Group CV Acc: {mean_score:.4f} (+/- {std_score:.4f}) | "
            f"Time: {elapsed:.2f}s{status_flag}",
            flush=True,
        )

    assert best_params_svm is not None
    print("-" * 90)
    print(f"SVM Tuning Complete in {time.time() - t_start:.2f}s")
    print(f"Best Parameters: {best_params_svm} | Best Group CV Accuracy: {best_cv_score_svm:.4f}\n")

    val_model_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(**cast(Dict[str, Any], best_params_svm), random_state=42)),
    ])
    val_model_svm.fit(x_train, y_train)
    val_score_svm = float(val_model_svm.score(x_val, y_val))
    print(f"SVM Validation Accuracy (fit on train, eval on val): {val_score_svm:.4f}")

    # Final Evaluation & Artifact Generation across all SVM combinations
    print(f"\n[EVAL] Generating evaluation artifacts for all {total_svm_combos} SVM combinations in 'eval/SVM'...")
    svm_final_model = None
    svm_best_combo_name = ""
    svm_best_metrics: dict[str, float] = {}

    for C_, gam, ker in svm_combos:
        combo_name = format_svm_name(C_, gam, ker)
        is_best = ({"C": C_, "gamma": gam, "kernel": ker} == best_params_svm)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=C_, kernel=ker, gamma=gam, random_state=42)),
        ])
        pipeline.fit(x_train_valid, y_train_valid)
        metrics = evaluate_and_log_model(
            model=pipeline,
            x_test=x_test,
            y_test=y_test,
            model_family="SVM",
            combo_name=combo_name,
            classes=CLASSES,
            is_best=is_best,
            output_root="eval",
            x_val=x_val,
            y_val=y_val,
            cv_score=svm_cv_scores.get(combo_name),
        )
        if is_best:
            svm_final_model = pipeline
            svm_best_combo_name = combo_name
            svm_best_metrics = metrics
            print(
                f"  -> Best SVM ({combo_name}) | Test Acc: {metrics['accuracy']:.4f} | "
                f"Weighted Acc: {metrics['weighted_accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}"
            )

    # Persist best SVM model to eval/best_SVM_model.pkl
    best_svm_path = os.path.join("eval", "best_SVM_model.pkl")
    with open(best_svm_path, "wb") as f:
        pickle.dump(svm_final_model, f)
    PICKLED_MODELS["best_SVM_model"] = pickle.dumps(svm_final_model)
    print(f"[DISK PICKLE] Persisted best SVM model to: {best_svm_path}")

    # ==============================================================================
    # SECOND MODEL: Random Forest
    # ==============================================================================
    print("\n" + "=" * 32 + " SECOND MODEL: Random Forest " + "=" * 32)
    best_params_rf, best_cv_score_rf = None, -1.0
    rf_cv_scores: dict[str, float] = {}
    total_rf_combos = len(rf_combos)
    print(f"Starting 5-Fold Stratified Group CV Search (grouped by source_id, {total_rf_combos} combinations)...")
    print("-" * 90)

    t_start_rf = time.time()
    for idx, (crit, dep, feats) in enumerate(rf_combos, start=1):
        t_iter = time.time()
        rfc = RandomForestClassifier(
            criterion=crit,
            max_depth=dep,
            max_features=feats,
            random_state=42,
            n_jobs=-1,
        )
        scores = cross_val_score(
            rfc, x_train, y_train,
            cv=sgkf, groups=groups_train, scoring="accuracy", n_jobs=-1,
        )
        mean_score = float(scores.mean())
        std_score = float(scores.std())
        elapsed = time.time() - t_iter
        combo_key = format_rf_name(crit, dep, feats)
        rf_cv_scores[combo_key] = mean_score

        is_new_best, best_cv_score_rf, best_params_rf = update_best_params(
            mean_score, best_cv_score_rf, best_params_rf,
            {"criterion": crit, "max_depth": dep, "max_features": feats},
        )

        status_flag = " <-- [NEW BEST]" if is_new_best else ""
        print(
            f"[{idx:02d}/{total_rf_combos}] criterion={crit:<7} max_depth={dep:<3} max_features={feats:<5} | "
            f"Group CV Acc: {mean_score:.4f} (+/- {std_score:.4f}) | "
            f"Time: {elapsed:.2f}s{status_flag}",
            flush=True,
        )

    assert best_params_rf is not None
    print("-" * 90)
    print(f"Random Forest Tuning Complete in {time.time() - t_start_rf:.2f}s")
    print(f"Best Parameters: {best_params_rf} | Best Group CV Accuracy: {best_cv_score_rf:.4f}\n")

    val_model_rf = RandomForestClassifier(
        **cast(Dict[str, Any], best_params_rf),
        random_state=42,
        n_jobs=-1,
    )
    val_model_rf.fit(x_train, y_train)
    val_score_rf = float(val_model_rf.score(x_val, y_val))
    print(f"Random Forest Validation Accuracy (fit on train, eval on val): {val_score_rf:.4f}")

    # Final Evaluation & Artifact Generation across all Random Forest combinations
    print(f"\n[EVAL] Generating evaluation artifacts for all {total_rf_combos} Random Forest combinations in 'eval/RandomForest'...")
    rf_final_model = None
    rf_best_combo_name = ""
    rf_best_metrics: dict[str, float] = {}

    for crit, dep, feats in rf_combos:
        combo_name = format_rf_name(crit, dep, feats)
        is_best = ({"criterion": crit, "max_depth": dep, "max_features": feats} == best_params_rf)
        rfc = RandomForestClassifier(
            criterion=crit,
            max_depth=dep,
            max_features=feats,
            random_state=42,
            n_jobs=-1,
        )
        rfc.fit(x_train_valid, y_train_valid)
        metrics = evaluate_and_log_model(
            model=rfc,
            x_test=x_test,
            y_test=y_test,
            model_family="RandomForest",
            combo_name=combo_name,
            classes=CLASSES,
            is_best=is_best,
            output_root="eval",
            x_val=x_val,
            y_val=y_val,
            cv_score=rf_cv_scores.get(combo_name),
        )
        if is_best:
            rf_final_model = rfc
            rf_best_combo_name = combo_name
            rf_best_metrics = metrics
            print(
                f"  -> Best Random Forest ({combo_name}) | Test Acc: {metrics['accuracy']:.4f} | "
                f"Weighted Acc: {metrics['weighted_accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}"
            )

    # Persist best Random Forest model to eval/best_RF_model.pkl
    best_rf_path = os.path.join("eval", "best_RF_model.pkl")
    with open(best_rf_path, "wb") as f:
        pickle.dump(rf_final_model, f)
    PICKLED_MODELS["best_RF_model"] = pickle.dumps(rf_final_model)
    print(f"[DISK PICKLE] Persisted best RF model to: {best_rf_path}")

    # ==============================================================================
    # THIRD MODEL: HistGradientBoosting
    # ==============================================================================
    print("\n" + "=" * 27 + " THIRD MODEL: Gradient Boosting " + "=" * 27)
    best_params_gbc, best_cv_score_gbc = None, -1.0
    gbc_cv_scores: dict[str, float] = {}
    total_gbc_combos = len(gbc_combos)
    print(f"Starting 5-Fold Stratified Group CV Search (grouped by source_id, {total_gbc_combos} combinations)...")
    print("-" * 90)

    t_start_gbc = time.time()
    for idx, (lr, m_iter, mln) in enumerate(gbc_combos, start=1):
        t_iter = time.time()
        gbc = HistGradientBoostingClassifier(
            learning_rate=lr,
            max_iter=m_iter,
            max_leaf_nodes=mln,
            random_state=42,
        )
        scores = cross_val_score(
            gbc, x_train, y_train,
            cv=sgkf, groups=groups_train, scoring="accuracy", n_jobs=-1,
        )
        mean_score = float(scores.mean())
        std_score = float(scores.std())
        elapsed = time.time() - t_iter
        combo_key = format_gbc_name(lr, m_iter, mln)
        gbc_cv_scores[combo_key] = mean_score

        is_new_best, best_cv_score_gbc, best_params_gbc = update_best_params(
            mean_score, best_cv_score_gbc, best_params_gbc,
            {"learning_rate": lr, "max_iter": m_iter, "max_leaf_nodes": mln},
        )

        status_flag = " <-- [NEW BEST]" if is_new_best else ""
        print(
            f"[{idx:02d}/{total_gbc_combos}] lr={lr:<5} max_iter={m_iter:<3} max_leaf_nodes={mln:<2} | "
            f"Group CV Acc: {mean_score:.4f} (+/- {std_score:.4f}) | "
            f"Time: {elapsed:.2f}s{status_flag}",
            flush=True,
        )

    assert best_params_gbc is not None
    print("-" * 90)
    print(f"Gradient Boosting Tuning Complete in {time.time() - t_start_gbc:.2f}s")
    print(f"Best Parameters: {best_params_gbc} | Best Group CV Accuracy: {best_cv_score_gbc:.4f}\n")

    val_model_gbc = HistGradientBoostingClassifier(
        **cast(Dict[str, Any], best_params_gbc),
        random_state=42,
    )
    val_model_gbc.fit(x_train, y_train)
    val_score_gbc = float(val_model_gbc.score(x_val, y_val))
    print(f"Gradient Boosting Validation Accuracy (fit on train, eval on val): {val_score_gbc:.4f}")

    # Final Evaluation & Artifact Generation across all Gradient Boosting combinations
    print(f"\n[EVAL] Generating evaluation artifacts for all {total_gbc_combos} Gradient Boosting combinations in 'eval/Gradient Boosting'...")
    gbc_final_model = None
    gbc_best_combo_name = ""
    gbc_best_metrics: dict[str, float] = {}

    for lr, m_iter, mln in gbc_combos:
        combo_name = format_gbc_name(lr, m_iter, mln)
        is_best = ({"learning_rate": lr, "max_iter": m_iter, "max_leaf_nodes": mln} == best_params_gbc)
        gbc = HistGradientBoostingClassifier(
            learning_rate=lr,
            max_iter=m_iter,
            max_leaf_nodes=mln,
            random_state=42,
        )
        gbc.fit(x_train_valid, y_train_valid)
        metrics = evaluate_and_log_model(
            model=gbc,
            x_test=x_test,
            y_test=y_test,
            model_family="Gradient Boosting",
            combo_name=combo_name,
            classes=CLASSES,
            is_best=is_best,
            output_root="eval",
            x_val=x_val,
            y_val=y_val,
            cv_score=gbc_cv_scores.get(combo_name),
        )
        if is_best:
            gbc_final_model = gbc
            gbc_best_combo_name = combo_name
            gbc_best_metrics = metrics
            print(
                f"  -> Best Gradient Boosting ({combo_name}) | Test Acc: {metrics['accuracy']:.4f} | "
                f"Weighted Acc: {metrics['weighted_accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}"
            )

    # Persist best Gradient Boosting model to eval/best_GB_model.pkl
    best_gb_path = os.path.join("eval", "best_GB_model.pkl")
    with open(best_gb_path, "wb") as f:
        pickle.dump(gbc_final_model, f)
    PICKLED_MODELS["best_GB_model"] = pickle.dumps(gbc_final_model)
    print(f"[DISK PICKLE] Persisted best GB model to: {best_gb_path}")

    # ==============================================================================
    # OVERALL FINAL MODEL SELECTION & IN-MEMORY PICKLE DUMP
    # ==============================================================================
    print("\n" + "=" * 32 + " OVERALL FINAL MODEL SELECTION " + "=" * 32)
    candidate_models = [
        ("SVM", svm_final_model, svm_best_combo_name, svm_best_metrics),
        ("RandomForest", rf_final_model, rf_best_combo_name, rf_best_metrics),
        ("Gradient Boosting", gbc_final_model, gbc_best_combo_name, gbc_best_metrics),
    ]
    # Rank models by test accuracy
    candidate_models.sort(key=lambda item: item[3].get("accuracy", 0.0), reverse=True)
    best_family, overall_best_model, best_combo_name, best_overall_metrics = candidate_models[0]

    print(f"Overall Best Model Family: {best_family}")
    print(f"Optimal Hyperparameters:   {best_combo_name}")
    print(f"Test Accuracy:             {best_overall_metrics.get('accuracy', 0.0):.4f}")
    print(f"Test Weighted Accuracy:    {best_overall_metrics.get('weighted_accuracy', 0.0):.4f}")
    print(f"Test F1 Score (Macro):     {best_overall_metrics.get('f1_macro', 0.0):.4f}")
    print(f"Test Precision (Macro):    {best_overall_metrics.get('precision_macro', 0.0):.4f}")
    print(f"Test Recall (Macro):       {best_overall_metrics.get('recall_macro', 0.0):.4f}")

    # 1. Dump to memory using pickle
    overall_pickle_bytes = pickle.dumps(overall_best_model)
    PICKLED_MODELS["overall_final_model"] = overall_pickle_bytes
    PICKLED_MODELS["final_model"] = overall_pickle_bytes
    print(f"[MEMORY PICKLE] Successfully dumped final model to memory using pickle ({len(overall_pickle_bytes)} bytes in RAM).")

    # 2. Persist final model to disk at eval/final_model.pkl
    overall_pkl_path = os.path.join("eval", "final_model.pkl")
    with open(overall_pkl_path, "wb") as f:
        f.write(overall_pickle_bytes)
    print(f"[DISK PICKLE] Successfully persisted final model to disk at: {overall_pkl_path}")

    # 3. Create comprehensive summary report in eval/summary_leaderboard.txt
    summary_path = os.path.join("eval", "summary_leaderboard.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DOMESTIC SOUNDS CLASSIFICATION - FINAL MODEL LEADERBOARD\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Family':<20} | {'Test Accuracy':<15} | {'Weighted Acc':<15} | {'F1 (Macro)':<12} | {'Optimal Config'}\n")
        f.write("-" * 80 + "\n")
        for fam, _, cfg, met in candidate_models:
            f.write(
                f"{fam:<20} | {met.get('accuracy', 0.0):<15.4f} | "
                f"{met.get('weighted_accuracy', 0.0):<15.4f} | "
                f"{met.get('f1_macro', 0.0):<12.4f} | {cfg}\n"
            )
        f.write("=" * 80 + "\n")
        f.write(f"Best Model Selected: {best_family} ({best_combo_name})\n")
        f.write(f"In-Memory Pickle Key: 'final_model' in PICKLED_MODELS\n")
        f.write(f"Persisted Path:       {overall_pkl_path}\n")
    print(f"[SUMMARY] Leaderboard written to: {summary_path}\n")

if __name__ == "__main__":
    main()
    load_metadata_splits()
    training()