"""Dataset analysis and feature extraction entry point."""

import argparse
from itertools import combinations
import os
import random
from typing import Dict, cast

import importlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from __init__ import (
    CLASSES,
    TRAIN,
    TEST,
    VAL,
    TRAIN_TEST_VAL,
    class_files,
    get_entry,
    sample_stats,
    visual_file,
)
from features import extract_dataset_features, extract_level_features
from spectrograms import (
    RMSE_ZCR_plot,
    SNR_plot,
    plot_continuous_distr,
    plot_spectrogram_1,
    plot_spectrogram_2,
    plot_train_correlation_matrices,
    plot_train_feature_boxplots,
)

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
    """Compute train/test/validation file lists for every class."""
    for class_name in CLASSES:
        entries = class_files(class_name)
        train_count, test_count, val_count = TRAIN_TEST_VAL[class_name]
        if verbose:
            print(
                f"class={class_name}; train={train_count}, "
                f"test={test_count}, val={val_count}; total={len(entries)}"
            )
        TRAIN.extend(entries[:train_count])
        TEST.extend(entries[train_count:train_count + test_count])
        VAL.extend(entries[train_count + test_count:])


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
    handles = labels = None
    for ax, (x_component, y_component) in zip(axes.flat, component_pairs):
        for label, group in projected.groupby("label", sort=True):
            scatter = ax.scatter(
                group.iloc[:, x_component],
                group.iloc[:, y_component],
                label=label,
                alpha=0.7,
                s=18,
            )
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        ax.set_xlabel(f"Principal Component {x_component + 1}")
        ax.set_ylabel(f"Principal Component {y_component + 1}")
        ax.set_title(f"PC{x_component + 1} vs PC{y_component + 1}")
        ax.grid(alpha=0.25)
    fig.suptitle("Metadata projected onto all PCA component pairs")
    fig.legend(
        handles,
        labels,
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
        counts.values(),
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
    feature_table = extract_dataset_features(
        dataframe, include_mel_spectrum=True
    )
    if not args.skip_feature_dumping:
        feature_table.to_csv("features.csv", index=False)
    plot_train_feature_boxplots(feature_table)
    plot_train_correlation_matrices(feature_table)


if __name__ == "__main__":
    main()
