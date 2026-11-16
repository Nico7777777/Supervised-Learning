"""Utilities for rendering and saving audio spectrograms."""

from pathlib import Path
import os
from importlib import import_module
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import importlib
import numpy as np
import pandas as pd

librosa = importlib.import_module("librosa")
__all__ = [
    "plot_spectrogram_1",
    "plot_spectrogram_2",
    "RMSE_ZCR_plot",
    "calculate_snr",
    "SNR_plot",
    "plot_train_feature_boxplots",
    "plot_train_correlation_matrices",
]

FRAME_SIZE = 1024
HOP_LENGTH = 160
def _librosa():
    """Load librosa without requiring it to be resolved during analysis."""
    try:
        return import_module("librosa")
    except ImportError as exc:
        raise ImportError(
            "librosa is required to render spectrograms; install it with "
            "'pip install librosa'."
        ) from exc


def _output_path(filename: str, sample_index: int, output_root: str) -> Path:
    """Create and return the output path for a spectrogram image."""
    directory = (
        Path(output_root) / "spectograms" / f"sample_{sample_index}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def _eda_output_path(filename: str, sample_index: int, output_root: str) -> Path:
    directory = Path(output_root) / "EDA" / f"sample_{sample_index}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def calculate_snr(
    audio: np.ndarray,
    *,
    sr: int = 16000,
    frame_length: int = 512,
    hop_length: int = 160,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate clip SNR from short-time RMS levels using librosa.

    The 10th percentile of frame levels estimates the noise floor and the
    95th percentile estimates the signal level. The returned frame SNR
    values are relative to the estimated noise floor.
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise ValueError("audio must contain at least one sample")
    if not np.all(np.isfinite(samples)):
        raise ValueError("audio must contain only finite samples")

    rms = librosa.feature.rms(
        y=samples,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
    )[0]
    levels_db = librosa.amplitude_to_db(
        rms,
        ref=1.0,
        amin=1e-10,
    )
    levels_db = np.maximum(levels_db, -120.0)
    noise_floor_db = float(np.percentile(levels_db, 10))
    signal_level_db = float(np.percentile(levels_db, 95))
    return signal_level_db - noise_floor_db, levels_db, rms


def SNR_plot(
    audio: np.ndarray,
    name: str,
    sample_index: int,
    output_root: str = "visuals",
    *,
    sr: int = 16000,
) -> float:
    """Save the frame-level SNR plot and return the clip SNR in dB."""
    snr_db, frame_levels_db, _ = calculate_snr(audio, sr=sr)
    noise_floor_db = float(np.percentile(frame_levels_db, 10))
    frame_snr_db = frame_levels_db - noise_floor_db
    times = librosa.frames_to_time(
        np.arange(frame_snr_db.size),
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    output_path = _eda_output_path(f"SNR_{name}.png", sample_index, output_root)

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(times, frame_snr_db, color="tab:blue", label="Frame SNR")
    ax.axhline(
        snr_db,
        color="tab:red",
        linestyle="--",
        label=f"Clip SNR: {snr_db:.2f} dB",
    )
    ax.set_title(f"Signal-to-noise ratio: {name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SNR relative to noise floor (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)
    return snr_db


def plot_train_feature_boxplots(
    feature_table: pd.DataFrame,
    output_root: str = "visuals",
) -> None:
    """Save class-wise boxplots for the requested scalar train features."""
    required_columns = {
        "label",
        "split",
        "approx_snr_db",
        "noise_floor_dbfs_p10",
        "active_frame_fraction",
        "spectral_centroid",
        "spectral_rolloff_85",
    }
    missing_columns = required_columns.difference(feature_table.columns)
    if missing_columns:
        raise ValueError(
            f"feature table is missing columns: {sorted(missing_columns)}"
        )

    train_features = feature_table.loc[feature_table["split"] == "train"].copy()
    if train_features.empty:
        raise ValueError("feature table contains no train rows")

    features = {
        "approx_snr_db": "SNR aproximativ (dB)",
        "noise_floor_dbfs_p10": "Zgomot de fond P10 (dBFS)",
        "active_frame_fraction": "Pondere cadre active",
        "spectral_centroid": "Centroid spectral (Hz)",
        "spectral_rolloff_85": "Roll-off 85% (Hz)",
    }
    classes = sorted(train_features["label"].dropna().unique())
    output_directory = Path(output_root) / "EDA" / "train"
    output_directory.mkdir(parents=True, exist_ok=True)

    for column, title in features.items():
        values_by_class = [
            train_features.loc[train_features["label"] == class_name, column]
            .dropna()
            .to_numpy()
            for class_name in classes
        ]
        if any(values.size == 0 for values in values_by_class):
            raise ValueError(f"no train values available for {column}")

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.boxplot(values_by_class, labels=classes, showmeans=True)
        ax.set_title(f"{title} pe clase - setul train")
        ax.set_xlabel("Clasa")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_directory, f"{column}_by_class.png"), dpi=150)
        plt.close(fig)


def plot_train_correlation_matrices(
    feature_table: pd.DataFrame,
    output_root: str = "visuals",
) -> dict[str, pd.DataFrame]:
    """Save Pearson and Spearman correlations for scalar train features."""
    feature_columns = [
        "approx_snr_db",
        "noise_floor_dbfs_p10",
        "active_frame_fraction",
        "spectral_centroid",
        "spectral_rolloff_85",
    ]
    missing_columns = set(feature_columns).difference(feature_table.columns)
    if "split" not in feature_table.columns or missing_columns:
        missing = sorted(missing_columns | ({"split"} if "split" not in feature_table else set()))
        raise ValueError(f"feature table is missing columns: {missing}")

    train_features = feature_table.loc[
        feature_table["split"] == "train", feature_columns
    ].dropna()
    if train_features.empty:
        raise ValueError("feature table contains no complete train rows")

    correlations = {
        "pearson": train_features.corr(method="pearson"),
        "spearman": train_features.corr(method="spearman"),
    }
    output_directory = Path(output_root) / "EDA" / "train"
    output_directory.mkdir(parents=True, exist_ok=True)

    for method, matrix in correlations.items():
        matrix.to_csv(output_directory / f"{method}_correlation.csv")
        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(feature_columns)), feature_columns, rotation=45, ha="right")
        ax.set_yticks(range(len(feature_columns)), feature_columns)
        ax.set_title(f"{method.capitalize()} correlation - setul train")
        for row in range(len(feature_columns)):
            for column in range(len(feature_columns)):
                ax.text(
                    column,
                    row,
                    f"{matrix.iloc[row, column]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )
        fig.colorbar(image, ax=ax, label="Coeficient")
        fig.tight_layout()
        fig.savefig(os.path.join(output_directory, f"{method}_correlation.png"), dpi=150)
        plt.close(fig)
    return correlations


def plot_spectrogram_1(
    signal: np.ndarray,
    name: str,
    sample_index: int,
    output_root: str = "visuals",
    level_stats: Optional[Mapping[str, float]] = None,
) -> None:
    """Save a log-frequency power spectrogram.

    The spectrogram is computed using a short-time Fourier transform and
    rendered with a logarithmic frequency axis.

    Args:
        signal: One-dimensional audio samples.
        name: Descriptive name used in the plot title and output filename.
        sample_index: Index used to select the sample-specific output folder.
        output_root: Root directory where the ``spectograms`` directory is
            created.
            level_stats: Optional aggregate level features to display alongside
                the spectrogram.
    """
    librosa = _librosa()
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(signal)))
    fig, ax = plt.subplots(figsize=(20, 15))
    mesh = librosa.display.specshow(spectrogram, y_axis="log", ax=ax)
    colorbar = fig.colorbar(mesh, ax=ax, format="%+2.0f dB")
    ax.set_title(f"Log-frequency Power Spectrogram for {name}", fontsize=30)
    ax.set_xlabel("Time", fontsize=28)
    ax.set_xlim(0, len(signal) / 16000)
    ax.tick_params(axis="both", labelsize=22)
    colorbar.ax.tick_params(labelsize=22)
    filename = f"spectogram_{name[:name.rfind('_')]}_v1.png"
    fig.savefig(str(_output_path(filename, sample_index, output_root)))
    plt.close(fig)


def plot_spectrogram_2(
    signal: np.ndarray,
    name: str,
    sample_index: int,
    output_root: str = "visuals",
) -> None:
    """Save an RMS and log-power spectrogram visualization.

    Args:
        signal: One-dimensional audio samples.
        name: Descriptive name used in the plot title and output filename.
        sample_index: Index used to select the sample-specific output folder.
        output_root: Root directory where the ``spectograms`` directory is
            created.
    """
    librosa = _librosa()
    magnitude, _ = librosa.magphase(librosa.stft(signal))
    rms = librosa.feature.rms(S=magnitude)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    times = librosa.times_like(rms)

    ax[0].semilogy(times, rms[0], label="RMS Energy")
    ax[0].set(xticks=[])
    ax[0].legend()
    ax[0].label_outer()
    librosa.display.specshow(
        magnitude,
        vscale="dBFS",
        y_axis="log",
        x_axis="time",
        ax=ax[1],
    )
    ax[1].set(title=f"log Power spectrogram for {name}")
    filename = f"spectogram_{name}_v2.png"
    fig.savefig(str(_output_path(filename, sample_index, output_root)))
    plt.close(fig)



def plot_continuous_distr(data, path: str, *, save: bool = True) -> None:
    # Calculate quartiles
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)  # Median
    q3 = np.percentile(data, 75)

    # Calculate some percentiles
    p10 = np.percentile(data, 10)
    p90 = np.percentile(data, 90)

    # Plotting the data
    plt.figure(figsize=(10, 6))

    # Histogram
    plt.hist(data, bins=50, alpha=0.6, color='g', label="Data Distribution")

    # Quartiles and percentiles
    plt.axvline(x=q1, color='r', linestyle='--', label="Q1 (25th percentile)")
    plt.axvline(x=q2, color='b', linestyle='--', label="Q2 (Median/50th percentile)")
    plt.axvline(x=q3, color='r', linestyle='--', label="Q3 (75th percentile)")
    plt.axvline(x=p10, color='c', linestyle='-.', label="10th percentile")
    plt.axvline(x=p90, color='m', linestyle='-.', label="90th percentile")

    # Setting labels, title, and legend
    plt.title("Visualization of Quartiles and Percentiles")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(os.path.join('stats', path))
    plt.close()

def RMSE_ZCR_plot(
        audio: np.ndarray, name: str,
        sample_index: int, output_root: str = "visuals") -> None:
    rms = librosa.feature.rms(
        y=audio, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        audio, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH
    )[0]
    frames = np.arange(len(rms))
    t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH, sr=16000)
    output_directory = _output_path("placeholder", sample_index, output_root).parent

    fig, ax = plt.subplots(figsize=(15, 7))
    librosa.display.waveshow(audio, sr=16000, alpha=0.5, ax=ax)
    ax.plot(t, rms, color="r", label="RMS")
    ax.set_title(f"RMS {name}")
    ax.legend()
    fig.savefig(os.path.join(output_directory, f"RMS_{name}.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 7))
    librosa.display.waveshow(audio, sr=16000, alpha=0.5, ax=ax)
    ax.plot(t, zcr, color="y", label="ZCR")
    ax.set_title(f"ZCR {name}")
    ax.legend()
    fig.savefig(os.path.join(output_directory, f"ZCR_{name}.png"))
    plt.close(fig)