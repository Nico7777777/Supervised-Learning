"""Audio feature extraction for the project dataset."""

from typing import List

import librosa  # pyright: ignore[reportMissingImports]
import numpy as np
import pandas as pd

from spectrograms import calculate_snr


def extract_level_features(
    clip: np.ndarray,
    src: int,
    *,
    include_mel_spectrum: bool = False,
) -> np.ndarray:
    """Return level, spectral and MFCC features for one audio clip."""
    samples = np.asarray(clip, dtype=np.float32).reshape(-1)
    if samples.size != 32_000:
        raise ValueError(
            f"clip must contain exactly 32000 samples, got {samples.size}"
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("clip must contain only finite samples")

    epsilon = np.finfo(np.float32).tiny
    dbfs_floor = -120.0
    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples)))
    rms_dbfs = max(20.0 * np.log10(max(rms, epsilon)), dbfs_floor)

    crest_db = 20.0 * np.log10(max(peak, epsilon) / max(rms, epsilon))
    snr_db, frame_levels_dbfs, frame_rms = calculate_snr(
        samples, sr=src, frame_length=512, hop_length=160
    )
    frame_levels_dbfs = np.maximum(frame_levels_dbfs, dbfs_floor)
    noise_floor_dbfs = float(np.percentile(frame_levels_dbfs, 10))
    signal_level_dbfs = float(np.percentile(frame_levels_dbfs, 95))
    active_frame_fraction = float(
        np.mean(frame_levels_dbfs >= np.max(frame_levels_dbfs) - 20.0)
    )

    spectral_features = {
        "spectral_centroid": librosa.feature.spectral_centroid(
            y=samples, sr=src, n_fft=512, hop_length=160,
            win_length=512, center=False,
        )[0],
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(
            y=samples, sr=src, n_fft=512, hop_length=160,
            win_length=512, center=False,
        )[0],
        "spectral_rolloff_85": librosa.feature.spectral_rolloff(
            y=samples, sr=src, roll_percent=0.85, n_fft=512,
            hop_length=160, win_length=512, center=False,
        )[0],
        "spectral_flatness": librosa.feature.spectral_flatness(
            y=samples, n_fft=512, hop_length=160, win_length=512,
            center=False,
        )[0],
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(
            samples, frame_length=512, hop_length=160, center=False,
        )[0],
    }
    # Sound Activity Detection (SAD): frames with energy within 25dB of peak frame
    active_mask = frame_levels_dbfs >= (np.max(frame_levels_dbfs) - 25.0)
    if not np.any(active_mask):
        active_mask = np.ones_like(frame_levels_dbfs, dtype=bool)

    # Active-weighted frames to avoid averaging background silence (Improvement 3)
    active_weights = np.where(active_mask, frame_rms, epsilon)
    w = active_weights / np.sum(active_weights)

    spectral_aggregates = np.array([
        np.sum(values * w)
        for values in spectral_features.values()
    ], dtype=np.float32)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=samples, sr=src, n_fft=512, win_length=512,
        hop_length=160, n_mels=64, center=False, power=2.0,
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel_spectrogram, n_mfcc=20)
    
    # Sound activity-weighted MFCC statistics
    mfcc_mean = np.sum(mfcc * w, axis=1)
    mfcc_std = np.sqrt(np.maximum(np.sum(w * (mfcc - mfcc_mean[:, None]) ** 2, axis=1), 0.0))
    mfcc_features = np.concatenate((mfcc_mean, mfcc_std))

    scalar_features = np.array([
        rms_dbfs, crest_db, noise_floor_dbfs, signal_level_dbfs,
        snr_db, active_frame_fraction,
    ], dtype=np.float32)
    features = np.concatenate((scalar_features, spectral_aggregates, mfcc_features))
    if include_mel_spectrum:
        mel_mean = np.sum(log_mel_spectrogram * w, axis=1)
        features = np.concatenate((features, mel_mean))
    return features.astype(np.float32)


def feature_column_names(include_mel_spectrum: bool = False) -> List[str]:
    """Return names matching the order produced by ``extract_level_features``."""
    names = [
        "rms_dbfs", "crest_factor_db", "noise_floor_dbfs_p10",
        "signal_level_dbfs_p95", "approx_snr_db", "active_frame_fraction",
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff_85",
        "spectral_flatness", "zero_crossing_rate",
    ]
    names.extend(f"mfcc_mean_{index}" for index in range(1, 21))
    names.extend(f"mfcc_std_{index}" for index in range(1, 21))
    if include_mel_spectrum:
        names.extend(f"mean_mel_{index}" for index in range(1, 65))
    return names


def extract_dataset_features(
    metadata: pd.DataFrame,
    *,
    include_mel_spectrum: bool = False,
) -> pd.DataFrame:
    """Extract one feature row per metadata entry and retain dataset identity."""
    rows = []
    feature_names = feature_column_names(include_mel_spectrum)
    for row in metadata.itertuples(index=False):
        clip, sample_rate = librosa.load(row.filename, sr=16000, mono=True)
        values = extract_level_features(
            clip, sample_rate, include_mel_spectrum=include_mel_spectrum
        )
        if values.size != len(feature_names):
            raise RuntimeError(
                f"feature name/value mismatch: {len(feature_names)} names "
                f"for {values.size} values"
            )
        rows.append({
            **dict(zip(feature_names, values.astype(float))),
            "filename": row.filename,
            "label": row.label,
            "split": row.split,
            "source_id": row.source_id,
        })
    return pd.DataFrame(
        rows, columns=feature_names + ["filename", "label", "split", "source_id"]
    )
