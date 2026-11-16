"""Project constants, paths, and dataset metadata."""

import os
import csv
from pathlib import Path
from typing import TypeAlias

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

MetadataEntry: TypeAlias = dict[str, object]

CLASSES = [
    "Cough",
    "DoorOpenClose",
    "DrawerOpenClose",
    "KeyboardTyping",
    "KeyJingle",
    "Knock",
    "MouseButtonClick",
    "PaperTear",
    "Pour",
    "SwitchToggle",
    "Walk",
    "ZipperZip",
]

TRAIN_TEST_VAL = {
    "Cough": (120, 15, 15),
    "DoorOpenClose": (120, 15, 15),
    "DrawerOpenClose": (94, 12, 11),
    "KeyJingle": (83, 10, 11),
    "KeyboardTyping": (67, 8, 9),
    "Knock": (100, 12, 13),
    "MouseButtonClick": (60, 8, 7),
    "PaperTear": (105, 14, 13),
    "Pour": (120, 15, 15),
    "SwitchToggle": (120, 15, 15),
    "Walk": (120, 15, 15),
    "ZipperZip": (120, 15, 15),
}

TRAIN: list[MetadataEntry] = []
TEST: list[MetadataEntry] = []
VAL: list[MetadataEntry] = []
SPLIT: dict[str, list[MetadataEntry]] = {
    "train": TRAIN,
    "test": TEST,
    "val": VAL,
}
PREPROCESSING_PARAMS: dict[str, float] = {}
NORMALIZED_FEATURES = ("approx_snr_db", "spectral_centroid")
FLOAT_METADATA_COLUMNS = (
    "orig_duration_s",
    "window_start_s",
    "pad_s",
    "peak_dbfs_before",
    "pad_noise_rms",
)
EXCLUDED_METADATA_COLUMNS = (
    "source_id",
    "username",
    "url",
    "orig_file",
    "content_hash",
)


def _fit_outlier_bounds(values: pd.Series) -> tuple[float, float]:
    """Return Tukey IQR bounds fitted on training values."""
    numeric = pd.to_numeric(values, errors="raise")
    q1, q3 = numeric.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def preprocess_audio_features() -> None:
    """Remove train-fitted IQR outliers and standardize two audio features."""
    if not TRAIN or "approx_snr_db" not in TRAIN[0]:
        return

    train_frame = pd.DataFrame(TRAIN)
    bounds = {
        feature: _fit_outlier_bounds(train_frame[feature])
        for feature in NORMALIZED_FEATURES
    }
    PREPROCESSING_PARAMS.clear()
    for feature, (lower, upper) in bounds.items():
        PREPROCESSING_PARAMS[f"{feature}_lower"] = lower
        PREPROCESSING_PARAMS[f"{feature}_upper"] = upper

    kept_entries: dict[str, list[MetadataEntry]] = {}
    for split_name, entries in SPLIT.items():
        frame = pd.DataFrame(entries)
        keep = pd.Series(True, index=frame.index)
        for feature, (lower, upper) in bounds.items():
            values = pd.to_numeric(frame[feature], errors="raise")
            keep &= values.between(lower, upper)
        kept_entries[split_name] = [
            entry for index, entry in enumerate(entries) if keep.iloc[index]
        ]

    train_clean = pd.DataFrame(kept_entries["train"])
    scaler = StandardScaler()
    scaler.fit(train_clean[list(NORMALIZED_FEATURES)])
    for index, feature in enumerate(NORMALIZED_FEATURES):
        PREPROCESSING_PARAMS[f"{feature}_mean"] = float(scaler.mean_[index])
        PREPROCESSING_PARAMS[f"{feature}_scale"] = float(scaler.scale_[index])

    for split_name, entries in kept_entries.items():
        values = pd.DataFrame(entries)
        scaled = scaler.transform(values[list(NORMALIZED_FEATURES)])
        for row_index, entry in enumerate(entries):
            for feature in NORMALIZED_FEATURES:
                entry[f"{feature}_raw"] = float(entry[feature])
            for feature_index, feature in enumerate(NORMALIZED_FEATURES):
                entry[feature] = float(scaled[row_index, feature_index])

        SPLIT[split_name].clear()
        SPLIT[split_name].extend(entries)


def preprocess_metadata_features() -> None:
    """Encode categorical metadata columns and cast numeric metadata fields."""
    if not TRAIN:
        return

    train_frame = pd.DataFrame(TRAIN)
    ordinal: OrdinalEncoder | None = None
    if "shot" in train_frame.columns:
        ordinal = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        ordinal.fit(train_frame[["shot"]])

    ohe: OneHotEncoder | None = None
    category_columns: list[str] = []
    if "category" in train_frame.columns:
        try:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")  # type: ignore[call-arg]
        ohe.fit(train_frame[["category"]])
        category_columns = [
            str(column) for column in ohe.get_feature_names_out(["category"])
        ]

    for entries in SPLIT.values():
        if not entries:
            continue
        split_frame = pd.DataFrame(entries)
        shot_encoded = (
            ordinal.transform(split_frame[["shot"]])
            if ordinal is not None and "shot" in split_frame.columns
            else None
        )
        category_encoded = (
            ohe.transform(split_frame[["category"]])
            if ohe is not None and "category" in split_frame.columns
            else None
        )

        for row_index, entry in enumerate(entries):
            if shot_encoded is not None:
                entry["shot"] = float(shot_encoded[row_index, 0])
            if category_encoded is not None:
                entry.pop("category", None)
                for col_index, col_name in enumerate(category_columns):
                    entry[col_name] = float(
                        category_encoded[row_index, col_index]
                    )
            for column in FLOAT_METADATA_COLUMNS:
                if column in entry and entry[column] is not None and entry[column] != "":
                    entry[column] = float(entry[column])  # type: ignore[arg-type]
            for column in EXCLUDED_METADATA_COLUMNS:
                entry.pop(column, None)


def load_metadata_splits(
    metadata_path: str = "metadata.csv",
    feature_table: pd.DataFrame | None = None,
) -> None:
    """Populate split lists with complete metadata rows.

    Rows are grouped by their declared ``label`` and split according to
    ``TRAIN_TEST_VAL``. If provided, ``feature_table`` is joined by
    ``source_id`` so every entry retains metadata and computed audio features.
    """
    TRAIN.clear()
    TEST.clear()
    VAL.clear()

    with open(metadata_path, newline="", encoding="utf-8") as metadata_file:
        rows = list(csv.DictReader(metadata_file))

    if feature_table is None and Path("features.csv").exists():
        candidate = pd.read_csv("features.csv")
        if "filename" in candidate.columns or len(candidate) == len(rows):
            feature_table = candidate

    if feature_table is not None:
        if "filename" in feature_table.columns:
            feature_by_key = {
                str(key): values
                for key, values in feature_table.set_index(
                    "filename"
                ).to_dict(orient="index").items()
            }
            for row in rows:
                key = row.get("filename")
                if key not in feature_by_key:
                    raise ValueError(
                        f"missing computed features for filename {key!r}"
                    )
                row.update(feature_by_key[key])
        elif len(feature_table) == len(rows):
            for row, feature_row in zip(
                rows, feature_table.to_dict(orient="records")
            ):
                row.update(feature_row)
        else:
            feature_by_key = {
                str(key): values
                for key, values in feature_table.set_index(
                    "source_id"
                ).to_dict(orient="index").items()
            }
            for row in rows:
                key = row.get("source_id")
                if key not in feature_by_key:
                    raise ValueError(
                        f"missing computed features for source_id {key!r}"
                    )
                row.update(feature_by_key[key])

    rows_by_label: dict[str, list[MetadataEntry]] = {
        label: [] for label in TRAIN_TEST_VAL
    }
    for row in rows:
        label = row.get("label")
        if label not in rows_by_label:
            raise ValueError(f"unknown metadata label: {label!r}")
        rows_by_label[label].append(row)

    for label, (train_count, test_count, val_count) in TRAIN_TEST_VAL.items():
        entries = rows_by_label[label]
        expected_total = train_count + test_count + val_count
        if len(entries) != expected_total:
            raise ValueError(
                f"{label}: expected {expected_total} metadata rows, "
                f"found {len(entries)}"
            )
        train_end = train_count
        test_end = train_end + test_count
        for split_name, split_entries in (
            ("train", entries[:train_end]),
            ("test", entries[train_end:test_end]),
            ("val", entries[test_end:]),
        ):
            for entry in split_entries:
                entry["split"] = split_name
            SPLIT[split_name].extend(split_entries)
    preprocess_audio_features()
    preprocess_metadata_features()




def visual_file(filename: str) -> str:
    Path("visuals").mkdir(exist_ok=True)
    return os.path.join("visuals", filename)


def class_files(class_name: str) -> list[str]:
    return os.listdir(os.path.join("audio", class_name))


def get_entry(class_name: str, filename: str) -> str:
    return os.path.join("audio", class_name, filename)


def sample_stats(filename: str, sample_index: int) -> str:
    return os.path.join(
        "visuals",
        "spectograms",
        f"sample_{sample_index}",
        f"statistics_{filename}.txt",
    )
