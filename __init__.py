"""Project constants, paths, and dataset metadata."""

import os
from pathlib import Path

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

TRAIN, TEST, VAL = [], [], []


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
