"""Loading raw BionixDB CSVs (as written by EMG-and-IMU-Data-Collection).

Files follow the BionixDB naming convention
``{modality}-{pid}-{action}[-{exercise}]-{trial}.csv``. A *trial key* is the
filename without the modality prefix, e.g. ``p020-stairs-stairmaster-03``.

Pipeline v1 handles the recordings made so far ("v1 format"). Recordings from
app versions that add a ``seq`` column to EMG/IMU or write CVKAS one row per
frame (``pos_x_0``, ...) are a different format: they're detected and refused
with ``UnsupportedFormat`` rather than misread.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

MODALITIES = ("emg", "imu", "cvkas")

V1_COLUMNS = {
    "emg": ["timestamp", "emg1", "emg2"],
    "imu": ["timestamp"] + [f"imu{i}_{a}" for i in (1, 2) for a in ("ax", "ay", "az", "gx", "gy", "gz")],
    "cvkas": ["timestamp", "marker_id", "pos_x", "pos_y", "velocity", "acceleration",
              "hip_angle", "knee_angle"],
}


class UnsupportedFormat(ValueError):
    """The file isn't in the v1 recording format this pipeline version handles."""


def check_format(path: str | Path, modality: str) -> None:
    """Raise UnsupportedFormat unless the file's header is the v1 layout."""
    cols = list(pd.read_csv(path, nrows=0).columns)
    if cols == V1_COLUMNS[modality]:
        return
    if "seq" in cols or "pos_x_0" in cols:
        raise UnsupportedFormat(
            f"{Path(path).name}: newer recording format (seq / one-row-per-frame CVKAS); "
            "not supported by pipeline v1")
    raise UnsupportedFormat(f"{Path(path).name}: unexpected columns {cols}")


def parse_key(key: str) -> dict:
    """``p020-stairs-stairmaster-03`` -> pid, action, exercise (None if absent), trial."""
    parts = key.split("-")
    if len(parts) == 3:
        pid, action, trial = parts
        exercise = None
    elif len(parts) == 4:
        pid, action, exercise, trial = parts
    else:
        return {"pid": None, "action": None, "exercise": None, "trial": None}
    return {"pid": pid, "action": action, "exercise": exercise,
            "trial": int(trial) if trial.isdigit() else None}


def _magnitude(value) -> float:
    """CVKAS velocity/acceleration cells are either a scalar magnitude (older
    app versions) or a stringified vector like ``"[vx, vy]"`` (current app).
    Return the magnitude either way, NaN for empty cells."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    if text.startswith("["):
        return float(np.linalg.norm(np.asarray(ast.literal_eval(text), dtype=float)))
    return float(text)


def read_cvkas(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"velocity": object, "acceleration": object}, **_EXACT_FLOATS)
    for col in ("velocity", "acceleration"):
        df[col] = df[col].map(_magnitude)
    df["marker_id"] = df["marker_id"].astype("Int64")
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


# pandas < 3's default CSV float parser can change the last digit of epoch
# timestamps (shifting samples by ~0.1 us and breaking exact matches);
# "round_trip" fixes that. pandas 3's default is already exact and deprecates
# the option, so it's only passed where it's needed.
_EXACT_FLOATS = {} if int(pd.__version__.split(".")[0]) >= 3 else {"float_precision": "round_trip"}


def read_emg(path: str | Path) -> pd.DataFrame:
    return (pd.read_csv(path, **_EXACT_FLOATS)
            .sort_values("timestamp", kind="stable").reset_index(drop=True))


def read_imu(path: str | Path) -> pd.DataFrame:
    return (pd.read_csv(path, **_EXACT_FLOATS)
            .sort_values("timestamp", kind="stable").reset_index(drop=True))


READERS = {"emg": read_emg, "imu": read_imu, "cvkas": read_cvkas}


def discover_trials(directory: str | Path) -> dict[str, dict[str, Path]]:
    """Group the CSVs in ``directory`` by trial key -> {modality: path}."""
    trials: dict[str, dict[str, Path]] = {}
    for path in sorted(Path(directory).glob("*.csv")):
        modality, _, key = path.stem.partition("-")
        if modality in MODALITIES and key:
            trials.setdefault(key, {})[modality] = path
    return trials


def load_trial(files: dict[str, Path]) -> dict[str, pd.DataFrame]:
    for m, p in files.items():
        check_format(p, m)
    return {m: READERS[m](p) for m, p in files.items()}
