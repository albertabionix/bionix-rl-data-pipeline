"""EMG and IMU preprocessing.

EMG and IMU rows are written in pairs with identical host timestamps
(~1 kHz). The IMU updates more slowly (~100 Hz), so IMU values repeat for
several rows; only rows where a value changes are real samples.

IMU output is always SI (accel m/s^2, gyro rad/s); see ``imu_to_si``. Gyro
values are not bias-corrected: v1 recordings stored uncalibrated values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IMU_ACCEL = [f"imu{i}_{a}" for i in (1, 2) for a in ("ax", "ay", "az")]
IMU_GYRO = [f"imu{i}_{a}" for i in (1, 2) for a in ("gx", "gy", "gz")]

# ISM330DHCX settings used by the firmware (arduino/ESP-32.txt in
# EMG-and-IMU-Data-Collection): accel +/-4 g = 0.122 mg/LSB, gyro +/-500 dps = 17.5 mdps/LSB.
G = 9.80665
ACCEL_MS2_PER_COUNT = 0.122e-3 * G
GYRO_RADS_PER_COUNT = np.deg2rad(17.5e-3)


def imu_to_si(imu: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return the IMU table in SI units (accel m/s^2, gyro rad/s) and the
    units the file was recorded in.

    v1 data has two formats: participants p002-p005 were recorded by firmware
    that already sent m/s^2 and rad/s (Adafruit unified-sensor units), later
    ones by the binary-packet firmware, which sends raw counts. The format is
    detected per file from the accelerometer magnitude at rest.
    """
    units = imu_accel_units(imu)
    out = imu.copy()
    if units == "raw_counts":
        cols_a = [c for c in IMU_ACCEL if c in out]
        cols_g = [c for c in IMU_GYRO if c in out]
        out[cols_a] = out[cols_a].astype(float) * ACCEL_MS2_PER_COUNT
        out[cols_g] = out[cols_g].astype(float) * GYRO_RADS_PER_COUNT
    return out, units


def imu_quality(imu: pd.DataFrame) -> dict:
    """Rate of real (changing) IMU samples and the longest gap between them."""
    s = imu_samples(imu)
    t = s["timestamp"].to_numpy(float)
    span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    return {
        "imu_rate_hz": len(t) / span if span else float("nan"),
        "imu_gap_max_s": float(np.diff(t).max()) if len(t) > 1 else float("nan"),
    }


def emg_quality(emg: pd.DataFrame, channels=("emg1", "emg2")) -> dict:
    """Effective EMG sample rate, the longest gap between samples, and which
    channels look dead (constant). v1 timestamps are host arrival times, so
    samples arrive in bursts; the gap shows how bursty the trial was."""
    t = emg["timestamp"].to_numpy(float)
    span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    dt = np.diff(t)
    return {
        "emg_rate_hz": len(t) / span if span else float("nan"),
        "emg_gap_p95_s": float(np.percentile(dt, 95)) if len(dt) else float("nan"),
        "emg_gap_max_s": float(dt.max()) if len(dt) else float("nan"),
        "emg_flat_channels": [c for c in channels if c in emg and emg[c].nunique() <= 1],
    }


def imu_samples(imu: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where any IMU channel changed (plus the first row)."""
    vals = imu.drop(columns="timestamp")
    changed = vals.diff().abs().sum(axis=1).ne(0)
    changed.iloc[0] = True
    return imu[changed].reset_index(drop=True)


def imu_accel_units(imu: pd.DataFrame) -> str:
    """Two firmware/app versions exist in the data: accelerometer already in
    m/s^2 (|a| ~ 9.8 at rest) or raw sensor counts (|a| ~ 8200 at rest).
    "unknown" if there's no usable accelerometer data at all."""
    cols = [c for c in ("imu1_ax", "imu1_ay", "imu1_az") if c in imu]
    if not cols:
        return "unknown"
    mag = np.linalg.norm(imu[cols].to_numpy(dtype=float), axis=1)
    if not np.isfinite(mag).any():
        return "unknown"
    return "m/s^2" if np.nanmedian(mag) < 100 else "raw_counts"


def emg_envelope(emg: pd.DataFrame, grid_t: np.ndarray, window_s: float = 0.05,
                 channels=("emg1", "emg2")) -> pd.DataFrame:
    """Causal RMS envelope of each EMG channel over (t - window_s, t], with
    the per-trial mean removed as the DC offset. Windows with no samples
    (host stalls) give NaN."""
    t = emg["timestamp"].to_numpy()
    lo = np.searchsorted(t, grid_t - window_s, side="right")
    hi = np.searchsorted(t, grid_t, side="right")
    n = hi - lo
    out = {}
    for ch in channels:
        x = emg[ch].to_numpy(dtype=float)
        x = x - np.mean(x)
        c = np.concatenate([[0.0], np.cumsum(x * x)])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{ch}_rms"] = np.where(n > 0, np.sqrt((c[hi] - c[lo]) / n), np.nan)
    return pd.DataFrame(out)
