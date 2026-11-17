"""Put CVKAS, IMU and EMG for one trial onto a single uniform time grid."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sensors import emg_envelope, imu_samples


def _interp_no_bridge(t_src: np.ndarray, y: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Linear interpolation that returns NaN unless both bracketing source
    samples are finite, so NaN stretches are never bridged."""
    out = np.full(len(t_dst), np.nan)
    j = np.searchsorted(t_src, t_dst, side="right")
    ok = (j > 0) & (j < len(t_src))
    a, b = j[ok] - 1, j[ok]
    ya, yb = y[a], y[b]
    fin = np.isfinite(ya) & np.isfinite(yb)
    w = (t_dst[ok] - t_src[a]) / (t_src[b] - t_src[a])
    vals = ya + w * (yb - ya)
    vals[~fin] = np.nan
    out[ok] = vals
    exact = np.isin(t_dst, t_src)  # grid point landing exactly on a frame
    if exact.any():
        out[exact] = y[np.searchsorted(t_src, t_dst[exact])]
    return out


def build_table(cvkas_frames: pd.DataFrame | None, imu: pd.DataFrame | None,
                emg: pd.DataFrame | None, rate_hz: float = 50.0,
                imu_tolerance_s: float = 0.05, emg_window_s: float = 0.05) -> pd.DataFrame:
    """One row per grid step over the time all given modalities overlap.

    - CVKAS numeric columns: linear interpolation between frames, never across NaN.
      ``segment_id`` is kept only where both bracketing frames share a segment.
    - IMU: latest real sample at or before t (within ``imu_tolerance_s``).
    - EMG: causal RMS envelope over ``emg_window_s``.
    """
    spans = []
    if cvkas_frames is not None:
        spans.append((cvkas_frames.index[0], cvkas_frames.index[-1]))
    for df in (imu, emg):
        if df is not None:
            spans.append((df["timestamp"].iloc[0], df["timestamp"].iloc[-1]))
    t0, t1 = max(s[0] for s in spans), min(s[1] for s in spans)
    grid = t0 + np.arange(0, max(t1 - t0, 0) + 1e-9, 1.0 / rate_hz)
    table = pd.DataFrame({"timestamp": grid, "t": grid - grid[0]})

    if cvkas_frames is not None:
        f = cvkas_frames
        tf = f.index.to_numpy()
        numeric = [c for c in f.columns
                   if c not in ("segment_id", "cvkas_valid") and not c.endswith("_src")]
        for c in numeric:
            table[c] = _interp_no_bridge(tf, f[c].to_numpy(dtype=float), grid)
        seg = f["segment_id"].to_numpy()
        j = np.clip(np.searchsorted(tf, grid, side="right"), 1, len(tf) - 1)
        prev_seg, next_seg = seg[j - 1], seg[j]
        on_frame = np.isin(grid, tf)
        sid = np.where((prev_seg == next_seg) | on_frame, prev_seg, -1)
        table["segment_id"] = sid
        table["cvkas_valid"] = sid >= 0

    if imu is not None:
        s = imu_samples(imu)
        table = pd.merge_asof(table, s, on="timestamp", direction="backward",
                              tolerance=imu_tolerance_s)

    if emg is not None:
        env = emg_envelope(emg, grid, emg_window_s)
        table = pd.concat([table, env], axis=1)

    return table
