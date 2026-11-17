"""CVKAS cleaning: per-frame marker table, short-gap filling, kinematics,
joint angles, and validity segments.

Raw CVKAS has one row per detected marker per camera frame (~26 Hz, irregular),
and a row with only ``timestamp`` when nothing was detected. A missing marker
is therefore an *absent row*, so everything here works on a wide per-frame
table: one row per frame timestamp, ``m{id}_x`` / ``m{id}_y`` per marker.

The recorded velocity/acceleration (Kalman state, reset after 0.5 s gaps,
spiking on re-acquisition) and the recorded angles (only present when all
four markers are seen at once) are discarded and recomputed from the cleaned
positions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .compress import compress_cvkas

# Current EMG-and-IMU-Data-Collection layout (cv_processor.py):
#   0 = thigh proximal (hip), 1 = thigh distal (knee),
#   2 = shank proximal (knee), 3 = shank distal (ankle)
MARKERS = (0, 1, 2, 3)
VERTICAL = np.array([0.0, -1.0])  # image "up" (y grows downward)

MISSING, OBSERVED, INTERPOLATED = 0, 1, 2


@dataclass
class CvkasConfig:
    # Gaps whose total duration (last detection -> next detection) is at most
    # this are filled. Measured on p020 stairs: ~190 ms (5 frames) keeps p90
    # position error around 4 px against ~60 px range of motion.
    max_gap_s: float = 0.2
    # Markers that must be present (observed or filled) for a frame to count
    # as valid. None = every marker with >= auto_require_coverage observed.
    required_markers: tuple[int, ...] | None = None
    auto_require_coverage: float = 0.5
    # Valid stretches shorter than this are dropped (not useful as episodes).
    min_segment_s: float = 1.0
    markers: tuple[int, ...] = field(default=MARKERS)


def to_frames(raw: pd.DataFrame, markers=MARKERS) -> pd.DataFrame:
    """Raw long CVKAS rows -> one row per frame with m{id}_x / m{id}_y
    (positions only; see ``compress.compress_cvkas`` for the full pivot)."""
    wide = compress_cvkas(raw, markers=markers, on_duplicate="last")
    cols = [f"m{m}_{ax}" for m in markers for ax in ("x", "y")]
    return wide[cols].astype(float)


def coverage(wide: pd.DataFrame, markers=MARKERS) -> dict[int, float]:
    return {m: float(wide[f"m{m}_x"].notna().mean()) for m in markers}


def gap_lengths(seen: np.ndarray, t: np.ndarray) -> list[float]:
    """Durations (s) between consecutive detections that have missing frames
    between them. Leading/trailing missing stretches are not counted."""
    idx = np.flatnonzero(seen)
    return [float(t[b] - t[a]) for a, b in zip(idx[:-1], idx[1:]) if b - a > 1]


def fill_gaps(wide: pd.DataFrame, max_gap_s: float, markers=MARKERS) -> pd.DataFrame:
    """PCHIP-fill interior gaps no longer than ``max_gap_s``. Adds an int8
    ``m{id}_src`` column: 0 missing, 1 observed, 2 interpolated. PCHIP is
    used over a cubic spline so fills never overshoot the observed motion."""
    out = wide.copy()
    t = out.index.to_numpy() - out.index[0]
    for m in markers:
        xc, yc = f"m{m}_x", f"m{m}_y"
        seen = out[xc].notna().to_numpy()
        src = np.where(seen, OBSERVED, MISSING).astype(np.int8)
        idx = np.flatnonzero(seen)
        fill = np.zeros(len(t), dtype=bool)
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > 1 and t[b] - t[a] <= max_gap_s:
                fill[a + 1 : b] = True
        if fill.any():
            for col in (xc, yc):
                f = PchipInterpolator(t[idx], out[col].to_numpy()[idx])
                out.loc[fill, col] = f(t[fill])
            src[fill] = INTERPOLATED
        out[f"m{m}_src"] = src
    return out


def add_kinematics(wide: pd.DataFrame, markers=MARKERS) -> pd.DataFrame:
    """Velocity (px/s) per axis plus speed, from cleaned positions using
    non-uniform central differences. NaN at the edge of each valid run, so
    no derivative is ever taken across a gap."""
    out = wide.copy()
    t = out.index.to_numpy() - out.index[0]
    if len(t) < 3:
        return out
    for m in markers:
        for ax in ("x", "y"):
            out[f"m{m}_v{ax}"] = np.gradient(out[f"m{m}_{ax}"].to_numpy(), t)
        out[f"m{m}_speed"] = np.hypot(out[f"m{m}_vx"], out[f"m{m}_vy"])
    return out


def _angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Signed angle v1 -> v2 in degrees, same convention as cv_processor."""
    dot = (v1 * v2).sum(axis=-1)
    det = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    return np.degrees(np.arctan2(det, dot))


def add_angles(wide: pd.DataFrame) -> pd.DataFrame:
    """thigh = m0->m1, shank = m2->m3. hip = vertical->thigh,
    knee = thigh->shank (as in the app), plus shank = vertical->shank, which
    only needs markers 2 and 3 and so survives a missing hip marker."""
    out = wide.copy()
    p = {m: out[[f"m{m}_x", f"m{m}_y"]].to_numpy() for m in MARKERS}
    thigh, shank = p[1] - p[0], p[3] - p[2]
    vert = np.broadcast_to(VERTICAL, thigh.shape)
    out["hip_angle"] = _angle(vert, thigh)
    out["knee_angle"] = _angle(thigh, shank)
    out["shank_angle"] = _angle(vert, shank)
    return out


def segment(valid: np.ndarray, t: np.ndarray, min_segment_s: float) -> np.ndarray:
    """Label contiguous valid runs lasting >= min_segment_s with 0, 1, ...;
    everything else gets -1."""
    seg = np.full(len(valid), -1, dtype=np.int32)
    next_id = 0
    i = 0
    while i < len(valid):
        if not valid[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(valid) and valid[j + 1]:
            j += 1
        if t[j] - t[i] >= min_segment_s:
            seg[i : j + 1] = next_id
            next_id += 1
        i = j + 1
    return seg


@dataclass
class CvkasResult:
    frames: pd.DataFrame          # cleaned per-frame table
    required: tuple[int, ...]
    report: dict


def clean_cvkas(raw: pd.DataFrame, cfg: CvkasConfig | None = None) -> CvkasResult:
    cfg = cfg or CvkasConfig()
    wide = to_frames(raw, cfg.markers)
    t = wide.index.to_numpy()
    cov = coverage(wide, cfg.markers)

    required = cfg.required_markers
    if required is None:
        required = tuple(m for m in cfg.markers if cov[m] >= cfg.auto_require_coverage)

    gaps = {m: gap_lengths(wide[f"m{m}_x"].notna().to_numpy(), t) for m in cfg.markers}
    wide = fill_gaps(wide, cfg.max_gap_s, cfg.markers)
    wide = add_kinematics(wide, cfg.markers)
    wide = add_angles(wide)

    if required:
        valid = np.all([wide[f"m{m}_src"].to_numpy() != MISSING for m in required], axis=0)
    else:
        valid = np.zeros(len(wide), dtype=bool)
    wide["cvkas_valid"] = valid
    wide["segment_id"] = segment(valid, t - t[0], cfg.min_segment_s)

    seg = wide["segment_id"].to_numpy()
    seg_durations = [float(t[seg == s][-1] - t[seg == s][0]) for s in np.unique(seg[seg >= 0])]
    span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    report = {
        "frames": int(len(t)),
        "span_s": span,
        "frame_rate_hz": len(t) / span if span else float("nan"),
        "max_frame_dt_s": float(np.diff(t).max()) if len(t) > 1 else float("nan"),
        "coverage_observed": cov,
        "coverage_after_fill": {
            m: float((wide[f"m{m}_src"] != MISSING).mean()) for m in cfg.markers
        },
        "gaps": {
            m: {
                "count": len(g),
                "filled": int(sum(d <= cfg.max_gap_s for d in g)),
                "max_s": max(g) if g else 0.0,
            }
            for m, g in gaps.items()
        },
        "required_markers": list(required),
        "segments": len(seg_durations),
        "segment_time_s": float(sum(seg_durations)),
        "usable_fraction": float(sum(seg_durations) / span) if span else 0.0,
        "angles_available": {
            "hip/knee (needs 0,1,2,3)": float(wide["knee_angle"].notna().mean()),
            "shank (needs 2,3)": float(wide["shank_angle"].notna().mean()),
        },
    }
    return CvkasResult(frames=wide, required=required, report=report)
