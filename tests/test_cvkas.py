import io as stdio

import numpy as np
import pandas as pd

from bionix_pipeline.align import build_table
from bionix_pipeline.cvkas import (INTERPOLATED, MISSING, OBSERVED, CvkasConfig,
                                   clean_cvkas, fill_gaps, segment, to_frames)
from bionix_pipeline.io import _magnitude, read_cvkas
from bionix_pipeline.sensors import emg_envelope, imu_accel_units, imu_samples

DT = 0.04


def synthetic_cvkas(n=200, drop3=(), drop2=()):
    """Markers 2 and 3 moving sinusoidally; frames in drop* lose that marker."""
    rows = []
    for i in range(n):
        t = 1000 + i * DT
        any_row = False
        for m, drop in ((2, drop2), (3, drop3)):
            if i in drop:
                continue
            x = 100 + 30 * np.sin(2 * np.pi * t / 1.8) + 10 * m
            y = 150 + 20 * np.cos(2 * np.pi * t / 1.8) + 20 * m
            rows.append((t, m, x, y))
            any_row = True
        if not any_row:
            rows.append((t, np.nan, np.nan, np.nan))
    df = pd.DataFrame(rows, columns=["timestamp", "marker_id", "pos_x", "pos_y"])
    df["marker_id"] = df["marker_id"].astype("Int64")
    for c in ("velocity", "acceleration", "hip_angle", "knee_angle"):
        df[c] = np.nan
    return df


def test_magnitude_handles_scalar_vector_and_empty():
    assert _magnitude(5.0) == 5.0
    assert _magnitude("[3.0, 4.0]") == 5.0
    assert np.isnan(_magnitude(np.nan))
    assert np.isnan(_magnitude(""))


def test_read_cvkas_parses_vector_strings():
    csv = ("timestamp,marker_id,pos_x,pos_y,velocity,acceleration,hip_angle,knee_angle\n"
           '1.0,2,10,20,"[3.0, 4.0]","[6.0, 8.0]",,\n'
           "1.04,,,,,,,\n")
    df = read_cvkas(stdio.StringIO(csv))
    assert df.loc[0, "velocity"] == 5.0 and df.loc[0, "acceleration"] == 10.0
    assert df["marker_id"].isna().iloc[1]


def test_to_frames_keeps_empty_frames():
    wide = to_frames(synthetic_cvkas(n=10, drop2=range(10), drop3={4}))
    assert len(wide) == 10                     # frame 4 had no detections at all
    assert wide["m3_x"].isna().sum() == 1
    assert wide["m2_x"].isna().all()


def test_short_gaps_filled_long_gaps_left():
    raw = synthetic_cvkas(drop3=set(range(20, 22)) | set(range(100, 115)))
    wide = fill_gaps(to_frames(raw), max_gap_s=0.2)
    src = wide["m3_src"].to_numpy()
    assert (src[20:22] == INTERPOLATED).all()
    assert (src[100:115] == MISSING).all()
    assert (src[:20] == OBSERVED).all()
    # filled values close to the true sinusoid
    true = to_frames(synthetic_cvkas())["m3_x"].to_numpy()
    assert np.abs(wide["m3_x"].to_numpy()[20:22] - true[20:22]).max() < 0.5


def test_edges_never_extrapolated():
    wide = fill_gaps(to_frames(synthetic_cvkas(drop3=set(range(0, 2)))), max_gap_s=1.0)
    assert (wide["m3_src"].to_numpy()[:2] == MISSING).all()


def test_segment_min_length():
    valid = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=bool)
    t = np.arange(10) * 0.1
    seg = segment(valid, t, min_segment_s=0.3)
    assert (seg[:3] == -1).all() and (seg[4:] == 0).all()


def test_clean_cvkas_splits_at_long_gap_and_computes_shank_angle():
    res = clean_cvkas(synthetic_cvkas(drop3=set(range(100, 115))),
                      CvkasConfig(required_markers=(2, 3)))
    f = res.frames
    assert res.report["segments"] == 2
    assert f["segment_id"].iloc[50] != f["segment_id"].iloc[150]
    assert (f["segment_id"].iloc[100:115] == -1).all()
    assert f["shank_angle"].notna().mean() > 0.9
    assert f["knee_angle"].isna().all()           # no thigh markers
    # recomputed speed is finite inside segments, NaN inside the gap
    assert np.isfinite(f["m3_speed"].iloc[50])
    assert np.isnan(f["m3_speed"].iloc[105])


def test_auto_required_markers_uses_coverage():
    res = clean_cvkas(synthetic_cvkas(drop2=set(range(0, 150))))
    assert res.required == (3,)


def test_imu_dedup_and_units():
    imu = pd.DataFrame({"timestamp": [0, .001, .002, .011],
                        "imu1_ax": [-5962, -5962, -5962, -6315],
                        "imu1_ay": [398, 398, 398, -173], "imu1_az": [-1978] * 4})
    assert len(imu_samples(imu)) == 2
    assert imu_accel_units(imu) == "raw_counts"
    assert imu_accel_units(imu.assign(imu1_ax=0.1, imu1_ay=9.4, imu1_az=1.6)) == "m/s^2"


def test_emg_envelope_rms():
    t = np.arange(0, 1, 0.001)
    emg = pd.DataFrame({"timestamp": t, "emg1": 1300 + 100 * np.sign(np.sin(2 * np.pi * 50 * t)),
                        "emg2": np.full_like(t, 1300)})
    env = emg_envelope(emg, np.array([0.5, 2.0]), 0.05)
    assert abs(env["emg1_rms"][0] - 100) < 5
    assert np.isnan(env["emg1_rms"][1])


def test_build_table_does_not_bridge_gaps():
    res = clean_cvkas(synthetic_cvkas(drop3=set(range(100, 115))),
                      CvkasConfig(required_markers=(2, 3)))
    table = build_table(res.frames, None, None, rate_hz=50)
    gap = (table["timestamp"] > 1000 + 100 * DT) & (table["timestamp"] < 1000 + 114 * DT)
    assert table.loc[gap, "m3_x"].isna().all()
    assert not table.loc[gap, "cvkas_valid"].any()
    assert table["cvkas_valid"].mean() > 0.8
