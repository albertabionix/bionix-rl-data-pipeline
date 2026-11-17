"""Pipeline v1: IMU units, quality stats, format checks, manifest, Drive filter."""
import numpy as np
import pandas as pd
import pytest

from bionix_pipeline import PIPELINE_VERSION
from bionix_pipeline.cli import MANIFEST_COLUMNS, main, process_trial, RunConfig
from bionix_pipeline.cvkas import CvkasConfig
from bionix_pipeline.drive import _matches
from bionix_pipeline.io import UnsupportedFormat, check_format, parse_key
from bionix_pipeline.sensors import (ACCEL_MS2_PER_COUNT, GYRO_RADS_PER_COUNT, emg_quality,
                                     imu_quality, imu_to_si)

IMU_COLS = [f"imu{i}_{a}" for i in (1, 2) for a in ("ax", "ay", "az", "gx", "gy", "gz")]


# ---------- synthetic v1 recordings, shaped like the real files ----------

def v1_trial(tmp_path, key="p020-stairs-stairmaster-03", seconds=4.0, raw_counts=True,
             markers=(2, 3)):
    """Write emg/imu/cvkas CSVs for one trial in the v1 layout."""
    t0 = 1788727215.0
    t = t0 + np.arange(0, seconds, 0.001)                      # EMG/IMU rows ~1 kHz
    emg = pd.DataFrame({"timestamp": t,
                        "emg1": 1550 + (np.sin(2 * np.pi * 3 * t) * 80).astype(int),
                        "emg2": 1600 + (np.cos(2 * np.pi * 2 * t) * 60).astype(int)})
    held = np.floor((t - t0) * 100) / 100                       # IMU refreshes at 100 Hz
    one_g = 8197 if raw_counts else 9.80665
    imu = pd.DataFrame({"timestamp": t})
    for i in (1, 2):
        imu[f"imu{i}_ax"] = np.round(0.05 * one_g * np.sin(2 * np.pi * held), 2)
        imu[f"imu{i}_ay"] = one_g
        imu[f"imu{i}_az"] = 0.0
        imu[f"imu{i}_gx"] = np.round((573 if raw_counts else 0.1745) * np.cos(2 * np.pi * held), 4)
        imu[f"imu{i}_gy"] = 0.0
        imu[f"imu{i}_gz"] = 0.0
    rows = []
    for f in t0 + np.arange(0, seconds, 0.039):                 # ~26 fps camera
        for m in markers:
            rows.append((f, m, 200 + 10 * m + 5 * np.sin(f), 100 + 40 * m, 1.0, 2.0, np.nan, np.nan))
        if not markers:  # like the app: a timestamp-only row for a frame with no detections
            rows.append((f, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
    cv = pd.DataFrame(rows, columns=["timestamp", "marker_id", "pos_x", "pos_y", "velocity",
                                     "acceleration", "hip_angle", "knee_angle"])
    files = {}
    for m, df in (("emg", emg), ("imu", imu), ("cvkas", cv)):
        p = tmp_path / f"{m}-{key}.csv"
        df.to_csv(p, index=False)
        files[m] = p
    return files


def cfg(tmp_path):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return RunConfig(out=out, rate_hz=100.0, cvkas=CvkasConfig(), emg_window_s=0.05)


# ---------- IMU units ----------

def test_raw_counts_converted_to_si():
    imu = pd.DataFrame({"timestamp": [0.0, 0.01], **{c: [0.0, 0.0] for c in IMU_COLS}})
    imu["imu1_ay"] = 8197.0
    imu["imu1_gx"] = 57.142857  # 1 degree/s
    si, units = imu_to_si(imu)
    assert units == "raw_counts"
    assert si["imu1_ay"].iloc[0] == pytest.approx(9.80665, rel=1e-3)
    assert si["imu1_gx"].iloc[0] == pytest.approx(np.deg2rad(1.0), rel=1e-4)
    assert imu["imu1_ay"].iloc[0] == 8197.0  # input not modified


def test_si_files_left_unchanged():
    imu = pd.DataFrame({"timestamp": [0.0], **{c: [0.0] for c in IMU_COLS}})
    imu["imu1_ay"], imu["imu1_gx"] = 9.4, 0.02
    si, units = imu_to_si(imu)
    assert units == "m/s^2"
    pd.testing.assert_frame_equal(si, imu)


def test_all_nan_imu_is_unknown_not_an_error():
    imu = pd.DataFrame({"timestamp": [0.0, 0.01], **{c: [np.nan, np.nan] for c in IMU_COLS}})
    assert imu_to_si(imu)[1] == "unknown"


def test_conversion_constants_match_firmware_ranges():
    assert 1 / (ACCEL_MS2_PER_COUNT / 9.80665) == pytest.approx(8197, rel=1e-3)   # +/-4 g
    assert np.rad2deg(GYRO_RADS_PER_COUNT) == pytest.approx(0.0175)              # +/-500 dps


# ---------- quality stats ----------

def test_emg_quality_rate_gaps_and_flat_channels():
    t = np.concatenate([np.arange(0, 1, 0.001), 1.5 + np.arange(0, 1, 0.001)])  # 0.5 s gap
    emg = pd.DataFrame({"timestamp": t, "emg1": np.arange(len(t)) % 50, "emg2": 0})
    q = emg_quality(emg)
    assert q["emg_rate_hz"] == pytest.approx(len(t) / t[-1], rel=1e-6)
    assert q["emg_gap_max_s"] == pytest.approx(0.501, abs=1e-6)
    assert q["emg_flat_channels"] == ["emg2"]


def test_imu_quality_counts_only_changing_samples():
    t = np.arange(0, 1, 0.001)
    imu = pd.DataFrame({"timestamp": t, **{c: 0.0 for c in IMU_COLS}})
    imu["imu1_ax"] = np.floor(t * 100)  # changes every 10 ms
    assert imu_quality(imu)["imu_rate_hz"] == pytest.approx(100, rel=0.02)


# ---------- formats and keys ----------

def test_check_format_accepts_v1_and_refuses_newer_formats(tmp_path):
    files = v1_trial(tmp_path)
    for m, p in files.items():
        check_format(p, m)  # no error
    v2 = tmp_path / "emg-new.csv"
    v2.write_text("timestamp,seq,emg1,emg2\n1.0,1,1500,1500\n")
    with pytest.raises(UnsupportedFormat, match="newer recording format"):
        check_format(v2, "emg")
    wide = tmp_path / "cvkas-new.csv"
    wide.write_text("timestamp,pos_x_0,pos_y_0,velocity_0,acceleration_0,hip_angle,knee_angle\n")
    with pytest.raises(UnsupportedFormat):
        check_format(wide, "cvkas")


def test_parse_key_with_and_without_exercise():
    assert parse_key("p020-stairs-stairmaster-03") == {
        "pid": "p020", "action": "stairs", "exercise": "stairmaster", "trial": 3}
    assert parse_key("p002-seated-01") == {
        "pid": "p002", "action": "seated", "exercise": None, "trial": 1}


def test_drive_filter_only_csvs_and_matching_trials():
    assert _matches("emg-p020-stairs-stairmaster-03.csv", "20", "stairs", None)
    assert _matches("emg-p020-stairs-stairmaster-03.csv", "p020", None, "stairmaster")
    assert not _matches("emg-p006-seated-01", None, None, None)       # Google Sheets copy
    assert not _matches("emg-p021-stairs-01.csv", "20", None, None)
    assert not _matches("emg-p020-walking-01.csv", None, "stairs", None)


# ---------- whole trials ----------

def test_process_trial_outputs_si_units_100hz_and_a_manifest_row(tmp_path):
    c = cfg(tmp_path)
    row, report = process_trial("p020-stairs-stairmaster-03", v1_trial(tmp_path), c)
    assert row["status"] == "ok", row["error"]
    assert row["pipeline_version"] == PIPELINE_VERSION
    assert row["imu_units"] == "raw_counts"
    assert row["imu_rate_hz"] == pytest.approx(100, rel=0.05)
    assert row["required_markers"] == "2,3"
    assert "emg_low_rate" not in row["quality_flags"]
    table = pd.read_csv(c.out / "p020-stairs-stairmaster-03.csv")
    assert np.allclose(np.diff(table["t"]), 0.01, atol=1e-6)
    assert table["imu1_ay"].median() == pytest.approx(9.80665, rel=1e-3)  # m/s^2, not counts
    assert table["shank_angle"].notna().mean() > 0.9


def test_old_firmware_si_trial_gets_same_units(tmp_path):
    c = cfg(tmp_path)
    row, _ = process_trial("p002-seated-01", v1_trial(tmp_path, "p002-seated-01", raw_counts=False), c)
    assert row["status"] == "ok" and row["imu_units"] == "m/s^2"
    table = pd.read_csv(c.out / "p002-seated-01.csv")
    assert table["imu1_ay"].median() == pytest.approx(9.80665, rel=1e-3)


def test_newer_format_trial_is_skipped_not_misread(tmp_path):
    files = v1_trial(tmp_path)
    files["emg"].write_text("timestamp,seq,emg1,emg2\n1.0,1,1500,1500\n")
    row, _ = process_trial("p020-stairs-stairmaster-03", files, cfg(tmp_path))
    assert row["status"] == "skipped" and "pipeline v1" in row["error"]


def test_empty_cvkas_file_means_no_camera_frames(tmp_path):
    files = v1_trial(tmp_path)
    files["cvkas"].write_text("timestamp,marker_id,pos_x,pos_y,velocity,acceleration,hip_angle,knee_angle\n")
    row, _ = process_trial("p020-stairs-stairmaster-03", files, cfg(tmp_path))
    assert row["status"] == "ok" and "no_cvkas_frames" in row["quality_flags"]
    assert row["imu_coverage"] > 0.9  # sensors still cleaned


def test_broken_trial_reported_as_error_not_raised(tmp_path):
    files = v1_trial(tmp_path)
    files["imu"].write_text(",".join(["timestamp"] + IMU_COLS) + "\n")  # header only
    row, report = process_trial("p020-stairs-stairmaster-03", files, cfg(tmp_path))
    assert row["status"] == "error" and report["status"] == "error"


def test_cli_writes_manifest_for_every_trial(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    v1_trial(data, "p020-stairs-stairmaster-03")
    v1_trial(data, "p002-seated-01", raw_counts=False, markers=())   # no markers at all
    bad = v1_trial(data, "p021-walking-01")
    bad["cvkas"].write_text("timestamp,pos_x_0\n")                   # newer format
    out = tmp_path / "out"
    assert main([str(data), "--out", str(out), "--jobs", "1"]) == 0
    man = pd.read_csv(out / "manifest.csv")
    assert list(man.columns) == MANIFEST_COLUMNS
    assert man.set_index("key")["status"].to_dict() == {
        "p002-seated-01": "ok", "p020-stairs-stairmaster-03": "ok", "p021-walking-01": "skipped"}
    assert "no_markers" in man.set_index("key").loc["p002-seated-01", "quality_flags"]
    assert (out / "report.json").exists() and (out / "p020-stairs-stairmaster-03.csv").exists()


def test_cli_refuses_to_write_into_the_input_folder(tmp_path):
    v1_trial(tmp_path)
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--out", str(tmp_path)])
