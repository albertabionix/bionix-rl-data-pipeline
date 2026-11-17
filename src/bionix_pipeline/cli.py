"""Clean every trial in a folder of BionixDB CSVs (pipeline v1).

    python -m bionix_pipeline.cli data/ --out cleaned/
    python -m bionix_pipeline.cli --from-drive --pid 20 --cache data/ --out cleaned/

Writes, per trial key:
    cleaned/{key}.csv          aligned table on a uniform grid (default 100 Hz)
    cleaned/{key}.frames.csv   cleaned CVKAS at native frame times
and for the whole run:
    cleaned/manifest.csv       one row per trial: status, quality stats, flags
    cleaned/report.json        the same stats in full detail (nested)
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import PIPELINE_VERSION
from .align import build_table
from .cvkas import MARKERS, CvkasConfig, clean_cvkas
from .io import MODALITIES, UnsupportedFormat, discover_trials, load_trial, parse_key
from .sensors import emg_quality, imu_quality, imu_to_si

# Thresholds for the manifest's quality flags. A flag doesn't remove a trial;
# it tells you why you might want to filter it out.
EMG_MIN_RATE_HZ = 500       # firmware samples at 1 kHz
EMG_MAX_GAP_S = 0.2
IMU_MIN_RATE_HZ = 50        # firmware reads IMUs at 100 Hz
MIN_DURATION_S = 2.0


@dataclass
class RunConfig:
    out: Path
    rate_hz: float
    cvkas: CvkasConfig
    emg_window_s: float


def _flags(row: dict) -> list[str]:
    f = []
    if "emg" not in row["modalities"]:
        f.append("no_emg")
    else:
        if row["emg_rate_hz"] < EMG_MIN_RATE_HZ:
            f.append("emg_low_rate")
        if row["emg_gap_max_s"] > EMG_MAX_GAP_S:
            f.append("emg_long_gap")
        f += [f"emg_flat_{c}" for c in row["emg_flat_channels"].split(",") if c]
    if "imu" not in row["modalities"]:
        f.append("no_imu")
    else:
        if row["imu_units"] == "unknown":
            f.append("imu_no_data")
        elif row["imu_rate_hz"] < IMU_MIN_RATE_HZ:
            f.append("imu_low_rate")
    if "cvkas" not in row["modalities"]:
        f.append("no_cvkas")
    elif row["cvkas_frames"] == 0:
        f.append("no_cvkas_frames")
    else:
        if not row["required_markers"]:
            f.append("no_markers")
        elif row["usable_fraction"] == 0:
            f.append("no_usable_cvkas")
    if row["duration_s"] < MIN_DURATION_S:
        f.append("short")
    return f


def process_trial(key: str, files: dict[str, Path], cfg: RunConfig) -> tuple[dict, dict]:
    """Clean one trial; returns (manifest row, detailed report). Never raises."""
    row: dict = {"key": key, **parse_key(key), "pipeline_version": PIPELINE_VERSION,
                 "status": "ok", "error": "", "modalities": ",".join(sorted(files))}
    report: dict = {"modalities": sorted(files)}
    try:
        data = load_trial(files)

        frames = None
        if "cvkas" in data and len(data["cvkas"]) == 0:
            # CVKAS rows are only written while the camera runs; an empty file
            # means no camera data at all for this trial.
            report["cvkas"] = {"frames": 0}
            row["cvkas_frames"] = 0
        elif "cvkas" in data:
            res = clean_cvkas(data["cvkas"], cfg.cvkas)
            frames = res.frames
            frames.to_csv(cfg.out / f"{key}.frames.csv")
            c = report["cvkas"] = res.report
            row.update({
                "cvkas_frames": c["frames"],
                "cvkas_fps": c["frame_rate_hz"],
                **{f"cvkas_cov_m{m}": c["coverage_observed"][m] for m in cfg.cvkas.markers},
                "required_markers": ",".join(map(str, c["required_markers"])),
                "segments": c["segments"],
                "usable_fraction": c["usable_fraction"],
            })

        imu = None
        if "imu" in data:
            imu, units = imu_to_si(data["imu"])
            q = imu_quality(imu)
            report["imu"] = {"units_recorded": units, **q}
            row.update({"imu_units": units, **q})
            if units == "unknown":
                imu = None  # no usable values; leave IMU columns out

        if "emg" in data:
            q = emg_quality(data["emg"])
            report["emg"] = q
            row.update({**q, "emg_flat_channels": ",".join(q["emg_flat_channels"])})

        table = build_table(frames, imu, data.get("emg"), rate_hz=cfg.rate_hz,
                            emg_window_s=cfg.emg_window_s)
        table.to_csv(cfg.out / f"{key}.csv", index=False)

        row["duration_s"] = float(table["t"].iloc[-1]) if len(table) else 0.0
        row["grid_rows"] = len(table)
        for name, col in (("emg_coverage", "emg1_rms"), ("imu_coverage", "imu1_ax"),
                          ("knee_angle_coverage", "knee_angle"),
                          ("shank_angle_coverage", "shank_angle")):
            row[name] = float(table[col].notna().mean()) if col in table and len(table) else 0.0
        if "cvkas_valid" in table:
            row["grid_valid_fraction"] = float(table["cvkas_valid"].mean()) if len(table) else 0.0
            report["grid_valid_fraction"] = row["grid_valid_fraction"]
        row["quality_flags"] = ";".join(_flags({**_defaults(), **row}))
    except UnsupportedFormat as e:
        row.update(status="skipped", error=str(e))
    except Exception as e:  # one broken trial must not stop a run over hundreds
        row.update(status="error", error=f"{type(e).__name__}: {e}")
    report["status"], report["error"] = row["status"], row["error"]
    return row, report


def _defaults() -> dict:
    """Values _flags() reads, for modalities a trial doesn't have."""
    return {"emg_rate_hz": np.nan, "emg_gap_max_s": np.nan, "emg_flat_channels": "",
            "imu_units": "unknown", "imu_rate_hz": np.nan, "required_markers": "", "cvkas_frames": 0,
            "usable_fraction": 0.0, "duration_s": 0.0}


MANIFEST_COLUMNS = [
    "key", "pid", "action", "exercise", "trial", "pipeline_version", "status", "error", "quality_flags",
    "modalities", "duration_s", "grid_rows",
    "emg_rate_hz", "emg_gap_p95_s", "emg_gap_max_s", "emg_flat_channels", "emg_coverage",
    "imu_units", "imu_rate_hz", "imu_gap_max_s", "imu_coverage",
    "cvkas_frames", "cvkas_fps", *[f"cvkas_cov_m{m}" for m in MARKERS], "required_markers", "segments",
    "usable_fraction", "grid_valid_fraction", "knee_angle_coverage", "shank_angle_coverage",
]


def _run_one(args):
    return process_trial(*args)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, nargs="?", help="folder of BionixDB CSVs (omit with --from-drive)")
    p.add_argument("--out", type=Path, default=Path("cleaned"))
    p.add_argument("--rate", type=float, default=100.0, help="output grid rate in Hz (default 100)")
    p.add_argument("--max-gap", type=float, default=0.2, help="longest CVKAS gap to fill, seconds")
    p.add_argument("--min-segment", type=float, default=1.0, help="shortest kept valid stretch, seconds")
    p.add_argument("--require", type=str, default=None,
                   help="comma-separated marker ids that must be present, e.g. 2,3 "
                        "(default: markers with >=50%% observed coverage)")
    p.add_argument("--emg-window", type=float, default=0.05, help="EMG RMS window, seconds")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="trials processed in parallel")
    d = p.add_argument_group("download from the Shared Drive (needs: pip install -e '.[drive]')")
    d.add_argument("--from-drive", action="store_true", help="download recordings first, then clean them")
    d.add_argument("--cache", type=Path, default=Path("data"), help="where downloaded CSVs are kept")
    d.add_argument("--pid", help="only this participant, e.g. 20 or p020")
    d.add_argument("--action", help="only this action, e.g. stairs")
    d.add_argument("--exercise", help="only this exercise, e.g. stairmaster")
    d.add_argument("--token", type=Path, default=Path("token.json"), help="Google login token file")
    args = p.parse_args(argv)

    if args.from_drive:
        from .drive import fetch
        fetch(args.cache, pid=args.pid, action=args.action, exercise=args.exercise,
              token_file=args.token)
        source = args.cache
    elif args.input is None:
        p.error("give an input folder, or use --from-drive")
    else:
        source = args.input
    if args.out.resolve() == Path(source).resolve():
        p.error("--out must be a different folder from the input; raw files are never overwritten")

    cfg = RunConfig(
        out=args.out, rate_hz=args.rate, emg_window_s=args.emg_window,
        cvkas=CvkasConfig(
            max_gap_s=args.max_gap, min_segment_s=args.min_segment,
            required_markers=tuple(int(x) for x in args.require.split(",")) if args.require else None,
        ),
    )
    trials = discover_trials(source)
    if args.from_drive and (args.pid or args.action or args.exercise):
        trials = {k: v for k, v in trials.items() if _key_matches(k, args)}
    if not trials:
        p.error(f"no recordings found in {source}/")
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [(k, f, cfg) for k, f in trials.items()]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            results = list(pool.map(_run_one, jobs, chunksize=4))
    else:
        results = [_run_one(j) for j in jobs]

    rows, reports = [r for r, _ in results], {r["key"]: rep for r, rep in results}
    manifest = pd.DataFrame(rows).reindex(columns=MANIFEST_COLUMNS).sort_values("key")
    manifest.to_csv(args.out / "manifest.csv", index=False)
    (args.out / "report.json").write_text(json.dumps(reports, indent=2, default=float))

    counts = manifest["status"].value_counts().to_dict()
    print(f"{len(manifest)} trials -> {args.out}/: " + ", ".join(f"{n} {s}" for s, n in counts.items()))
    flags = manifest.loc[manifest.status == "ok", "quality_flags"].fillna("").str.split(";").explode()
    flags = flags[flags != ""].value_counts()
    if len(flags):
        print("flags: " + ", ".join(f"{f} ({n})" for f, n in flags.items()))
    for _, r in manifest[manifest.status != "ok"].head(10).iterrows():
        print(f"  {r.status}: {r.key}: {r.error}")
    return 0


def _key_matches(key: str, args) -> bool:
    k = parse_key(key)
    pid = args.pid and (f"p{int(args.pid):03d}" if str(args.pid).isdigit() else args.pid)
    return ((not pid or k["pid"] == pid) and (not args.action or k["action"] == args.action)
            and (not args.exercise or k["exercise"] == args.exercise))


if __name__ == "__main__":
    raise SystemExit(main())
