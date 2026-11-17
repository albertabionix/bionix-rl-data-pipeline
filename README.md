# bionix-rl-data-pipeline
Pipeline for handling DCU data collection. This will perform data preprocessing, interpolation, and formatting for the Reinforcement Learning algorithm.

It takes the raw EMG, IMU and CVKAS recordings that [EMG-and-IMU-Data-Collection](https://github.com/albertabionix/EMG-and-IMU-Data-Collection) uploads to the Alberta Bionix Shared Drive (through [BionixDB](https://github.com/albertabionix/bionix-db)), and turns each trial into one clean, time-aligned table.

**This is pipeline v1.** It handles the recordings made so far. Newer recordings, from app versions that add a `seq` column to EMG/IMU or write CVKAS one row per frame, are a different format: v1 detects them and skips them (`status = skipped` in the manifest) instead of misreading them. A later pipeline version will handle them.

## Setup
```
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
```
To download recordings from the Shared Drive as well, install the `drive` extra (needs `git`):
```
.venv/bin/pip install -e '.[dev,drive]'
```

## Running it

**From the Shared Drive** (opens a Google login in your browser the first time; the token is saved to `token.json`, which is git-ignored):
```
.venv/bin/python -m bionix_pipeline.cli --from-drive --out cleaned/                          # everything
.venv/bin/python -m bionix_pipeline.cli --from-drive --pid 20 --action stairs --out cleaned/  # a subset
```
Recordings are downloaded unchanged into `--cache` (default `data/`, git-ignored) and then cleaned. Running it again only downloads files that are new or changed. Filters: `--pid` (`20` or `p020`), `--action`, `--exercise`. Only real CSV files are downloaded. The Drive's dataset folders also hold a few Google Sheets copies of recordings, which are skipped.

**From a local folder** of BionixDB CSVs (e.g. from `db.download("emg", "data/")`, and the same for `imu` and `cvkas`):
```
.venv/bin/python -m bionix_pipeline.cli data/ --out cleaned/
```

Options: `[--rate 100] [--max-gap 0.2] [--min-segment 1.0] [--require 2,3] [--emg-window 0.05] [--jobs N]`. Trials are cleaned in parallel. The whole v1 dataset (957 trials) takes under 10 seconds. Raw files are only ever read; `--out` must be a different folder from the input.

## Output

| File | Contents |
|---|---|
| `cleaned/{key}.csv` | The trial on a uniform grid (default 100 Hz): CVKAS, IMU and EMG aligned |
| `cleaned/{key}.frames.csv` | Cleaned CVKAS at the camera's own frame times (no resampling) |
| `cleaned/manifest.csv` | One row per trial: status, quality stats and flags. Open it to choose which trials to use |
| `cleaned/report.json` | The same stats in full detail |

`{key}` is the BionixDB name without the modality, e.g. `p020-stairs-stairmaster-03`.

### Aligned table columns (`{key}.csv`)

| Columns | Meaning | Units |
|---|---|---|
| `timestamp`, `t` | Grid time: Unix epoch seconds, and seconds since the start of the trial | s |
| `m{0-3}_x`, `m{0-3}_y` | Cleaned marker centres (0 thigh proximal, 1 thigh distal, 2 shank proximal, 3 shank distal) | image px |
| `m{0-3}_vx`, `_vy`, `_speed` | Marker velocity, recomputed from cleaned positions | px/s |
| `hip_angle`, `knee_angle` | As the app defines them; need all four markers | degrees |
| `shank_angle` | Shank vs. vertical; needs only markers 2 and 3, so it's available far more often | degrees |
| `segment_id`, `cvkas_valid` | Which valid episode a row belongs to (`-1` = none), and whether the required markers are present | |
| `imu{1,2}_ax/ay/az` | Accelerometer | m/s² |
| `imu{1,2}_gx/gy/gz` | Gyroscope (not bias-corrected) | rad/s |
| `emg{1,2}_rms` | EMG RMS envelope over the last `--emg-window` seconds, per-trial DC offset removed | ADC counts |

Empty cells mean the signal wasn't available at that moment. Nothing is invented across a gap.

### Manifest (`manifest.csv`)

One row per trial. `status` is `ok`, `skipped` (a newer recording format) or `error` (with the reason in `error`). A broken trial never stops the run. The other columns are quality stats: effective EMG and IMU rates and their longest gaps, how much of the grid has each signal (`*_coverage`), camera frame rate, how often each marker was seen (`cvkas_cov_m*`), and how much of the trial is a usable CVKAS episode (`usable_fraction`, `grid_valid_fraction`).

`quality_flags` lists anything worth knowing before using a trial (`;`-separated). A flag never removes a trial; filter on it yourself:

| Flag | Meaning |
|---|---|
| `emg_low_rate` | EMG arrived at under 500 samples/s (the sensor samples at 1 kHz) |
| `emg_long_gap` | A gap of over 0.2 s between EMG samples |
| `emg_flat_emg1` / `emg_flat_emg2` | That channel is constant: a dead or disconnected electrode |
| `imu_low_rate` | Under 50 real IMU samples/s (the sensor reads at 100 Hz) |
| `imu_no_data` | The IMU file has no values; IMU columns are left out |
| `no_markers` | No marker was seen in at least half the frames, so there are no CVKAS episodes |
| `no_usable_cvkas` | Markers were seen, but never long enough for a usable episode |
| `no_cvkas_frames` | The CVKAS file is empty (the camera wasn't running) |
| `short` | The aligned trial is under 2 s long |

For example, to keep only trials with good sensor data and a usable shank angle:
```python
import pandas as pd
m = pd.read_csv("cleaned/manifest.csv")
good = m[(m.status == "ok")
         & ~m.quality_flags.fillna("").str.contains("emg_low_rate|imu_low_rate|imu_no_data|no_markers|no_usable_cvkas")]
```

## What the cleaning does

**CVKAS**
- Pivots to one row per camera frame (a missing marker is an absent row in the raw file).
- Fills interior gaps up to `--max-gap` seconds with PCHIP; longer gaps are left as NaN. `m{id}_src` marks each value as missing (0), observed (1) or interpolated (2).
- Throws away the recorded velocity, acceleration and angles, and recomputes them from the cleaned positions. The recorded values are Kalman states that reset after gaps.
- Adds `shank_angle` (needs markers 2 and 3 only) alongside the app's hip and knee angles (need markers 0 to 3).
- Splits each trial into `segment_id` episodes wherever a required marker is missing for longer than `--max-gap`.

**IMU**
- Keeps only real samples. v1 files repeat the last IMU reading on every EMG row (~1 kHz) while the IMU updates at ~100 Hz.
- Converts everything to SI units. v1 data has two recording formats, detected per file: participants p002–p005 were recorded in m/s² and rad/s, later ones in raw sensor counts (ISM330DHCX at ±4 g and ±500 dps: 0.122 mg and 17.5 mdps per count).
- On the grid, each row takes the latest IMU sample from up to 50 ms before it.

**EMG**
- RMS envelope over a trailing `--emg-window` (default 50 ms) on the recorded timestamps, after removing each channel's mean. Windows with no samples are left empty.

**Alignment**
- All three modalities share the recording computer's clock (`timestamp`), so they're aligned on it directly. The grid covers the time all available modalities overlap.

## What the v1 data looks like

From a full run over the Drive in November 2026: 957 trials (stairs 425, seated 299, walking 233), about 2.4 hours, 20 participant IDs. Every trial has all three modalities.

| | p002–p005 (184 trials) | p006 onwards (773 trials) |
|---|---|---|
| EMG rate | ~69 samples/s | ~722 samples/s |
| IMU rate | ~14 Hz | ~99 Hz |
| IMU units recorded | m/s², rad/s | raw counts |
| Shank angle available | 0% of the time | 64% |
| Usable CVKAS time | 0% | 71% |

- The earlier participants were recorded with older firmware: their EMG and IMU are far sparser, and they have no usable marker data. They're flagged `emg_low_rate` and `imu_low_rate`.
- Markers 0 and 1 (thigh) are rarely detected, so `hip_angle` and `knee_angle` are almost never available. `shank_angle` is the usable CVKAS signal.
- After excluding flagged trials, about 650 trials remain, with about 1.2 hours of valid CVKAS time.
- One participant ID, `p412`, has a single trial and looks like a typo (perhaps p012). It's processed like the rest; check it before relying on it.
- v1 timestamps are the times data *arrived* at the computer, not when it was sampled. EMG arrives in bursts, so per-sample spacing isn't reliable; this is fixed for future recordings by the app's `seq` column (pipeline v2).

### One row per frame (no cleaning)

Raw CVKAS has one row per detected marker. To get one row per frame, with each marker's fields as columns, without changing anything:

```
.venv/bin/bionix-compress data/ --out compressed/
```

Columns: `timestamp, n_markers, hip_angle, knee_angle, m{id}_x, m{id}_y, m{id}_vel, m{id}_acc` for each marker id. Values are copied as recorded. Source files are only read, and `--out` must be a different folder. `expand_cvkas` turns the result back into the original rows exactly; the tests check this.
