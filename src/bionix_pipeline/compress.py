"""Lossless reshaping of raw CVKAS: one row per marker -> one row per frame.

Raw CVKAS writes one row per detected marker per camera frame, plus a
timestamp-only row for frames with no detections. ``compress_cvkas`` turns
that into one row per frame timestamp with each marker's fields as columns:

    timestamp, n_markers, hip_angle, knee_angle,
    m0_x, m0_y, m0_vel, m0_acc, m1_x, ...

Nothing is cleaned, filled or recomputed: values are copied as recorded
(velocity/acceleration stay scalars or "[vx, vy]" strings, whichever the app
version wrote). ``expand_cvkas`` inverts it, which the tests use to prove the
round trip is exact. Source files are only ever read, never written.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FIELDS = {"pos_x": "x", "pos_y": "y", "velocity": "vel", "acceleration": "acc"}
FRAME_FIELDS = ("hip_angle", "knee_angle")  # written once per frame, repeated on every marker row


def compress_cvkas(raw: pd.DataFrame, markers=None, on_duplicate: str = "error") -> pd.DataFrame:
    """Raw long CVKAS -> one row per unique timestamp, indexed by timestamp.

    ``markers``: marker ids to emit columns for (default: every id seen).
    ``on_duplicate``: "error" (default, guarantees losslessness) or "last" when
    the same marker appears twice in one frame.
    """
    frames = pd.Index(pd.unique(raw["timestamp"]), name="timestamp").sort_values()
    det = raw.dropna(subset=["marker_id"]).copy()
    det["marker_id"] = det["marker_id"].astype(int)

    dup = det.duplicated(["timestamp", "marker_id"], keep="last")
    if dup.any():
        if on_duplicate == "error":
            raise ValueError(f"{int(dup.sum())} duplicate (timestamp, marker_id) rows; "
                             "pass on_duplicate='last' to keep the last one")
        det = det[~dup]

    for c in FRAME_FIELDS:
        if c in raw and (raw.groupby("timestamp")[c].nunique() > 1).any():
            raise ValueError(f"{c} differs between rows of the same frame")

    if markers is None:
        markers = sorted(det["marker_id"].unique())

    out = pd.DataFrame(index=frames)
    out["n_markers"] = det.groupby("timestamp").size().reindex(frames, fill_value=0).astype(int)
    for c in FRAME_FIELDS:
        if c in raw:
            out[c] = raw.groupby("timestamp")[c].first().reindex(frames)
    for m in markers:
        d = det[det["marker_id"] == m].set_index("timestamp")
        for src, short in FIELDS.items():
            if src in d:
                out[f"m{m}_{short}"] = d[src].reindex(frames)
    return out


def expand_cvkas(wide: pd.DataFrame) -> pd.DataFrame:
    """Inverse of ``compress_cvkas``: back to one row per detected marker
    (plus a timestamp-only row for frames with no detections)."""
    markers = sorted({int(c[1:].split("_")[0]) for c in wide.columns
                      if c.startswith("m") and c[1:].split("_")[0].isdigit()})
    parts = []
    for m in markers:
        cols = {f"m{m}_{short}": src for src, short in FIELDS.items() if f"m{m}_{short}" in wide}
        d = wide[[*cols, *[c for c in FRAME_FIELDS if c in wide]]].rename(columns=cols)
        d = d[d["pos_x"].notna()].reset_index()
        d.insert(1, "marker_id", m)
        parts.append(d)
    empty = wide.index[wide["n_markers"] == 0]
    parts.append(pd.DataFrame({"timestamp": empty}))
    long = pd.concat(parts, ignore_index=True)
    long["marker_id"] = long["marker_id"].astype("Int64")
    cols = ["timestamp", "marker_id", *FIELDS, *FRAME_FIELDS]
    return long[[c for c in cols if c in long]].sort_values(["timestamp", "marker_id"],
                                                            ignore_index=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Write one-row-per-frame copies of raw CVKAS CSVs.")
    p.add_argument("input", type=Path, help="folder of BionixDB CSVs (read only)")
    p.add_argument("--out", type=Path, default=Path("compressed"))
    args = p.parse_args(argv)
    if args.out.resolve() == args.input.resolve():
        p.error("--out must be a different folder from input; source files are never overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    for f in sorted(args.input.glob("cvkas-*.csv")):
        raw = pd.read_csv(f)
        wide = compress_cvkas(raw)
        wide.to_csv(args.out / f.name)
        print(f"{f.name}: {len(raw)} rows -> {len(wide)} frames, "
              f"markers {[c[1:-2] for c in wide.columns if c.endswith('_x')]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
