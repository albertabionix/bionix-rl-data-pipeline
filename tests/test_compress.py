import io as stdio

import numpy as np
import pandas as pd
import pytest

from bionix_pipeline.compress import compress_cvkas, expand_cvkas
from bionix_pipeline.cvkas import add_angles, to_frames

# Two real frames from cvkas-p020-stairs-f_o_l_leg-02, the only ones in the
# samples where the app saw all four markers and wrote hip/knee angles,
# plus an empty frame and a partial frame.
RAW = """timestamp,marker_id,pos_x,pos_y,velocity,acceleration,hip_angle,knee_angle
1788726877.7163205,1,207.0,126.25,54.05762481689453,3068.04931640625,,
1788726877.7163205,3,206.5,216.0,6.757203102111816,1123.13671875,,
1788726877.7553241,1,206.0,126.5,26.427719116210938,746.926513671875,178.46565850029924,7.24493408203125
1788726877.7553241,3,206.5,216.0,0.0,173.24557495117188,178.46565850029924,7.24493408203125
1788726877.7553241,0,203.75,42.5,26.315385818481445,540.820068359375,178.46565850029924,7.24493408203125
1788726877.7553241,2,212.5,156.0,70.7210693359375,1285.544921875,178.46565850029924,7.24493408203125
1788726877.8753197,,,,,,,
1788726877.912327,3,205.25,215.75,0.0,173.2209930419922,180.0,3.087651014328003
1788726877.912327,1,201.0,126.5,20.266254425048828,1026.0877685546875,180.0,3.087651014328003
1788726877.912327,0,201.0,42.75,17.587825775146484,56.95878219604492,180.0,3.087651014328003
1788726877.912327,2,208.5,155.5,26.31365203857422,173.12216186523438,180.0,3.087651014328003
"""


def raw():
    return pd.read_csv(stdio.StringIO(RAW))


def test_one_row_per_timestamp():
    wide = compress_cvkas(raw())
    assert len(wide) == 4
    assert list(wide["n_markers"]) == [2, 4, 0, 4]
    assert {"m0_x", "m0_vel", "m3_acc", "hip_angle"} <= set(wide.columns)
    assert np.isnan(wide["m0_x"].iloc[0])           # marker 0 not seen in frame 1


def test_round_trip_is_exact():
    r = raw().sort_values(["timestamp", "marker_id"], ignore_index=True)
    r["marker_id"] = r["marker_id"].astype("Int64")
    back = expand_cvkas(compress_cvkas(r))
    pd.testing.assert_frame_equal(back, r)


def test_does_not_modify_input():
    r = raw()
    before = r.copy()
    compress_cvkas(r)
    pd.testing.assert_frame_equal(r, before)


def test_duplicate_marker_in_frame_raises():
    r = pd.concat([raw(), raw().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        compress_cvkas(r)
    assert len(compress_cvkas(r, on_duplicate="last")) == 4


def test_recomputed_angles_match_app():
    f = add_angles(to_frames(raw()))
    full = f.dropna(subset=["knee_angle"])
    assert len(full) == 2
    assert np.allclose(full["hip_angle"], [178.46565850029924, 180.0])
    assert np.allclose(full["knee_angle"], [7.24493408203125, 3.087651014328003])
