"""Cleaning and alignment of BionixDB EMG / IMU / CVKAS recordings for RL."""

# Pipeline version, written to every output row of manifest.csv. "1" handles the
# recordings made so far; later recording formats (seq column, one-row-per-frame
# CVKAS) will be handled by pipeline version 2.
PIPELINE_VERSION = "1"
