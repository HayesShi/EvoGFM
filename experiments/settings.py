"""Shared paths for experiment modules, set by ``python -m experiments.run``."""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("EVOGFM_RESULTS", REPO / "runs/paper")).resolve()
DATA = Path(os.environ.get("EVOGFM_DATA", REPO / "data")).resolve()
CHECKPOINT = Path(
    os.environ.get("EVOGFM_CHECKPOINT", REPO / "checkpoints/reference.pt")
).resolve()
DEVICE = os.environ.get("EVOGFM_DEVICE", "cpu")
NC_RUN = ROOT / "node_classification"
STREAM_RUN = ROOT / "cross_task"

SOURCE_SPLITS = Path(
    os.environ.get("EVOGFM_SOURCE_SPLITS", REPO / "checkpoints/source_splits.json")
).resolve()
