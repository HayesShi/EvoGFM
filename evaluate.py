"""Evaluate a frozen checkpoint; validation is the default split."""

import argparse
from evogfm.evaluation import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/source/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="runs/public")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--ablation", choices=["none", "zero-inverse", "propagation"], default="none"
    )
    parser.add_argument("--nc", nargs="*", default=["cora", "roman-empire"])
    parser.add_argument("--lp", nargs="*", default=["cora", "citeseer"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--node-readout",
        choices=["current", "adaptive", "enhanced", "advanced"],
        default="current",
    )
    parser.add_argument(
        "--nc-metric", choices=["accuracy", "auc", "auto"], default="accuracy"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="NC training labels per class; 0 uses the official training set",
    )
    parser.add_argument(
        "--edge-rate",
        type=float,
        default=1.0,
        help="Fraction of NC graph edges observed",
    )
    evaluate(**vars(parser.parse_args()))
