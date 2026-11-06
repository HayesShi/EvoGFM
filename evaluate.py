"""Evaluate a frozen checkpoint; validation is the default split."""

import argparse
from evogfm.evaluation import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/source/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="runs/public")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--ablation", choices=["none", "zero-inverse"], default="none")
    parser.add_argument("--nc", nargs="*", default=["cora", "roman-empire"])
    parser.add_argument("--lp", nargs="*", default=["cora", "citeseer"])
    parser.add_argument("--threads", type=int, default=4)
    evaluate(**vars(parser.parse_args()))
