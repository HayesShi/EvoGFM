"""Evaluate retention and accumulation along a graph-domain stream."""

import argparse
from evogfm.continual import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/source/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="runs/continual")
    parser.add_argument("--stream", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--future", nargs="+", default=["roman-empire"])
    parser.add_argument("--task", choices=["both", "nc", "lp"], default="both")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--nc-metric", choices=["accuracy", "auto"], default="accuracy")
    parser.add_argument(
        "--source-anchor",
        type=float,
        default=0.0,
        help="Quadratic penalty around source coefficients, weighted by source GGN",
    )
    run(**vars(parser.parse_args()))
