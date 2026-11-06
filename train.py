"""Train one shared model on source graphs."""

import argparse
from evogfm.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="runs/source")
    parser.add_argument("--sources", nargs="+", default=["pubmed", "photo", "texas"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    train(**vars(args))
