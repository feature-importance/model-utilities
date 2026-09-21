#!/usr/bin/env python3
"""Shuffle samples within existing ImageNet tar shards into a fresh directory."""

import argparse
import json

from model_utilities.datasets.imagenet_shuffle import shuffle_imagenet_shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Existing directory containing dataset.json and local tar shards")
    parser.add_argument("--output", required=True, help="New output directory; must not already exist")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    def progress(completed, total):
        print(f"[shuffle and verify] {completed}/{total} shards", flush=True)

    result = shuffle_imagenet_shards(args.input, args.output, seed=args.seed, progress=progress)
    print(json.dumps({"num_samples": result["num_samples"],
                      "num_shards": len(result["shardlist"]),
                      "seed": args.seed, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
