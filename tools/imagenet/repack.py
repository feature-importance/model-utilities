#!/usr/bin/env python3
"""Repack the legacy ImageNet HDF5 layout into training-friendly shards."""

from __future__ import annotations

import argparse
import json

from model_utilities.datasets import repack_imagenet_hdf5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Legacy HDF5 dataset directory")
    parser.add_argument("--hdf5-output", help="Output directory for sharded HDF5")
    parser.add_argument(
        "--webdataset-output",
        help="Output directory for tar shards shared by WIDS and WebDataset",
    )
    parser.add_argument("--num-shards", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="imagenet-train")
    args = parser.parse_args()
    if args.hdf5_output is None and args.webdataset_output is None:
        parser.error("at least one of --hdf5-output or --webdataset-output is required")
    return args


def main():
    args = parse_args()

    def progress(phase, completed, total):
        if completed == total or completed == 1 or completed % 25 == 0:
            print(f"[{phase}] {completed}/{total} classes", flush=True)

    result = repack_imagenet_hdf5(
        args.input,
        hdf5_output=args.hdf5_output,
        webdataset_output=args.webdataset_output,
        num_shards=args.num_shards,
        seed=args.seed,
        name=args.name,
        progress=progress,
    )
    summary = {
        kind: {
            "num_samples": manifest["num_samples"],
            "num_shards": len(manifest.get("shards", manifest.get("shardlist", []))),
        }
        for kind, manifest in result.items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
