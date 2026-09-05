#!/usr/bin/env python3
"""Benchmark ImageNet storage backends with an identical PIL transform pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchvision import transforms

from model_utilities.datasets import (
    ImageNetHDF5,
    ImageNetHDF5Handles,
    ImageNetShardedHDF5,
    ImageNetWebDataset,
    ImageNetWIDS,
    make_wids_sampler,
)


BACKENDS = (
    "original-hdf5",
    "original-hdf5-handles",
    "sharded-hdf5",
    "wids",
    "webdataset",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", help="Legacy one-file-per-class HDF5 root")
    parser.add_argument("--sharded-hdf5-root", help="Sharded HDF5 root")
    parser.add_argument("--webdataset-root", help="Tar shard root containing dataset.json")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--samples-per-rank", type=int, default=50_000)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--transform", choices=("eval", "train"), default="eval")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-open-files", type=int, default=900)
    parser.add_argument("--wids-lru-size", type=int, default=8)
    parser.add_argument("--wids-cache-dir")
    parser.add_argument("--wids-chunksize", type=int, default=2_000)
    parser.add_argument("--shuffle-buffer", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="Write the rank-zero results to this JSON file")
    args = parser.parse_args()

    if args.workers < 0 or args.batch_size < 1 or args.samples_per_rank < 1:
        parser.error("workers must be non-negative; batch size and samples must be positive")
    if args.repeats < 1 or args.warmup_batches < 0:
        parser.error("repeats must be positive and warmup batches non-negative")
    if args.workers > 0 and args.prefetch_factor < 1:
        parser.error("prefetch factor must be positive when workers are enabled")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is unavailable")
    if args.pin_memory is None:
        args.pin_memory = args.device == "cuda"

    inferred = []
    if args.original_root:
        inferred.extend(("original-hdf5", "original-hdf5-handles"))
    if args.sharded_hdf5_root:
        inferred.append("sharded-hdf5")
    if args.webdataset_root:
        inferred.extend(("wids", "webdataset"))
    args.backends = args.backends or inferred
    if not args.backends:
        parser.error("provide at least one dataset root")

    required_roots = {
        "original-hdf5": args.original_root,
        "original-hdf5-handles": args.original_root,
        "sharded-hdf5": args.sharded_hdf5_root,
        "wids": args.webdataset_root,
        "webdataset": args.webdataset_root,
    }
    missing = [backend for backend in args.backends if not required_roots[backend]]
    if missing:
        parser.error(f"missing dataset root for: {', '.join(missing)}")
    return args


def distributed_context(device_name):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device_name == "cuda" else "gloo")
    if device_name == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return world_size, rank, device


def image_transform(name):
    if name == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )


def make_dataset(backend, args, transform):
    if backend == "original-hdf5":
        return ImageNetHDF5(args.original_root, transform=transform)
    if backend == "original-hdf5-handles":
        return ImageNetHDF5Handles(
            args.original_root,
            transform=transform,
            max_open_files=args.max_open_files,
        )
    if backend == "sharded-hdf5":
        return ImageNetShardedHDF5(
            args.sharded_hdf5_root,
            transform=transform,
            max_open_files=args.max_open_files,
        )
    if backend == "wids":
        return ImageNetWIDS(
            args.webdataset_root,
            transform=transform,
            cache_dir=args.wids_cache_dir,
            lru_size=args.wids_lru_size,
        )
    if backend == "webdataset":
        return ImageNetWebDataset(
            args.webdataset_root,
            transform=transform,
            shuffle=True,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
            split_by_node=True,
        )
    raise ValueError(backend)


def make_loader(dataset, backend, args, world_size, rank):
    sampler = None
    sampling = "streaming"
    if backend == "wids":
        sampler = make_wids_sampler(
            dataset,
            distributed=world_size > 1,
            chunksize=args.wids_chunksize,
            shuffle=True,
            seed=args.seed,
            num_replicas=world_size,
            rank=rank,
        )
        sampling = "chunk-shuffled-indexed"
    elif backend != "webdataset":
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
            )
        else:
            generator = torch.Generator().manual_seed(args.seed)
            sampler = RandomSampler(dataset, generator=generator)
        sampling = "random-indexed"

    kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "sampler": sampler,
        "num_workers": args.workers,
        "pin_memory": args.pin_memory,
        "drop_last": False,
    }
    if args.workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**kwargs), sampler, sampling


def synchronize(device, world_size):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if world_size > 1:
        dist.barrier()


def benchmark_once(loader, args, world_size, device):
    iterator = iter(loader)
    for _ in range(args.warmup_batches):
        try:
            images, _ = next(iterator)
        except StopIteration:
            raise RuntimeError("Dataset ended during warmup") from None
        if device.type == "cuda":
            images = images.to(device, non_blocking=True)

    synchronize(device, world_size)
    started = time.perf_counter()
    local_samples = 0
    while local_samples < args.samples_per_rank:
        try:
            images, _ = next(iterator)
        except StopIteration:
            break
        if device.type == "cuda":
            images = images.to(device, non_blocking=True)
        local_samples += int(images.shape[0])
    synchronize(device, world_size)
    elapsed = time.perf_counter() - started

    totals = torch.tensor(
        [float(local_samples), elapsed], dtype=torch.float64, device=device
    )
    if world_size > 1:
        sample_total = totals[0].clone()
        elapsed_max = totals[1].clone()
        dist.all_reduce(sample_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed_max, op=dist.ReduceOp.MAX)
        return int(sample_total.item()), elapsed_max.item()
    return local_samples, elapsed


def close_dataset(dataset):
    close = getattr(dataset, "close", None)
    if close is not None:
        close()


def main():
    args = parse_args()
    world_size, rank, device = distributed_context(args.device)
    torch.manual_seed(args.seed + rank)
    transform = image_transform(args.transform)
    results = []

    for backend in args.backends:
        dataset = make_dataset(backend, args, transform)
        loader, sampler, sampling = make_loader(dataset, backend, args, world_size, rank)
        for repeat in range(args.repeats):
            set_epoch = getattr(sampler, "set_epoch", None)
            if set_epoch is not None:
                set_epoch(repeat)
            samples, seconds = benchmark_once(loader, args, world_size, device)
            result = {
                "backend": backend,
                "repeat": repeat,
                "sampling": sampling,
                "transform": args.transform,
                "device": str(device),
                "world_size": world_size,
                "workers_per_rank": args.workers,
                "batch_size_per_rank": args.batch_size,
                "samples": samples,
                "seconds": seconds,
                "images_per_second": samples / seconds,
            }
            if rank == 0:
                results.append(result)
                print(json.dumps(result), flush=True)
        del loader
        close_dataset(dataset)
        del dataset
        gc.collect()

    if rank == 0 and args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as stream:
            json.dump(results, stream, indent=2)
            stream.write("\n")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
