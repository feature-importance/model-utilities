# ImageNet storage tools

The repacker reads the existing `dest.p` plus one-HDF5-file-per-class layout. It
can produce two independent physical layouts:

- mixed-class HDF5 shards for indexed reads;
- uncompressed tar shards used by both indexed WIDS and streaming WebDataset.

The original JPEG/PNG bytes are copied unchanged. PIL still performs image decode,
and the benchmark applies the same torchvision transform to every backend.

## Repack once

Install the optional readers in the `feature-importance` environment:

```bash
conda run -n feature-importance pip install -e '.[imagenet]'
```

Run the conversion on a node with a reliable connection to the shared filesystem:

```bash
conda run -n feature-importance python tools/imagenet/repack.py \
  --input /shared/imagenet/hdf5-by-class \
  --hdf5-output /shared/imagenet/hdf5-sharded \
  --webdataset-output /shared/imagenet/webdataset \
  --num-shards 256
```

The output directories must be empty. The conversion balances every class across
the shards. HDF5 conversion reads the input twice: once to size the flat byte
arrays and once to write them. For streaming, use at least as many shards as the
total number of distributed ranks times DataLoader workers; several times that
number gives better shard shuffling. The default 256 is a sensible starting point
for common cluster jobs and remains comfortably below a 1024-file descriptor
limit.

## Benchmark

Single node, comparing all implementations:

```bash
conda run -n feature-importance python tools/imagenet/benchmark.py \
  --original-root /shared/imagenet/hdf5-by-class \
  --sharded-hdf5-root /shared/imagenet/hdf5-sharded \
  --webdataset-root /shared/imagenet/webdataset \
  --workers 8 --batch-size 256 --samples-per-rank 100000 \
  --output benchmark.json
```

Multi-node or multi-GPU execution uses the usual `torchrun` environment:

```bash
torchrun --nproc-per-node=4 tools/imagenet/benchmark.py \
  --original-root /shared/imagenet/hdf5-by-class \
  --sharded-hdf5-root /shared/imagenet/hdf5-sharded \
  --webdataset-root /shared/imagenet/webdataset \
  --device cuda --workers 8 --batch-size 256
```

Use the same requested nodes, worker count, batch size, transform, and sample count
for comparisons. Repeat runs after a cold run to distinguish shared-filesystem
throughput from Linux page-cache throughput. WIDS defaults to direct reads from
the shared path (its small lock files and symlinks live under `/tmp`, not beside
the shared shards); pass `--wids-cache-dir "$TMPDIR/wids"` only when intentionally
benchmarking full node-local staging.

## ImageNet-50 and ImageNet-100 indexed subsets

The predefined subsets are map-style views over the same tar shards; no image
data is copied or rewritten:

```python
from model_utilities.datasets import ImageNet50WIDS, ImageNet100WIDS

train100 = ImageNet100WIDS(
    "/shared/imagenet/webdataset",
    transform=train_transform,
)
train50 = ImageNet50WIDS(
    "/shared/imagenet/webdataset",
    transform=train_transform,
)
```

The first construction scans tar headers and the small `.cls` members, without
reading or decoding image payloads. It writes a compact full-target index plus a
subset index. ImageNet-50 and ImageNet-100 share the full-target index, and later
runs load the cached arrays directly. A file lock prevents distributed ranks from
building the same index concurrently.

By default these indices are written beside `dataset.json` and the tar shards.
Once generated on a node with write access, compute nodes can load them from a
read-only mount without creating a lock file. If the dataset directory is not
writable and the indices do not yet exist, construction falls back to node-local
`/tmp`. This is independent of `cache_dir`, which controls whether WIDS stages the
much larger tar shards themselves.
