"""ImageNet readers for mixed-class HDF5 and WebDataset shards."""

from __future__ import annotations

import bisect
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import unquote, urlparse

import h5py
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision.datasets import VisionDataset


HDF5_MANIFEST = "manifest.json"
WIDS_MANIFEST = "dataset.json"
IMAGE_EXTENSIONS = (
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "gif",
    "tif",
    "tiff",
    "webp",
    "ppm",
    "pgm",
    "pbm",
    "pnm",
)


def _decode_image(encoded):
    with Image.open(io.BytesIO(encoded)) as source:
        return source.convert("RGB")


def _decode_target(encoded):
    if hasattr(encoded, "read"):
        encoded = encoded.read()
    if isinstance(encoded, (bytes, bytearray, memoryview)):
        encoded = bytes(encoded).decode("ascii")
    return int(encoded)


def _find_image(sample, dotted=False):
    prefix = "." if dotted else ""
    for extension in IMAGE_EXTENSIONS:
        key = prefix + extension
        if key in sample:
            encoded = sample[key]
            if hasattr(encoded, "read"):
                encoded = encoded.read()
            return encoded
    key = sample.get("__key__", "<unknown>")
    raise KeyError(f"Sample {key!r} has no supported image component")


def _local_shard_name(url):
    """Convert a local WIDS URL to a filesystem path."""
    parsed = urlparse(url)
    if parsed.scheme not in ("", "file"):
        raise ValueError(
            "Direct WIDS access only supports local paths; provide cache_dir "
            f"for {url!r}"
        )
    return unquote(parsed.path) if parsed.scheme == "file" else url


class _DirectLocalName:
    """Make a local symlink so WIDS puts lock files outside shared storage."""

    def __init__(self):
        default = os.path.join(tempfile.gettempdir(), f"_wids_direct_{os.getuid()}")
        self.directory = os.environ.get("WIDS_DIRECT_CACHE", default)

    def __call__(self, url):
        target = os.path.abspath(_local_shard_name(url))
        os.makedirs(self.directory, exist_ok=True)
        digest = hashlib.sha256(target.encode("utf-8")).hexdigest()
        suffix = Path(target).suffix
        local = os.path.join(self.directory, digest + suffix)
        try:
            os.symlink(target, local)
        except FileExistsError:
            pass
        return local


def _load_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


class ImageNetShardedHDF5(VisionDataset):
    """Indexed ImageNet reader for mixed-class, flat-byte HDF5 shards."""

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        manifest=HDF5_MANIFEST,
        max_open_files=900,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if not isinstance(max_open_files, int) or max_open_files < 0:
            raise ValueError("max_open_files must be a non-negative integer")

        metadata = _load_json(os.path.join(root, manifest))
        if metadata.get("format") != "model-utilities.sharded-hdf5":
            raise ValueError(f"Unsupported sharded HDF5 manifest: {manifest}")
        if metadata.get("version") != 1:
            raise ValueError(f"Unsupported sharded HDF5 version: {metadata.get('version')}")

        self.classes = list(metadata["classes"])
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.shards = list(metadata["shards"])
        self._ends = []
        total = 0
        for shard in self.shards:
            total += int(shard["nsamples"])
            self._ends.append(total)
        if total != int(metadata["num_samples"]):
            raise ValueError("Shard sample counts do not match num_samples")

        self.max_open_files = max_open_files
        self._open_files = {}
        self._owner_pid = None

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def _ensure_process(self):
        pid = os.getpid()
        if self._owner_pid != pid:
            self.close()
            self._owner_pid = pid

    def _location(self, index):
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._ends, index)
        start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        return shard_index, index - start

    def _read(self, handle, local_index):
        offsets = handle["offsets"]
        start = int(offsets[local_index])
        end = int(offsets[local_index + 1])
        encoded = handle["data"][start:end].tobytes()
        target = int(handle["targets"][local_index])
        return encoded, target

    def _open(self, shard_index):
        path = os.path.join(self.root, self.shards[shard_index]["file"])
        handle = h5py.File(path, "r")
        for name in ("data", "offsets", "targets"):
            if name not in handle:
                handle.close()
                raise KeyError(f"{path} has no {name!r} dataset")
        return handle

    def _load_encoded(self, index):
        self._ensure_process()
        shard_index, local_index = self._location(index)

        if shard_index in self._open_files:
            return self._read(self._open_files[shard_index], local_index)

        if len(self._open_files) < self.max_open_files:
            handle = self._open(shard_index)
            self._open_files[shard_index] = handle
            return self._read(handle, local_index)

        with self._open(shard_index) as handle:
            return self._read(handle, local_index)

    def __getitem__(self, index):
        encoded, target = self._load_encoded(index)
        image = _decode_image(encoded)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def close(self):
        handles = getattr(self, "_open_files", {})
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass
        handles.clear()
        self._owner_pid = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_open_files"] = {}
        state["_owner_pid"] = None
        return state


class _PILSampleDecoder:
    def __init__(self, transform=None, target_transform=None, dotted=False):
        self.transform = transform
        self.target_transform = target_transform
        self.dotted = dotted

    def __call__(self, sample):
        image = _decode_image(_find_image(sample, dotted=self.dotted))
        target_key = ".cls" if self.dotted else "cls"
        target = _decode_target(sample[target_key])
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class ImageNetWIDS(VisionDataset):
    """Map-style, indexed reader for ImageNet WebDataset tar shards."""

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        manifest=WIDS_MANIFEST,
        cache_dir=None,
        lru_size=8,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if not isinstance(lru_size, int) or lru_size < 1:
            raise ValueError("lru_size must be a positive integer")
        self.manifest = os.path.join(root, manifest)
        metadata = _load_json(self.manifest)
        self.classes = list(metadata.get("classes", []))
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self._length = sum(int(shard["nsamples"]) for shard in metadata["shardlist"])
        self.cache_dir = cache_dir
        self.lru_size = lru_size
        self._dataset = None
        self._owner_pid = None
        self._decoder = _PILSampleDecoder(transform, target_transform, dotted=True)

    def __len__(self):
        return self._length

    def _get_dataset(self):
        pid = os.getpid()
        if self._owner_pid != pid:
            self.close()
            self._owner_pid = pid
        if self._dataset is None:
            try:
                import wids
            except ImportError as error:
                raise ImportError(
                    "ImageNetWIDS requires the optional 'wids' package; install "
                    "model-utilities[imagenet]"
                ) from error
            kwargs = {
                "lru_size": self.lru_size,
                "transformations": [],
            }
            if self.cache_dir is None:
                kwargs["localname"] = _DirectLocalName()
            else:
                kwargs["cache_dir"] = self.cache_dir
            self._dataset = wids.ShardListDataset(self.manifest, **kwargs)
        return self._dataset

    def __getitem__(self, index):
        return self._decoder(self._get_dataset()[index])

    def close(self):
        dataset = getattr(self, "_dataset", None)
        if dataset is not None:
            dataset.close()
        self._dataset = None
        self._owner_pid = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_dataset"] = None
        state["_owner_pid"] = None
        return state


class ImageNetWebDataset(IterableDataset):
    """Streaming WebDataset reader that retains the PIL transform pipeline."""

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        manifest=WIDS_MANIFEST,
        shuffle=True,
        shard_shuffle=100,
        shuffle_buffer=1_000,
        seed=0,
        split_by_node=True,
    ):
        super().__init__()
        self.root = str(root)
        self.manifest = os.path.join(self.root, manifest)
        metadata = _load_json(self.manifest)
        base = Path(self.manifest).resolve().parent
        self.urls = [str(base / shard["url"]) for shard in metadata["shardlist"]]
        self.classes = list(metadata.get("classes", []))
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self._length = sum(int(shard["nsamples"]) for shard in metadata["shardlist"])
        self.transform = transform
        self.target_transform = target_transform
        self.shuffle = shuffle
        self.shard_shuffle = shard_shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.split_by_node = split_by_node
        self._pipeline = None
        self._owner_pid = None
        if shuffle and shuffle_buffer < 1:
            raise ValueError("shuffle_buffer must be positive when shuffle=True")

    def __len__(self):
        return self._length

    def _get_pipeline(self):
        pid = os.getpid()
        if self._owner_pid != pid:
            self._pipeline = None
            self._owner_pid = pid
        if self._pipeline is None:
            try:
                import webdataset as wds
            except ImportError as error:
                raise ImportError(
                    "ImageNetWebDataset requires the optional 'webdataset' package; "
                    "install model-utilities[imagenet]"
                ) from error

            nodesplitter = wds.split_by_node if self.split_by_node else None
            pipeline = wds.WebDataset(
                self.urls,
                shardshuffle=self.shard_shuffle if self.shuffle else False,
                nodesplitter=nodesplitter,
                workersplitter=wds.split_by_worker,
                seed=self.seed,
                empty_check=False,
            )
            if self.shuffle:
                initial = min(self.shuffle_buffer, max(1, self.shuffle_buffer // 4))
                pipeline = pipeline.shuffle(self.shuffle_buffer, initial=initial)
            self._pipeline = pipeline.map(
                _PILSampleDecoder(self.transform, self.target_transform, dotted=False)
            )
        return self._pipeline

    def __iter__(self):
        return iter(self._get_pipeline())

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pipeline"] = None
        state["_owner_pid"] = None
        return state


def make_wids_sampler(
    dataset,
    *,
    distributed=False,
    chunksize=2_000,
    shuffle=True,
    seed=0,
    num_replicas=None,
    rank=None,
):
    """Create the locality-aware WIDS sampler for one or many ranks."""
    try:
        import wids
    except ImportError as error:
        raise ImportError(
            "make_wids_sampler requires the optional 'wids' package; install "
            "model-utilities[imagenet]"
        ) from error
    if distributed:
        return wids.DistributedChunkedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            chunksize=chunksize,
            shuffle=shuffle,
            seed=seed,
        )
    return wids.ChunkedSampler(
        dataset,
        chunksize=chunksize,
        shuffle=shuffle,
        seed=seed,
    )
