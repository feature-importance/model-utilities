"""Indexed ImageNet-50/100 views over an existing WIDS dataset."""

from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
from urllib.parse import unquote, urlparse

import numpy as np

from .imagenet_sharded import ImageNetWIDS, WIDS_MANIFEST
from .imagenet_subsets_hdf5 import IMAGENET_50_SUBSET, IMAGENET_100_SUBSET


def _local_shard_path(base, url):
    parsed = urlparse(url)
    if parsed.scheme not in ("", "file"):
        raise ValueError(
            "ImageNet subset index construction requires local WebDataset shards; "
            f"cannot scan {url!r}"
        )
    path = Path(unquote(parsed.path) if parsed.scheme == "file" else url)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _sample_name(name):
    match = re.match(r"^((?:.*/)?.*?)(\..*)$", name)
    if match is None:
        return None, None
    return match.groups()


def _scan_shard_targets(path, expected_samples):
    """Read sample targets in WIDS order without reading encoded image payloads."""
    targets = []
    last_key = None
    with tarfile.open(path, mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            key, extension = _sample_name(member.name)
            if key is None:
                continue
            if key != last_key:
                targets.append(-1)
                last_key = key
            if extension == ".cls":
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Unable to read {member.name!r} from {path}")
                with stream:
                    targets[-1] = int(stream.read().decode("ascii"))

    actual_samples = len(targets)
    if actual_samples != expected_samples:
        raise ValueError(
            f"WIDS descriptor says {path} has {expected_samples} samples, "
            f"but its tar members contain {actual_samples}"
        )
    if any(target < 0 for target in targets):
        raise ValueError(f"One or more samples in {path} have no .cls component")
    return targets


def _fallback_cache_dir():
    user_id = os.getuid() if hasattr(os, "getuid") else "user"
    default = os.path.join(tempfile.gettempdir(), f"_wids_subsets_{user_id}")
    return Path(os.environ.get("WIDS_SUBSET_CACHE", default))


def _cache_key(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dataset_cache_key(manifest_path, metadata):
    base = manifest_path.resolve().parent
    shard_files = []
    for shard in metadata["shardlist"]:
        path = _local_shard_path(base, shard["url"])
        status = path.stat()
        shard_files.append(
            {
                "url": shard["url"],
                "size": status.st_size,
                "mtime_ns": status.st_mtime_ns,
            }
        )
    identity = {
        "metadata": metadata,
        "shard_files": shard_files,
    }
    return _cache_key(identity)


def _save_array_atomic(path, values):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "wb") as stream:
            np.save(stream, values, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_array(path, description, expected_length=None):
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"Invalid cached {description}: {path}")
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(
            f"Cached {description} contains {len(values)} entries; "
            f"expected {expected_length}"
        )
    return values


@contextmanager
def _exclusive_lock(path):
    try:
        import fcntl
    except ImportError as error:
        raise RuntimeError("ImageNet WIDS subset index locking requires POSIX") from error
    with open(path, "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def _load_or_build_targets(
    manifest_path, metadata, cache_dir, dataset_key, source_dirs=()
):
    cache_path = cache_dir / f"imagenet-targets-{dataset_key}.npy"
    lock_path = cache_path.with_suffix(".lock")
    expected_total = sum(int(shard["nsamples"]) for shard in metadata["shardlist"])

    if cache_path.is_file():
        return np.asarray(
            _load_array(cache_path, "ImageNet targets", expected_total),
            dtype=np.int32,
        )

    with _exclusive_lock(lock_path):
        if not cache_path.is_file():
            for source_dir in source_dirs:
                source_path = source_dir / cache_path.name
                if source_path.is_file():
                    targets = _load_array(
                        source_path, "ImageNet targets", expected_total
                    )
                    _save_array_atomic(cache_path, targets)
                    break
            else:
                targets = []
                base = manifest_path.resolve().parent
                for shard in metadata["shardlist"]:
                    expected_samples = int(shard["nsamples"])
                    path = _local_shard_path(base, shard["url"])
                    targets.extend(_scan_shard_targets(path, expected_samples))
                _save_array_atomic(cache_path, np.asarray(targets, dtype=np.int32))

        return np.asarray(
            _load_array(cache_path, "ImageNet targets", expected_total),
            dtype=np.int32,
        )


def _load_or_build_indices_in(
    manifest_path,
    metadata,
    classes,
    selected_targets,
    cache_dir,
    dataset_key,
    source_dirs=(),
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    subset_key = _cache_key({"dataset": dataset_key, "classes": sorted(classes)})
    cache_path = cache_dir / f"imagenet-subset-{subset_key}.npy"
    lock_path = cache_path.with_suffix(".lock")

    # Completed cache files are installed atomically and need no writable lock.
    if cache_path.is_file():
        return np.asarray(_load_array(cache_path, "ImageNet subset index"), dtype=np.int64)

    with _exclusive_lock(lock_path):
        if not cache_path.is_file():
            for source_dir in source_dirs:
                source_path = source_dir / cache_path.name
                if source_path.is_file():
                    indices = _load_array(source_path, "ImageNet subset index")
                    _save_array_atomic(cache_path, indices)
                    break
            else:
                targets = _load_or_build_targets(
                    manifest_path,
                    metadata,
                    cache_dir,
                    dataset_key,
                    source_dirs=source_dirs,
                )
                wanted = np.asarray(sorted(selected_targets), dtype=np.int32)
                indices = np.flatnonzero(np.isin(targets, wanted)).astype(np.int64)
                _save_array_atomic(cache_path, indices)

        return np.asarray(_load_array(cache_path, "ImageNet subset index"), dtype=np.int64)


def _load_or_build_indices(manifest_path, metadata, classes):
    full_classes = list(metadata.get("classes", []))
    missing = sorted(set(classes) - set(full_classes))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"WebDataset manifest is missing subset classes: {preview}")

    full_class_to_idx = {name: index for index, name in enumerate(full_classes)}
    selected_targets = {full_class_to_idx[name] for name in classes}
    dataset_key = _dataset_cache_key(manifest_path, metadata)
    data_dir = manifest_path.resolve().parent
    temporary_dir = _fallback_cache_dir()
    try:
        return _load_or_build_indices_in(
            manifest_path,
            metadata,
            classes,
            selected_targets,
            data_dir,
            dataset_key,
            source_dirs=(temporary_dir,),
        )
    except OSError as error:
        if not isinstance(error, PermissionError) and error.errno not in (
            errno.EACCES,
            errno.EPERM,
            errno.EROFS,
        ):
            raise
        return _load_or_build_indices_in(
            manifest_path,
            metadata,
            classes,
            selected_targets,
            temporary_dir,
            dataset_key,
            source_dirs=(data_dir,),
        )


class ImageNetSubsetWIDS(ImageNetWIDS):
    """Map-style subset over existing ImageNet WebDataset tar shards."""

    def __init__(
        self,
        root,
        classes,
        transform=None,
        target_transform=None,
        manifest=WIDS_MANIFEST,
        cache_dir=None,
        lru_size=8,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=None,
            manifest=manifest,
            cache_dir=cache_dir,
            lru_size=lru_size,
        )
        metadata = json.loads(Path(self.manifest).read_text(encoding="utf-8"))
        self._full_classes = self.classes
        self.classes = sorted(classes)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        full_class_to_idx = {
            name: index for index, name in enumerate(self._full_classes)
        }
        self._target_mapping = {
            full_class_to_idx[name]: target
            for target, name in enumerate(self.classes)
            if name in full_class_to_idx
        }
        self.indices = _load_or_build_indices(
            Path(self.manifest), metadata, self.classes
        )
        self.target_transform = target_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)
        full_index = int(self.indices[index])
        image, full_target = self._decoder(self._get_dataset()[full_index])
        target = self._target_mapping[full_target]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class ImageNet100WIDS(ImageNetSubsetWIDS):
    def __init__(self, root, transform=None, target_transform=None, **kwargs):
        super().__init__(
            root,
            IMAGENET_100_SUBSET,
            transform=transform,
            target_transform=target_transform,
            **kwargs,
        )


class ImageNet50WIDS(ImageNetSubsetWIDS):
    def __init__(self, root, transform=None, target_transform=None, **kwargs):
        super().__init__(
            root,
            IMAGENET_50_SUBSET,
            transform=transform,
            target_transform=target_transform,
            **kwargs,
        )
