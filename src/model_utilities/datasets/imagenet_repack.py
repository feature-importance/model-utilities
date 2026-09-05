"""Repack the legacy one-HDF5-file-per-class ImageNet dataset."""

from __future__ import annotations

from collections import defaultdict
import io
import json
from pathlib import Path
import pickle
import random
import tarfile

import h5py
import numpy as np
from PIL import Image

from .imagenet_sharded import HDF5_MANIFEST, WIDS_MANIFEST


HDF5_FORMAT = "model-utilities.sharded-hdf5"
HDF5_VERSION = 1


def _empty_output_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    return path


def _encoded_bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, np.ndarray):
        return value.tobytes()
    return bytes(value)


def _class_indices(source):
    with open(source / "dest.p", "rb") as stream:
        destinations = pickle.load(stream)

    by_class = defaultdict(list)
    for class_name, local_index in destinations:
        by_class[str(class_name)].append(int(local_index))

    available = {path.stem for path in source.glob("*.hdf5")}
    missing = sorted(set(by_class) - available)
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing source HDF5 files for: {preview}")

    # Match the legacy loader's target mapping, which is based on all filenames.
    classes = sorted(available)
    return classes, by_class


def _assignment(indices, class_index, num_shards, seed):
    shuffled = list(indices)
    rng = random.Random((int(seed) << 32) ^ class_index)
    rng.shuffle(shuffled)
    offset = rng.randrange(num_shards)
    for position, local_index in enumerate(shuffled):
        yield (offset + position) % num_shards, local_index


def _check_file_limit(num_shards):
    try:
        import resource
    except ImportError:
        return
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit != resource.RLIM_INFINITY and num_shards + 64 > soft_limit:
        raise ValueError(
            f"{num_shards} simultaneous shard writers are unsafe with the current "
            f"open-file limit ({soft_limit}); use at most {max(1, soft_limit - 64)} shards"
        )


def _image_extension(encoded):
    # Fast paths avoid constructing 1.28 million PIL objects during repacking.
    if encoded.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if encoded.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if encoded.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if encoded.startswith(b"BM"):
        return "bmp"
    if encoded.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    if encoded.startswith(b"RIFF") and encoded[8:12] == b"WEBP":
        return "webp"
    if encoded[:2] in (b"P1", b"P2", b"P3", b"P4", b"P5", b"P6"):
        return "ppm"
    with Image.open(io.BytesIO(encoded)) as image:
        raise ValueError(f"Unsupported encoded image format: {image.format!r}")


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def _write_sharded_hdf5(
    source, output, classes, by_class, num_shards, seed, progress=None
):
    counts = np.zeros(num_shards, dtype=np.int64)
    byte_counts = np.zeros(num_shards, dtype=np.int64)

    # The flat byte arrays need exact sizes, so this first pass only measures.
    for target, class_name in enumerate(classes):
        with h5py.File(source / f"{class_name}.hdf5", "r") as source_file:
            source_data = source_file["data"]
            for shard_index, local_index in _assignment(
                by_class[class_name], target, num_shards, seed
            ):
                encoded = _encoded_bytes(source_data[local_index])
                counts[shard_index] += 1
                byte_counts[shard_index] += len(encoded)
        if progress is not None:
            progress("hdf5-sizing", target + 1, len(classes))

    paths = [output / f"train-{index:05d}.hdf5" for index in range(num_shards)]
    handles = []
    try:
        for path, count, byte_count in zip(paths, counts, byte_counts):
            handle = h5py.File(path, "w")
            handle.create_dataset("data", shape=(int(byte_count),), dtype="u1")
            handle.create_dataset("offsets", shape=(int(count) + 1,), dtype="u8")
            handle.create_dataset("targets", shape=(int(count),), dtype="i4")
            handle["offsets"][0] = 0
            handles.append(handle)

        sample_positions = np.zeros(num_shards, dtype=np.int64)
        byte_positions = np.zeros(num_shards, dtype=np.int64)
        for target, class_name in enumerate(classes):
            with h5py.File(source / f"{class_name}.hdf5", "r") as source_file:
                source_data = source_file["data"]
                for shard_index, local_index in _assignment(
                    by_class[class_name], target, num_shards, seed
                ):
                    encoded = _encoded_bytes(source_data[local_index])
                    sample_position = int(sample_positions[shard_index])
                    byte_position = int(byte_positions[shard_index])
                    byte_end = byte_position + len(encoded)
                    handle = handles[shard_index]
                    handle["data"][byte_position:byte_end] = np.frombuffer(encoded, dtype="u1")
                    handle["targets"][sample_position] = target
                    handle["offsets"][sample_position + 1] = byte_end
                    sample_positions[shard_index] += 1
                    byte_positions[shard_index] = byte_end
            if progress is not None:
                progress("hdf5-writing", target + 1, len(classes))
    finally:
        for handle in handles:
            handle.close()

    manifest = {
        "format": HDF5_FORMAT,
        "version": HDF5_VERSION,
        "num_samples": int(counts.sum()),
        "classes": classes,
        "shards": [
            {
                "file": path.name,
                "nsamples": int(count),
                "nbytes": int(byte_count),
            }
            for path, count, byte_count in zip(paths, counts, byte_counts)
        ],
    }
    _write_json(output / HDF5_MANIFEST, manifest)
    return manifest


def _tar_member(archive, name, payload):
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mtime = 0
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    archive.addfile(member, io.BytesIO(payload))


def _write_webdataset(
    source, output, classes, by_class, num_shards, seed, name, progress=None
):
    paths = [output / f"train-{index:05d}.tar" for index in range(num_shards)]
    counts = np.zeros(num_shards, dtype=np.int64)
    archives = []
    try:
        for path in paths:
            archives.append(tarfile.open(path, mode="w"))
        for target, class_name in enumerate(classes):
            with h5py.File(source / f"{class_name}.hdf5", "r") as source_file:
                source_data = source_file["data"]
                for shard_index, local_index in _assignment(
                    by_class[class_name], target, num_shards, seed
                ):
                    encoded = _encoded_bytes(source_data[local_index])
                    extension = _image_extension(encoded)
                    key = f"{class_name}_{local_index:08d}"
                    _tar_member(archives[shard_index], f"{key}.{extension}", encoded)
                    _tar_member(
                        archives[shard_index], f"{key}.cls", str(target).encode("ascii")
                    )
                    counts[shard_index] += 1
            if progress is not None:
                progress("webdataset-writing", target + 1, len(classes))
    finally:
        for archive in archives:
            archive.close()

    manifest = {
        "wids_version": 1,
        "name": name,
        "num_samples": int(counts.sum()),
        "classes": classes,
        "shardlist": [
            {
                "url": path.name,
                "nsamples": int(count),
                "filesize": path.stat().st_size,
            }
            for path, count in zip(paths, counts)
        ],
    }
    _write_json(output / WIDS_MANIFEST, manifest)
    return manifest


def repack_imagenet_hdf5(
    source,
    *,
    hdf5_output=None,
    webdataset_output=None,
    num_shards=256,
    seed=0,
    name="imagenet-train",
    progress=None,
):
    """Create mixed-class HDF5 shards and/or shared WIDS/WebDataset tar shards."""
    if hdf5_output is None and webdataset_output is None:
        raise ValueError("At least one output directory is required")
    if not isinstance(num_shards, int) or num_shards < 1:
        raise ValueError("num_shards must be a positive integer")
    _check_file_limit(num_shards)

    source = Path(source)
    if not (source / "dest.p").is_file():
        raise FileNotFoundError(f"No dest.p found in {source}")
    classes, by_class = _class_indices(source)
    num_samples = sum(len(indices) for indices in by_class.values())
    if num_samples == 0:
        raise ValueError("The source dataset contains no samples")
    if num_shards > num_samples:
        raise ValueError("num_shards cannot exceed the number of samples")
    if hdf5_output is not None and webdataset_output is not None:
        if Path(hdf5_output).resolve() == Path(webdataset_output).resolve():
            raise ValueError("HDF5 and WebDataset outputs must be different directories")

    results = {}
    if hdf5_output is not None:
        output = _empty_output_directory(hdf5_output)
        results["hdf5"] = _write_sharded_hdf5(
            source, output, classes, by_class, num_shards, seed, progress
        )
    if webdataset_output is not None:
        output = _empty_output_directory(webdataset_output)
        results["webdataset"] = _write_webdataset(
            source, output, classes, by_class, num_shards, seed, name, progress
        )
    return results
