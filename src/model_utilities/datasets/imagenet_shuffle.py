"""Shuffle existing local ImageNet tar shards without changing their membership."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile

from .imagenet_sharded import IMAGE_EXTENSIONS, WIDS_MANIFEST
from .imagenet_subsets_wids import _local_shard_path, _sample_name


def _sample_records(archive, expected_samples, num_classes):
    records = {}
    previous = None
    for member in archive:
        if not member.isfile():
            raise ValueError(f"Only regular tar members are supported: {member.name}")
        key, extension = _sample_name(member.name)
        if not key or not extension:
            raise ValueError(f"Not a WebDataset sample component: {member.name}")
        if key != previous and key in records:
            raise ValueError(f"Non-contiguous or repeated sample key: {key}")
        previous = key
        components = records.setdefault(key, {})
        if extension in components:
            raise ValueError(f"Duplicate sample component: {member.name}")
        components[extension] = member
    if len(records) != expected_samples:
        raise ValueError(f"Manifest expects {expected_samples} samples, found {len(records)}")
    for key, components in records.items():
        image_count = sum(f".{extension}" in components for extension in IMAGE_EXTENSIONS)
        if image_count != 1 or ".cls" not in components:
            raise ValueError(f"Sample {key} must contain one image and a .cls label")
        with archive.extractfile(components[".cls"]) as stream:
            target = int(stream.read().decode("ascii"))
        if target < 0 or (num_classes is not None and target >= num_classes):
            raise ValueError(f"Invalid class label {target} for {key}")
    return list(records.values())


def _verify_payloads(path, expected):
    """Check output member order, names and every copied payload's SHA-256."""
    with tarfile.open(path, mode="r:") as archive:
        actual = archive.getmembers()
        if [member.name for member in actual] != list(expected):
            raise ValueError(f"Written component order does not match: {path}")
        for member in actual:
            with archive.extractfile(member) as stream:
                digest = hashlib.sha256(stream.read()).hexdigest()
            if digest != expected[member.name]:
                raise ValueError(f"Payload verification failed for {member.name} in {path}")


def _shuffle_shard(source, output, expected_samples, num_classes, seed):
    temporary = output.with_suffix(".tar.partial")
    try:
        with tarfile.open(source, mode="r:") as original:
            records = _sample_records(original, expected_samples, num_classes)
            random.Random(seed).shuffle(records)
            digests = {}
            with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as shuffled:
                for components in records:
                    for member in components.values():
                        # Keep only one encoded component in memory, not the whole shard.
                        with original.extractfile(member) as stream:
                            payload = stream.read()
                        digests[member.name] = hashlib.sha256(payload).hexdigest()
                        shuffled.addfile(copy.copy(member), io.BytesIO(payload))
        _verify_payloads(temporary, digests)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def shuffle_imagenet_shards(source, output, *, seed=0, progress=None):
    """Copy flat local WIDS shards into a new directory, independently shuffled.

    Only sample order within each shard changes. Encoded payloads and labels are
    verified byte-for-byte using SHA-256. No images are decoded or re-encoded.
    The destination must not exist; a manifest is published only after success.
    """
    source = Path(source).resolve()
    output = Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be separate from, and not inside, the source")
    manifest_path = source / WIDS_MANIFEST
    with open(manifest_path, encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("datasets") or metadata.get("base"):
        raise ValueError("Only flat manifests with directly listed local shards are supported")
    shards = metadata.get("shardlist")
    if not isinstance(shards, list) or not shards:
        raise ValueError("The source manifest must contain a non-empty shardlist")
    paths = [_local_shard_path(source, shard["url"]) for shard in shards]
    if len(set(paths)) != len(paths):
        raise ValueError("The manifest lists a shard more than once")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    counts = [int(shard["nsamples"]) for shard in shards]
    if any(count < 0 for count in counts):
        raise ValueError("Shard sample counts must be non-negative")
    total = sum(counts)
    if "num_samples" in metadata and int(metadata["num_samples"]) != total:
        raise ValueError("Manifest total does not match its shard sample counts")
    num_classes = len(metadata["classes"]) if "classes" in metadata else None

    # A fresh directory also prevents accidentally reusing shard/subset indexes.
    output.mkdir(parents=True, exist_ok=False)
    result = {key: copy.deepcopy(metadata[key]) for key in
              ("wids_version", "name", "classes", "class_to_idx", "description") if key in metadata}
    result["wids_version"] = metadata.get("wids_version", 1)
    result["num_samples"] = total
    result["shardlist"] = []
    result["shuffle"] = {
        "algorithm": "independent-per-shard-python-random-v1",
        "seed": seed,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "payload_verification": "sha256",
    }
    for index, (path, count) in enumerate(zip(paths, counts)):
        # Stable independent seeds, without Python's process-randomised hash().
        shard_seed = int.from_bytes(hashlib.sha256(f"{seed}:{index}".encode("ascii")).digest(), "big")
        destination = output / f"shuffled-{index:05d}.tar"
        _shuffle_shard(path, destination, count, num_classes, shard_seed)
        result["shardlist"].append({
            "url": destination.name, "nsamples": count,
            "filesize": destination.stat().st_size,
        })
        if progress is not None:
            progress(index + 1, len(shards))
    temporary = output / f"{WIDS_MANIFEST}.partial"
    with open(temporary, "x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    temporary.replace(output / WIDS_MANIFEST)
    return result
