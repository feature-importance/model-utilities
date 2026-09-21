import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tarfile
from contextlib import contextmanager

import h5py
import numpy as np
from PIL import Image
import pytest

from model_utilities.datasets import (
    ImageNetShardedHDF5,
    ImageNet50WIDS,
    ImageNet100WIDS,
    ImageNetWebDataset,
    ImageNetWIDS,
    make_wids_sampler,
    repack_imagenet_hdf5,
)
from model_utilities.datasets.imagenet_subsets_hdf5 import (
    IMAGENET_50_SUBSET,
    IMAGENET_100_SUBSET,
)
import model_utilities.datasets.imagenet_subsets_wids as subsets_wids
import model_utilities.datasets.imagenet_repack as repack


def encoded_image(colour, image_format):
    output = io.BytesIO()
    Image.new("RGB", (12, 10), colour).save(output, format=image_format)
    return output.getvalue()


def add_tar_member(archive, name, value):
    member = tarfile.TarInfo(name)
    member.size = len(value)
    archive.addfile(member, io.BytesIO(value))


@pytest.fixture
def repacked_dataset(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    classes = ["n00000001", "n00000002"]
    destinations = []
    originals = []
    for target, class_name in enumerate(classes):
        values = [
            encoded_image((20 + target * 100, index * 30, 80), "PNG" if index < 2 else "JPEG")
            for index in range(3)
        ]
        with h5py.File(source / f"{class_name}.hdf5", "w") as handle:
            data = handle.create_dataset(
                "data", (len(values),), dtype=h5py.vlen_dtype(np.dtype("uint8"))
            )
            for index, value in enumerate(values):
                data[index] = np.frombuffer(value, dtype="uint8")
                destinations.append((class_name, index))
                originals.append((target, value))
    with open(source / "dest.p", "wb") as stream:
        pickle.dump(destinations, stream)

    hdf5_output = tmp_path / "hdf5"
    webdataset_output = tmp_path / "webdataset"
    repack_imagenet_hdf5(
        source,
        hdf5_output=hdf5_output,
        webdataset_output=webdataset_output,
        num_shards=2,
        seed=7,
    )
    return source, hdf5_output, webdataset_output, classes, originals


@pytest.fixture
def subset_webdataset(tmp_path):
    root = tmp_path / "subsets"
    root.mkdir()
    outside_class = "n00000000"
    full_classes = sorted(set(IMAGENET_100_SUBSET) | {outside_class})
    class_to_idx = {name: index for index, name in enumerate(full_classes)}
    included = [
        IMAGENET_50_SUBSET[0],
        IMAGENET_100_SUBSET[1],
        outside_class,
        IMAGENET_50_SUBSET[1],
    ]
    shard_samples = [included[:2], included[2:]]
    shardlist = []
    for shard_index, classes in enumerate(shard_samples):
        path = root / f"train-{shard_index:05d}.tar"
        with tarfile.open(path, mode="w") as archive:
            for sample_index, class_name in enumerate(classes):
                key = f"sample-{shard_index}-{sample_index}"
                add_tar_member(
                    archive,
                    f"{key}.png",
                    encoded_image((shard_index * 80, sample_index * 80, 30), "PNG"),
                )
                add_tar_member(
                    archive,
                    f"{key}.cls",
                    str(class_to_idx[class_name]).encode("ascii"),
                )
        shardlist.append(
            {"url": path.name, "nsamples": len(classes), "filesize": path.stat().st_size}
        )
    with open(root / "dataset.json", "w", encoding="utf-8") as stream:
        json.dump(
            {
                "wids_version": 1,
                "name": "subset-test",
                "num_samples": len(included),
                "classes": full_classes,
                "shardlist": shardlist,
            },
            stream,
        )
    return root


def assert_decoded_dataset(dataset, classes):
    assert len(dataset) == 6
    assert dataset.classes == classes
    targets = []
    for index in range(len(dataset)):
        image, target = dataset[index]
        assert image.mode == "RGB"
        assert image.size == (12, 10)
        targets.append(target)
    assert sorted(targets) == [0, 0, 0, 1, 1, 1]


def test_sharded_hdf5_is_indexed_and_limits_handles(repacked_dataset):
    _, hdf5_output, _, classes, originals = repacked_dataset
    dataset = ImageNetShardedHDF5(hdf5_output, max_open_files=1)
    assert_decoded_dataset(dataset, classes)
    encoded = [dataset._load_encoded(index) for index in range(len(dataset))]
    assert sorted(encoded) == sorted((value, target) for target, value in originals)
    assert len(dataset._open_files) == 1
    assert dataset[-1][1] in (0, 1)
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    dataset.close()
    assert dataset._open_files == {}


def test_hdf5_manifest_matches_flat_storage(repacked_dataset):
    _, hdf5_output, _, classes, originals = repacked_dataset
    with open(hdf5_output / "manifest.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    assert manifest["num_samples"] == len(originals)
    assert manifest["classes"] == classes
    assert sum(shard["nsamples"] for shard in manifest["shards"]) == len(originals)
    for shard in manifest["shards"]:
        with h5py.File(hdf5_output / shard["file"], "r") as handle:
            assert len(handle["offsets"]) == len(handle["targets"]) + 1
            assert int(handle["offsets"][-1]) == len(handle["data"])


def test_wids_reads_tar_shards_by_index(repacked_dataset):
    pytest.importorskip("wids")
    _, _, webdataset_output, classes, _ = repacked_dataset
    dataset = ImageNetWIDS(webdataset_output, lru_size=1)
    assert_decoded_dataset(dataset, classes)
    sampler = make_wids_sampler(dataset, chunksize=2, shuffle=False)
    assert sorted(sampler) == list(range(len(dataset)))
    dataset.close()


def test_webdataset_streams_tar_shards(repacked_dataset):
    pytest.importorskip("webdataset")
    _, _, webdataset_output, classes, _ = repacked_dataset
    dataset = ImageNetWebDataset(webdataset_output, shuffle=False)
    assert dataset.classes == classes
    samples = list(dataset)
    assert len(samples) == 6
    assert sorted(target for _, target in samples) == [0, 0, 0, 1, 1, 1]
    assert all(image.mode == "RGB" and image.size == (12, 10) for image, _ in samples)


def test_webdataset_tar_preserves_encoded_bytes(repacked_dataset):
    _, _, webdataset_output, _, originals = repacked_dataset
    encoded = []
    for path in webdataset_output.glob("*.tar"):
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if not member.name.endswith(".cls"):
                    encoded.append(archive.extractfile(member).read())
    assert sorted(encoded) == sorted(value for _, value in originals)


def test_repacker_refuses_nonempty_output(repacked_dataset):
    source, hdf5_output, _, _, _ = repacked_dataset
    with pytest.raises(FileExistsError):
        repack_imagenet_hdf5(source, hdf5_output=hdf5_output, num_shards=2)


def test_repacking_shuffles_reproducibly_in_both_formats(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    classes = [f"n{index:08d}" for index in range(4)]
    destinations = []
    originals = {}
    for target, class_name in enumerate(classes):
        with h5py.File(source / f"{class_name}.hdf5", "w") as handle:
            data = handle.create_dataset("data", (16,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            for index in range(16):
                value = encoded_image((target * 50, index * 10, 30), "PNG")
                data[index] = np.frombuffer(value, dtype="uint8")
                destinations.append((class_name, index))
                originals[f"{class_name}_{index:08d}"] = (value, target)
    with open(source / "dest.p", "wb") as stream:
        pickle.dump(destinations, stream)

    orders = {}
    for label, kwargs in [("default", {}), ("zero", {"seed": 0}), ("other", {"seed": 1})]:
        root = tmp_path / label
        phases = []
        manifests = repack_imagenet_hdf5(
            source, hdf5_output=root / "hdf5", webdataset_output=root / "tar", num_shards=2,
            progress=lambda phase, done, total: phases.append((phase, done, total)), **kwargs,
        )
        seed = kwargs.get("seed", 0)
        assert ("hdf5-shuffling", 2, 2) in phases
        assert ("webdataset-shuffling", 2, 2) in phases
        assert manifests["hdf5"]["shuffle"]["seed"] == seed
        assert manifests["webdataset"]["shuffle"]["seed"] == seed
        hdf5 = ImageNetShardedHDF5(root / "hdf5")
        position = 0
        orders[label] = []
        for shard_index, shard in enumerate(manifests["webdataset"]["shardlist"]):
            path = root / "tar" / shard["url"]
            with tarfile.open(path, "r:") as archive:
                members = archive.getmembers()
                keys, targets = [], []
                for index in range(0, len(members), 2):
                    image, target = members[index:index + 2]
                    key = image.name.rsplit(".", 1)[0]
                    assert target.name == key + ".cls"
                    value = archive.extractfile(image).read()
                    target_value = int(archive.extractfile(target).read())
                    assert (value, target_value) == originals[key]
                    assert hdf5._load_encoded(position) == (value, target_value)
                    keys.append(key)
                    targets.append(target_value)
                    position += 1
            expected_keys = {
                f"{class_name}_{index:08d}"
                for target, class_name in enumerate(classes)
                for assigned, index in repack._assignment(range(16), target, 2, seed)
                if assigned == shard_index
            }
            assert len(keys) == len(set(keys)) == shard["nsamples"]
            assert set(keys) == expected_keys
            assert targets != sorted(targets)
            assert [targets.count(target) for target in range(4)] == [8] * 4
            assert path.stat().st_size == shard["filesize"]
            orders[label].append(keys)
        hdf5.close()
        assert position == len(originals)
        assert not list(root.rglob("*.partial"))
    assert orders["default"] == orders["zero"]
    assert orders["default"] != orders["other"]
    for index in range(2):
        name = f"train-{index:05d}.tar"
        assert (tmp_path / "default/tar" / name).read_bytes() == (tmp_path / "zero/tar" / name).read_bytes()


def test_shard_shuffle_seeds_are_independent():
    first = repack._sample_order(100, 0, 0)
    assert sorted(first) == list(range(100))
    assert first == repack._sample_order(100, 0, 0)
    assert first != repack._sample_order(100, 0, 1)
    assert first != repack._sample_order(100, 1, 0)
    assert repack._sample_order(0, 0, 0) == []


@pytest.mark.parametrize("output_kind,helper,manifest_name", [
    ("hdf5_output", "_shuffle_hdf5", "manifest.json"),
    ("webdataset_output", "_shuffle_tar", "dataset.json"),
])
def test_shuffle_failure_does_not_publish_manifest(repacked_dataset, tmp_path, monkeypatch,
                                                  output_kind, helper, manifest_name):
    source, _, _, _, _ = repacked_dataset
    output = tmp_path / "failed"

    def fail(*args):
        raise ValueError("shuffle verification failed")

    monkeypatch.setattr(repack, helper, fail)
    with pytest.raises(ValueError, match="verification failed"):
        repack_imagenet_hdf5(source, num_shards=2, **{output_kind: output})
    assert not (output / manifest_name).exists()


@pytest.mark.parametrize("seed_args,expected_seed", [([], 0), (["--seed", "42"], 42)])
def test_repack_cli_shuffle_seed(repacked_dataset, tmp_path, seed_args, expected_seed):
    source, _, _, _, _ = repacked_dataset
    script = Path(__file__).resolve().parents[3] / "tools/imagenet/repack.py"
    output = tmp_path / "cli"
    completed = subprocess.run([
        sys.executable, str(script), "--input", str(source),
        "--webdataset-output", str(output), "--num-shards", "2", *seed_args,
    ], capture_output=True, text=True, check=True)
    assert "[webdataset-shuffling] 2/2 shards" in completed.stdout
    metadata = json.loads((output / "dataset.json").read_text())
    assert metadata["shuffle"]["seed"] == expected_seed


def test_imagenet_wids_subsets_reuse_existing_shards(subset_webdataset):
    pytest.importorskip("wids")
    tar_mtimes = {path: path.stat().st_mtime_ns for path in subset_webdataset.glob("*.tar")}

    imagenet100 = ImageNet100WIDS(subset_webdataset)
    assert len(imagenet100) == 3
    assert imagenet100.classes == sorted(IMAGENET_100_SUBSET)
    expected100 = {
        imagenet100.class_to_idx[IMAGENET_50_SUBSET[0]],
        imagenet100.class_to_idx[IMAGENET_100_SUBSET[1]],
        imagenet100.class_to_idx[IMAGENET_50_SUBSET[1]],
    }
    assert {imagenet100[index][1] for index in range(len(imagenet100))} == expected100
    imagenet100.close()

    imagenet50 = ImageNet50WIDS(
        subset_webdataset,
        target_transform=lambda target: target + 100,
    )
    assert len(imagenet50) == 2
    assert imagenet50.classes == sorted(IMAGENET_50_SUBSET)
    expected50 = {
        imagenet50.class_to_idx[IMAGENET_50_SUBSET[0]] + 100,
        imagenet50.class_to_idx[IMAGENET_50_SUBSET[1]] + 100,
    }
    assert {imagenet50[index][1] for index in range(len(imagenet50))} == expected50
    assert imagenet50[-1][0].size == (12, 10)
    with pytest.raises(IndexError):
        imagenet50[len(imagenet50)]
    imagenet50.close()

    assert tar_mtimes == {
        path: path.stat().st_mtime_ns for path in subset_webdataset.glob("*.tar")
    }
    assert len(list(subset_webdataset.glob("*.npy"))) == 3


def test_imagenet_wids_subset_uses_cached_index(subset_webdataset, monkeypatch):
    pytest.importorskip("wids")
    first = ImageNet50WIDS(subset_webdataset)
    first.close()
    monkeypatch.setattr(
        subsets_wids,
        "_scan_shard_targets",
        lambda *args, **kwargs: pytest.fail("tar shards should not be rescanned"),
    )
    second = ImageNet50WIDS(subset_webdataset)
    assert len(second) == 2
    second.close()


def test_imagenet_wids_subset_defaults_to_data_directory(
    subset_webdataset, monkeypatch
):
    pytest.importorskip("wids")
    first = ImageNet50WIDS(subset_webdataset)
    first.close()
    assert len(list(subset_webdataset.glob("imagenet-targets-*.npy"))) == 1
    assert len(list(subset_webdataset.glob("imagenet-subset-*.npy"))) == 1

    monkeypatch.setattr(
        subsets_wids,
        "_exclusive_lock",
        lambda path: pytest.fail("a completed index should not require a writable lock"),
    )
    monkeypatch.setattr(
        subsets_wids,
        "_scan_shard_targets",
        lambda *args, **kwargs: pytest.fail("tar shards should not be rescanned"),
    )
    second = ImageNet50WIDS(subset_webdataset)
    assert len(second) == 2
    second.close()


def test_imagenet_wids_subset_falls_back_to_tmp(
    subset_webdataset, tmp_path, monkeypatch
):
    pytest.importorskip("wids")
    fallback = tmp_path / "fallback"
    monkeypatch.setenv("WIDS_SUBSET_CACHE", str(fallback))
    original_lock = subsets_wids._exclusive_lock

    @contextmanager
    def deny_data_directory(path):
        if Path(path).parent == subset_webdataset:
            raise PermissionError("read-only dataset directory")
        with original_lock(path):
            yield

    monkeypatch.setattr(subsets_wids, "_exclusive_lock", deny_data_directory)
    dataset = ImageNet50WIDS(subset_webdataset)
    assert len(dataset) == 2
    dataset.close()
    assert not list(subset_webdataset.glob("*.npy"))
    assert len(list(fallback.glob("*.npy"))) == 2


def test_benchmark_smoke_test_all_backends(repacked_dataset, tmp_path):
    pytest.importorskip("wids")
    pytest.importorskip("webdataset")
    source, hdf5_output, webdataset_output, _, _ = repacked_dataset
    repository = Path(__file__).resolve().parents[3]
    output = tmp_path / "benchmark.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "tools/imagenet/benchmark.py"),
            "--original-root",
            str(source),
            "--sharded-hdf5-root",
            str(hdf5_output),
            "--webdataset-root",
            str(webdataset_output),
            "--workers",
            "2",
            "--batch-size",
            "2",
            "--samples-per-rank",
            "4",
            "--warmup-batches",
            "0",
            "--output",
            str(output),
        ],
        check=False,
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    with open(output, encoding="utf-8") as stream:
        results = json.load(stream)
    assert [result["backend"] for result in results] == [
        "original-hdf5",
        "original-hdf5-handles",
        "sharded-hdf5",
        "wids",
        "webdataset",
    ]
    assert all(result["samples"] >= 4 for result in results)
