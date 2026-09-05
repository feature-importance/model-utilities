import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tarfile

import h5py
import numpy as np
from PIL import Image
import pytest

from model_utilities.datasets import (
    ImageNetShardedHDF5,
    ImageNetWebDataset,
    ImageNetWIDS,
    make_wids_sampler,
    repack_imagenet_hdf5,
)


def encoded_image(colour, image_format):
    output = io.BytesIO()
    Image.new("RGB", (12, 10), colour).save(output, format=image_format)
    return output.getvalue()


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
