import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
from helpers import annotation_for, figure_image

from neural_detect.cache import build_cache, cache_exists, load_cache
from neural_detect.datasets import extract, parse_annotation
from neural_detect.evaluation import load_detections, save_detections
from neural_detect.splits import make_split

ANNOTATION = """<annotation>
  <filename>000005.jpg</filename>
  <size><width>500</width><height>375</height><depth>3</depth></size>
  <object><name>person</name><truncated>1</truncated><difficult>0</difficult>
    <bndbox><xmin>11</xmin><ymin>21</ymin><xmax>110</xmax><ymax>300</ymax></bndbox></object>
  <object><name>dog</name><truncated>0</truncated><difficult>0</difficult>
    <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
  <object><name>person</name><truncated>0</truncated><difficult>1</difficult>
    <bndbox><xmin>201</xmin><ymin>101</ymin><xmax>240</xmax><ymax>200</ymax></bndbox></object>
</annotation>"""


def test_parse_annotation_keeps_people_only() -> None:
    annotation = parse_annotation(ANNOTATION)
    assert annotation.image_id == "000005"
    assert (annotation.width, annotation.height) == (500, 375)
    assert np.allclose(annotation.boxes, [[10, 20, 110, 300], [200, 100, 240, 200]])
    assert annotation.difficult.tolist() == [False, True]
    assert annotation.truncated.tolist() == [True, False]


def test_extract_rejects_paths_outside_destination(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as bundle:
        payload = b"x"
        info = tarfile.TarInfo("../escaped.txt")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    with pytest.raises(tarfile.FilterError):
        extract(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_split_is_disjoint_and_reproducible() -> None:
    trainval = [f"{index:06d}" for index in range(100)]
    test = [f"9{index:05d}" for index in range(30)]
    split = make_split(trainval, test, validation_count=20, seed=3)
    assert len(split["validation"]) == 20 and len(split["train"]) == 80
    assert set(split["train"]).isdisjoint(split["validation"])
    assert split == make_split(trainval, test, validation_count=20, seed=3)


def test_cache_round_trip(tmp_path: Path, rng: np.random.Generator) -> None:
    images = {"a": figure_image(rng, [(40, 30, 80, 150)]), "b": figure_image(rng, [])}
    annotations = {
        key: annotation_for(key, image, [(40, 30, 80, 150)] if key == "a" else [])
        for key, image in images.items()
    }
    build_cache("unit", ["a", "b"], annotations, images.__getitem__, 160, tmp_path)
    assert cache_exists("unit", ["a", "b"], 160, tmp_path)
    ids, cached, geometries, boxes = load_cache("unit", 160, tmp_path)
    assert ids == ["a", "b"] and cached.shape == (2, 160, 160, 3)
    assert np.allclose(geometries[0].to_original(boxes[0]), [[40, 30, 80, 150]])
    assert boxes[1].shape == (0, 4)


def test_detections_round_trip(tmp_path: Path) -> None:
    detections = {
        "a": (np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([0.5])),
        "b": (np.zeros((0, 4)), np.zeros(0)),
    }
    path = tmp_path / "found.npz"
    save_detections(detections, path)
    loaded = load_detections(path)
    assert list(loaded) == ["a", "b"]
    assert np.allclose(loaded["a"][0], detections["a"][0]) and loaded["b"][1].size == 0
