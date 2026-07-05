import hashlib
import json
import ssl
import tarfile
import urllib.request
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import truststore

from neural_detect.config import (
    BACKBONE_URL,
    CLASS_NAME,
    DATA_DIR,
    VOC_ARCHIVES,
    VOC_BASE_URL,
    VOC_DIR,
    WEIGHTS_DIR,
)


@dataclass
class Annotation:
    image_id: str
    width: int
    height: int
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    difficult: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    truncated: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


def digest_of(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with urllib.request.urlopen(url, context=context, timeout=60) as response:
        with partial.open("wb") as handle:
            while block := response.read(1 << 20):
                handle.write(block)
    partial.replace(target)


def extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive) as bundle:
        bundle.extractall(destination, filter="data")


def ensure_voc(data_dir: Path = DATA_DIR) -> dict:
    record = {}
    for name, expected_md5 in VOC_ARCHIVES.items():
        archive = data_dir / "archives" / name
        if not archive.exists():
            print(f"Downloading {name}")
            fetch(VOC_BASE_URL + name, archive)
        md5 = digest_of(archive, "md5")
        if md5 != expected_md5:
            raise SystemExit(f"{name}: MD5 {md5} does not match the published {expected_md5}")
        record[name] = {
            "bytes": archive.stat().st_size,
            "md5": md5,
            "sha256": digest_of(archive, "sha256"),
        }
        print(f"Extracting {name}")
        extract(archive, data_dir)
    return record


def ensure_backbone(weights_dir: Path = WEIGHTS_DIR) -> dict:
    name = BACKBONE_URL.rsplit("/", 1)[1]
    target = weights_dir / name
    if not target.exists():
        print(f"Downloading {name}")
        fetch(BACKBONE_URL, target)
    sha256 = digest_of(target, "sha256")
    prefix = name.rsplit("-", 1)[1].split(".")[0]
    if not sha256.startswith(prefix):
        raise SystemExit(f"{name}: SHA-256 {sha256} does not start with {prefix}")
    return {name: {"bytes": target.stat().st_size, "sha256": sha256}}


def write_download_record(record: dict, data_dir: Path = DATA_DIR) -> None:
    (data_dir / "download.json").write_text(json.dumps(record, indent=2), encoding="ascii")


def image_ids(image_set: str, voc_dir: Path = VOC_DIR) -> list[str]:
    path = voc_dir / "ImageSets" / "Main" / f"{image_set}.txt"
    return [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]


def parse_annotation(text: str, class_name: str = CLASS_NAME) -> Annotation:
    root = ElementTree.fromstring(text)
    size = root.find("size")
    image_id = Path(root.findtext("filename", "")).stem
    boxes, difficult, truncated = [], [], []
    for item in root.iter("object"):
        if item.findtext("name", "").strip() != class_name:
            continue
        box = item.find("bndbox")
        x1, y1, x2, y2 = (float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((x1 - 1.0, y1 - 1.0, x2, y2))
        difficult.append(item.findtext("difficult", "0").strip() == "1")
        truncated.append(item.findtext("truncated", "0").strip() == "1")
    return Annotation(
        image_id,
        int(size.findtext("width")),
        int(size.findtext("height")),
        np.array(boxes, dtype=np.float64).reshape(-1, 4),
        np.array(difficult, dtype=bool),
        np.array(truncated, dtype=bool),
    )


def load_annotation(image_id: str, voc_dir: Path = VOC_DIR) -> Annotation:
    path = voc_dir / "Annotations" / f"{image_id}.xml"
    annotation = parse_annotation(path.read_text(encoding="utf-8"))
    annotation.image_id = image_id
    return annotation


def load_annotations(ids: list[str], voc_dir: Path = VOC_DIR) -> dict[str, Annotation]:
    return {image_id: load_annotation(image_id, voc_dir) for image_id in ids}


def load_image(image_id: str, voc_dir: Path = VOC_DIR) -> np.ndarray:
    path = voc_dir / "JPEGImages" / f"{image_id}.jpg"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
