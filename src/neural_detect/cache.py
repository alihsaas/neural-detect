import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from neural_detect.boxes import Letterbox, letterbox
from neural_detect.config import CACHE_DIR, INPUT_SIZE
from neural_detect.datasets import Annotation


def build_cache(
    name: str,
    ids: list[str],
    annotations: dict[str, Annotation],
    load: Callable[[str], np.ndarray],
    size: int = INPUT_SIZE,
    cache_dir: Path = CACHE_DIR,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(
        cache_dir / f"{name}_{size}.npy", mode="w+", dtype=np.uint8, shape=(len(ids), size, size, 3)
    )
    records = []
    for index, image_id in enumerate(ids):
        canvas, geometry = letterbox(load(image_id), size)
        images[index] = canvas
        boxes = geometry.to_input(annotations[image_id].boxes)
        records.append({"id": image_id, "geometry": geometry.__dict__, "boxes": boxes.tolist()})
    images.flush()
    del images
    (cache_dir / f"{name}_{size}.json").write_text(json.dumps(records), encoding="ascii")


def load_cache(
    name: str, size: int = INPUT_SIZE, cache_dir: Path = CACHE_DIR
) -> tuple[list[str], np.ndarray, list[Letterbox], list[np.ndarray]]:
    images = np.load(cache_dir / f"{name}_{size}.npy", mmap_mode="r")
    records = json.loads((cache_dir / f"{name}_{size}.json").read_text(encoding="ascii"))
    ids = [record["id"] for record in records]
    geometries = [Letterbox(**record["geometry"]) for record in records]
    boxes = [np.array(record["boxes"], dtype=np.float64).reshape(-1, 4) for record in records]
    return ids, images, geometries, boxes


def cache_exists(
    name: str, ids: list[str], size: int = INPUT_SIZE, cache_dir: Path = CACHE_DIR
) -> bool:
    path = cache_dir / f"{name}_{size}.json"
    if not path.exists() or not (cache_dir / f"{name}_{size}.npy").exists():
        return False
    records = json.loads(path.read_text(encoding="ascii"))
    return [record["id"] for record in records] == ids
