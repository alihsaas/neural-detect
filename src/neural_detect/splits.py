import json
from pathlib import Path

import numpy as np

from neural_detect.config import SPLIT_SEED, VALIDATION_COUNT


def make_split(
    trainval_ids: list[str],
    test_ids: list[str],
    validation_count: int = VALIDATION_COUNT,
    seed: int = SPLIT_SEED,
) -> dict[str, list[str]]:
    order = np.random.default_rng(seed).permutation(sorted(trainval_ids))
    validation = sorted(str(value) for value in order[:validation_count])
    train = sorted(str(value) for value in order[validation_count:])
    return {"train": train, "validation": validation, "test": sorted(test_ids)}


def save_split(split: dict[str, list[str]], person_boxes: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SPLIT_SEED,
        "counts": {name: len(values) for name, values in split.items()},
        "person_boxes": person_boxes,
        **split,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="ascii")


def load_split(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="ascii"))
    return {name: payload[name] for name in ("train", "validation", "test")}
