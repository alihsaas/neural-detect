import csv
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from neural_detect.config import STRICT_IOU_THRESHOLD
from neural_detect.datasets import Annotation
from neural_detect.metrics import (
    Detections,
    average_precision,
    bootstrap_ap_interval,
    match_detections,
    max_recall,
)


def run_detector(
    detect: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    ids: list[str],
    load: Callable[[str], np.ndarray],
    label: str = "",
) -> tuple[Detections, float]:
    detections: Detections = {}
    started = time.perf_counter()
    for index, image_id in enumerate(ids, start=1):
        detections[image_id] = detect(load(image_id))
        if label and index % 500 == 0:
            print(f"{label}: {index}/{len(ids)} images")
    return detections, time.perf_counter() - started


def ap50(detections: Detections, annotations: dict[str, Annotation]) -> float:
    return average_precision(match_detections(detections, annotations))


def summarize(
    method: str, detections: Detections, annotations: dict[str, Annotation]
) -> dict[str, float | str | int]:
    matched = match_detections(detections, annotations)
    low, high = bootstrap_ap_interval(matched)
    strict = match_detections(detections, annotations, STRICT_IOU_THRESHOLD)
    return {
        "method": method,
        "images": len(annotations),
        "person_boxes": int(matched.positives_per_image.sum()),
        "ap50": average_precision(matched),
        "ap50_ci_low": low,
        "ap50_ci_high": high,
        "ap75": average_precision(strict),
        "max_recall50": max_recall(matched),
    }


def save_detections(detections: Detections, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(detections)
    counts = np.array([len(detections[image_id][1]) for image_id in ids], dtype=np.int64)
    boxes = np.concatenate([detections[image_id][0].reshape(-1, 4) for image_id in ids])
    scores = np.concatenate([detections[image_id][1].reshape(-1) for image_id in ids])
    np.savez_compressed(path, ids=np.array(ids), counts=counts, boxes=boxes, scores=scores)


def load_detections(path: Path) -> Detections:
    payload = np.load(path)
    edges = np.concatenate(([0], np.cumsum(payload["counts"])))
    return {
        str(image_id): (
            payload["boxes"][edges[index] : edges[index + 1]],
            payload["scores"][edges[index] : edges[index + 1]],
        )
        for index, image_id in enumerate(payload["ids"])
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.4f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="ascii")


def print_summary(rows: list[dict]) -> None:
    print(f"{'method':28s} {'AP50':>7s} {'95% CI':>15s} {'AP75':>7s} {'recall':>7s}")
    for row in rows:
        interval = f"{row['ap50_ci_low']:.3f}-{row['ap50_ci_high']:.3f}"
        print(
            f"{row['method']:28s} {row['ap50']:7.3f} {interval:>15s} {row['ap75']:7.3f}"
            f" {row['max_recall50']:7.3f}"
        )
