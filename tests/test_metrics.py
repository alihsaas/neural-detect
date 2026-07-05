import numpy as np
import pytest

from neural_detect.datasets import Annotation
from neural_detect.metrics import (
    average_precision,
    bootstrap_ap_interval,
    match_detections,
    max_recall,
)


def truth(image_id: str, boxes: list, difficult: list | None = None) -> Annotation:
    count = len(boxes)
    return Annotation(
        image_id,
        100,
        100,
        np.array(boxes, dtype=float).reshape(-1, 4),
        np.array(difficult or [False] * count, dtype=bool),
        np.zeros(count, dtype=bool),
    )


def found(boxes: list, scores: list) -> tuple[np.ndarray, np.ndarray]:
    return np.array(boxes, dtype=float).reshape(-1, 4), np.array(scores, dtype=float)


def test_perfect_detections_give_ap_one() -> None:
    annotations = {"a": truth("a", [[0, 0, 10, 10]]), "b": truth("b", [[20, 20, 40, 40]])}
    detections = {"a": found([[0, 0, 10, 10]], [0.9]), "b": found([[20, 20, 40, 40]], [0.8])}
    matched = match_detections(detections, annotations)
    assert average_precision(matched) == pytest.approx(1.0)
    assert max_recall(matched) == pytest.approx(1.0)


def test_false_positive_ranked_first_halves_precision() -> None:
    annotations = {"a": truth("a", [[0, 0, 10, 10]])}
    detections = {"a": found([[50, 50, 60, 60], [0, 0, 10, 10]], [0.9, 0.8])}
    assert average_precision(match_detections(detections, annotations)) == pytest.approx(0.5)


def test_duplicate_is_false_positive_and_difficult_is_ignored() -> None:
    annotations = {"a": truth("a", [[0, 0, 10, 10], [50, 50, 70, 70]], [False, True])}
    detections = {"a": found([[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 70, 70]], [0.9, 0.8, 0.95])}
    matched = match_detections(detections, annotations)
    assert matched.positives_per_image.tolist() == [1]
    assert matched.true_positive.sum() == 1
    assert matched.false_positive.sum() == 1
    assert average_precision(matched) == pytest.approx(1.0)


def test_missing_detections_give_zero() -> None:
    annotations = {"a": truth("a", [[0, 0, 10, 10]])}
    assert average_precision(match_detections({}, annotations)) == 0.0


def test_unit_weights_match_plain_ap_and_interval_brackets_it(rng: np.random.Generator) -> None:
    annotations, detections = {}, {}
    for index in range(40):
        box = [10, 10, 50, 50]
        annotations[str(index)] = truth(str(index), [box])
        shift = rng.integers(0, 30)
        detections[str(index)] = found([[10 + shift, 10, 50 + shift, 50]], [rng.random()])
    matched = match_detections(detections, annotations)
    plain = average_precision(matched)
    assert average_precision(matched, np.ones(40)) == pytest.approx(plain)
    low, high = bootstrap_ap_interval(matched, samples=200)
    assert low <= plain <= high
