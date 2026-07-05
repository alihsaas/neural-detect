from dataclasses import dataclass

import numpy as np

from neural_detect.boxes import iou_matrix
from neural_detect.config import BOOTSTRAP_SAMPLES, IOU_THRESHOLD
from neural_detect.datasets import Annotation

Detections = dict[str, tuple[np.ndarray, np.ndarray]]


@dataclass
class MatchedDetections:
    scores: np.ndarray
    true_positive: np.ndarray
    false_positive: np.ndarray
    image_index: np.ndarray
    positives_per_image: np.ndarray


def match_detections(
    detections: Detections, annotations: dict[str, Annotation], iou_threshold: float = IOU_THRESHOLD
) -> MatchedDetections:
    ids = list(annotations)
    scores, true_positive, false_positive, image_index = [], [], [], []
    positives = np.zeros(len(ids), dtype=np.int64)
    for index, image_id in enumerate(ids):
        truth = annotations[image_id]
        positives[index] = int((~truth.difficult).sum())
        boxes, image_scores = detections.get(image_id, (np.zeros((0, 4)), np.zeros(0)))
        if len(image_scores) == 0:
            continue
        order = np.argsort(-image_scores, kind="stable")
        boxes, image_scores = boxes[order], image_scores[order]
        overlaps = iou_matrix(boxes, truth.boxes)
        taken = np.zeros(len(truth.boxes), dtype=bool)
        hit = np.zeros(len(image_scores), dtype=bool)
        miss = np.zeros(len(image_scores), dtype=bool)
        for row in range(len(image_scores)):
            if overlaps.shape[1] == 0:
                miss[row] = True
                continue
            best = int(np.argmax(overlaps[row]))
            if overlaps[row, best] < iou_threshold:
                miss[row] = True
            elif truth.difficult[best]:
                continue
            elif taken[best]:
                miss[row] = True
            else:
                taken[best] = True
                hit[row] = True
        scores.append(image_scores)
        true_positive.append(hit)
        false_positive.append(miss)
        image_index.append(np.full(len(image_scores), index))
    if not scores:
        empty = np.zeros(0)
        return MatchedDetections(
            empty, empty.astype(bool), empty.astype(bool), empty.astype(np.int64), positives
        )
    return MatchedDetections(
        np.concatenate(scores),
        np.concatenate(true_positive),
        np.concatenate(false_positive),
        np.concatenate(image_index).astype(np.int64),
        positives,
    )


def precision_recall(
    matched: MatchedDetections, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if weights is None:
        weights = np.ones(len(matched.positives_per_image))
    order = np.argsort(-matched.scores, kind="stable")
    detection_weight = weights[matched.image_index[order]]
    true_positive = np.cumsum(detection_weight * matched.true_positive[order])
    false_positive = np.cumsum(detection_weight * matched.false_positive[order])
    positives = float((weights * matched.positives_per_image).sum())
    recall = true_positive / positives if positives > 0 else np.zeros_like(true_positive)
    counted = true_positive + false_positive
    precision = np.divide(
        true_positive, counted, out=np.zeros_like(true_positive), where=counted > 0
    )
    return precision, recall


def area_under_curve(precision: np.ndarray, recall: np.ndarray) -> float:
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.flatnonzero(recall[1:] != recall[:-1])
    return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))


def average_precision(matched: MatchedDetections, weights: np.ndarray | None = None) -> float:
    precision, recall = precision_recall(matched, weights)
    return area_under_curve(precision, recall)


def max_recall(matched: MatchedDetections) -> float:
    _, recall = precision_recall(matched)
    return float(recall[-1]) if recall.size else 0.0


def bootstrap_ap_interval(
    matched: MatchedDetections, samples: int = BOOTSTRAP_SAMPLES, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    count = len(matched.positives_per_image)
    values = []
    for _ in range(samples):
        weights = np.bincount(rng.integers(0, count, count), minlength=count).astype(np.float64)
        values.append(average_precision(matched, weights))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
