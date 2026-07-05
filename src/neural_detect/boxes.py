from dataclasses import dataclass

import cv2
import numpy as np


def area(boxes: np.ndarray) -> np.ndarray:
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def iou_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64).reshape(-1, 4)
    second = np.asarray(second, dtype=np.float64).reshape(-1, 4)
    left = np.maximum(first[:, None, 0], second[None, :, 0])
    top = np.maximum(first[:, None, 1], second[None, :, 1])
    right = np.minimum(first[:, None, 2], second[None, :, 2])
    bottom = np.minimum(first[:, None, 3], second[None, :, 3])
    overlap = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    union = area(first)[:, None] + area(second)[None, :] - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float, limit: int) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size and len(keep) < limit:
        best = order[0]
        keep.append(int(best))
        overlaps = iou_matrix(boxes[best : best + 1], boxes[order[1:]])[0]
        order = order[1:][overlaps <= threshold]
    return np.array(keep, dtype=np.int64)


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = boxes.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
    return clipped


@dataclass(frozen=True)
class Letterbox:
    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float
    width: int
    height: int

    def to_input(self, boxes: np.ndarray) -> np.ndarray:
        shifted = boxes.astype(np.float64).copy()
        shifted[:, [0, 2]] = shifted[:, [0, 2]] * self.scale_x + self.pad_x
        shifted[:, [1, 3]] = shifted[:, [1, 3]] * self.scale_y + self.pad_y
        return shifted

    def to_original(self, boxes: np.ndarray) -> np.ndarray:
        restored = boxes.astype(np.float64).copy()
        restored[:, [0, 2]] = (restored[:, [0, 2]] - self.pad_x) / self.scale_x
        restored[:, [1, 3]] = (restored[:, [1, 3]] - self.pad_y) / self.scale_y
        return clip_boxes(restored, self.width, self.height)


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    height, width = image.shape[:2]
    scale = size / max(height, width)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - new_width) // 2, (size - new_height) // 2
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas, Letterbox(
        new_width / width, new_height / height, float(pad_x), float(pad_y), width, height
    )
