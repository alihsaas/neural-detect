from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.svm import LinearSVC

from neural_detect.boxes import nms
from neural_detect.config import (
    HOG_BOX_MARGIN,
    HOG_HIT_THRESHOLD,
    HOG_SCALE_STEP,
    HOG_WIN_STRIDE,
    HOG_WINDOW,
    MAX_DETECTIONS,
    NMS_IOU,
)
from neural_detect.datasets import Annotation

DEFAULT_DETECTOR = "opencv_default"


def hog_descriptor() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(HOG_WINDOW, (16, 16), (8, 8), (8, 8), 9)


def window_to_box(windows: np.ndarray) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float64).reshape(-1, 4)
    scale = windows[:, 2] / HOG_WINDOW[0]
    margin_x, margin_y = HOG_BOX_MARGIN[0] * scale, HOG_BOX_MARGIN[1] * scale
    return np.stack(
        [
            windows[:, 0] + margin_x,
            windows[:, 1] + margin_y,
            windows[:, 0] + windows[:, 2] - margin_x,
            windows[:, 1] + windows[:, 3] - margin_y,
        ],
        axis=1,
    )


def box_to_window(box: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    inner_width = HOG_WINDOW[0] - 2 * HOG_BOX_MARGIN[0]
    inner_height = HOG_WINDOW[1] - 2 * HOG_BOX_MARGIN[1]
    scale = max((x2 - x1) / inner_width, (y2 - y1) / inner_height)
    width, height = HOG_WINDOW[0] * scale, HOG_WINDOW[1] * scale
    centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
    return centre_x - width / 2, centre_y - height / 2, width, height


def crop_window(image: np.ndarray, window: tuple[float, float, float, float]) -> np.ndarray:
    x, y, width, height = (int(round(value)) for value in window)
    pad = max(0, -x, -y, x + width - image.shape[1], y + height - image.shape[0])
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        x, y = x + pad, y + pad
    crop = image[y : y + height, x : x + width]
    return cv2.resize(crop, HOG_WINDOW, interpolation=cv2.INTER_AREA)


@dataclass
class HogDetector:
    name: str
    coefficients: np.ndarray
    upscale: float = 1.0
    hit_threshold: float = HOG_HIT_THRESHOLD

    def __post_init__(self) -> None:
        self.descriptor = hog_descriptor()
        self.descriptor.setSVMDetector(np.asarray(self.coefficients, dtype=np.float32))

    def windows(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.upscale != 1.0:
            image = cv2.resize(
                image, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_LINEAR
            )
        found, weights = self.descriptor.detectMultiScale(
            image, self.hit_threshold, HOG_WIN_STRIDE, (16, 16), HOG_SCALE_STEP, 0
        )
        windows = np.asarray(found, dtype=np.float64).reshape(-1, 4) / self.upscale
        return windows, np.asarray(weights, dtype=np.float64).reshape(-1)

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        windows, scores = self.windows(image)
        boxes = window_to_box(windows)
        keep = nms(boxes, scores, NMS_IOU, MAX_DETECTIONS)
        return boxes[keep], scores[keep]


def default_detector(upscale: float = 1.0) -> HogDetector:
    coefficients = np.asarray(cv2.HOGDescriptor_getDefaultPeopleDetector()).reshape(-1)
    return HogDetector(DEFAULT_DETECTOR, coefficients, upscale)


def usable_positive(box: np.ndarray, difficult: bool, truncated: bool) -> bool:
    width, height = box[2] - box[0], box[3] - box[1]
    return not difficult and not truncated and height >= 48 and 0.25 <= width / height <= 0.8


def positive_crops(image: np.ndarray, annotation: Annotation) -> list[np.ndarray]:
    crops = []
    for box, difficult, truncated in zip(
        annotation.boxes, annotation.difficult, annotation.truncated, strict=True
    ):
        if usable_positive(box, difficult, truncated):
            crop = crop_window(image, box_to_window(box))
            crops += [crop, crop[:, ::-1].copy()]
    return crops


def random_negative_crops(
    image: np.ndarray, count: int, rng: np.random.Generator
) -> list[np.ndarray]:
    height, width = image.shape[:2]
    largest = min(height, 2 * width)
    if largest < HOG_WINDOW[1]:
        return []
    crops = []
    for _ in range(count):
        window_height = rng.uniform(HOG_WINDOW[1], largest)
        window_width = window_height / 2
        x = rng.uniform(0, width - window_width)
        y = rng.uniform(0, height - window_height)
        crops.append(crop_window(image, (x, y, window_width, window_height)))
    return crops


def features(crops: list[np.ndarray]) -> np.ndarray:
    descriptor = hog_descriptor()
    return np.stack([descriptor.compute(crop).reshape(-1) for crop in crops]).astype(np.float32)


def fit_svm(positives: np.ndarray, negatives: np.ndarray, c: float) -> np.ndarray:
    inputs = np.concatenate([positives, negatives])
    labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    model = LinearSVC(C=c, max_iter=5000, random_state=0)
    model.fit(inputs, labels)
    return np.concatenate([model.coef_.reshape(-1), model.intercept_]).astype(np.float32)


def hard_negative_crops(
    detector: HogDetector, image: np.ndarray, limit: int, rng: np.random.Generator
) -> list[np.ndarray]:
    windows, scores = detector.windows(image)
    windows = windows[scores > 0]
    if len(windows) > limit:
        windows = windows[rng.choice(len(windows), limit, replace=False)]
    return [crop_window(image, tuple(window)) for window in windows]


def train_hog_svm(
    ids: list[str],
    annotations: dict[str, Annotation],
    load: Callable[[str], np.ndarray],
    negatives_per_image: int = 4,
    mining_images: int = 1000,
    hard_per_image: int = 10,
    c: float = 0.01,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    positive, negative = [], []
    empty_ids = []
    for image_id in ids:
        annotation = annotations[image_id]
        if len(annotation.boxes) == 0:
            empty_ids.append(image_id)
            negative += random_negative_crops(load(image_id), negatives_per_image, rng)
        else:
            positive += positive_crops(load(image_id), annotation)
    positive_features, negative_features = features(positive), features(negative)
    coefficients = fit_svm(positive_features, negative_features, c)
    detector = HogDetector("mining", coefficients, hit_threshold=0.0)
    hard = []
    for image_id in rng.permutation(empty_ids)[:mining_images]:
        hard += hard_negative_crops(detector, load(str(image_id)), hard_per_image, rng)
    if hard:
        negative_features = np.concatenate([negative_features, features(hard)])
        coefficients = fit_svm(positive_features, negative_features, c)
    report = {
        "positive_windows": len(positive),
        "random_negative_windows": len(negative),
        "hard_negative_windows": len(hard),
        "person_free_images": len(empty_ids),
        "c": c,
    }
    return coefficients, report
