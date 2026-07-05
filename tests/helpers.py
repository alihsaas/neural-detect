import cv2
import numpy as np

from neural_detect.datasets import Annotation


def figure_image(
    rng: np.random.Generator, boxes: list[tuple[int, int, int, int]], shape=(240, 320)
) -> np.ndarray:
    image = rng.integers(150, 210, (*shape, 3), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        centre = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.ellipse(image, centre, ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, (30, 30, 40), -1)
        head = (centre[0], y1 + (y2 - y1) // 8)
        cv2.circle(image, head, max(2, (x2 - x1) // 3), (60, 40, 30), -1)
    return image


def annotation_for(image_id: str, image: np.ndarray, boxes: list) -> Annotation:
    count = len(boxes)
    return Annotation(
        image_id,
        image.shape[1],
        image.shape[0],
        np.array(boxes, dtype=np.float64).reshape(-1, 4),
        np.zeros(count, dtype=bool),
        np.zeros(count, dtype=bool),
    )
