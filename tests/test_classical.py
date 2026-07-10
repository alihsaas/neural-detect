import numpy as np
from helpers import annotation_for, figure_image

from neural_detect.boxes import iou_matrix
from neural_detect.classical import (
    HogDetector,
    box_to_window,
    crop_window,
    default_detector,
    train_hog_svm,
    window_to_box,
)


def test_window_and_box_are_inverse_for_upright_boxes() -> None:
    box = np.array([100.0, 50.0, 148.0, 146.0])
    assert np.allclose(window_to_box(np.array(box_to_window(box)))[0], box)


def test_crop_window_pads_outside_the_image(rng: np.random.Generator) -> None:
    image = rng.integers(0, 256, (100, 80, 3), dtype=np.uint8)
    assert crop_window(image, (-20, -30, 64, 128)).shape == (128, 64, 3)


def test_default_detector_returns_aligned_arrays(rng: np.random.Generator) -> None:
    image = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    boxes, scores = default_detector().detect(image)
    assert boxes.shape == (len(scores), 4)


def test_trained_hog_svm_finds_synthetic_figures() -> None:
    rng = np.random.default_rng(1)
    images, annotations = {}, {}
    for index in range(60):
        image_id = str(index)
        if index % 2:
            x, y = int(rng.integers(20, 220)), int(rng.integers(10, 60))
            boxes = [(x, y, x + 48, y + 160)]
        else:
            boxes = []
        images[image_id] = figure_image(rng, boxes)
        annotations[image_id] = annotation_for(image_id, images[image_id], boxes)
    coefficients, report = train_hog_svm(
        list(images), annotations, images.__getitem__, mining_images=20, hard_per_image=5
    )
    assert coefficients.shape == (3781,)
    assert report["positive_windows"] == 60
    detector = HogDetector("trained", coefficients, hit_threshold=0.0)
    image = figure_image(rng, [(120, 40, 168, 200)])
    boxes, _ = detector.detect(image)
    assert len(boxes) > 0
    assert iou_matrix(boxes[:1], np.array([[120, 40, 168, 200]]))[0, 0] > 0.5
