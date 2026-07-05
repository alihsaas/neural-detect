import numpy as np

from neural_detect.boxes import iou_matrix, letterbox, nms


def test_iou_values() -> None:
    first = np.array([[0, 0, 10, 10]])
    second = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert np.allclose(iou_matrix(first, second), [[1.0, 1 / 3, 0.0]])


def test_nms_keeps_best_and_drops_overlap() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [30, 30, 40, 40]], dtype=float)
    scores = np.array([0.6, 0.9, 0.5])
    assert nms(boxes, scores, 0.5, 10).tolist() == [1, 2]
    assert nms(boxes, scores, 0.5, 1).tolist() == [1]


def test_letterbox_round_trip(rng: np.random.Generator) -> None:
    image = rng.integers(0, 256, (375, 500, 3), dtype=np.uint8)
    canvas, geometry = letterbox(image, 320)
    assert canvas.shape == (320, 320, 3)
    boxes = np.array([[10.0, 20.0, 200.0, 370.0], [0.0, 0.0, 500.0, 375.0]])
    assert np.allclose(geometry.to_original(geometry.to_input(boxes)), boxes)
    assert np.allclose(geometry.to_input(boxes)[1], [0, geometry.pad_y, 320, 320 - geometry.pad_y])
