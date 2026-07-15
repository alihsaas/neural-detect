import numpy as np
import torch

from neural_detect.boxes import iou_matrix
from neural_detect.neural import (
    CentreNet,
    augment,
    decode,
    detection_loss,
    gaussian_targets,
    make_batch,
)


def test_targets_decode_back_to_boxes() -> None:
    boxes = np.array([[40.0, 30.0, 120.0, 250.0], [200.0, 100.0, 260.0, 220.0]])
    heatmap, size, offset, mask = gaussian_targets(boxes, 320, 4)
    assert mask.sum() == 2 and heatmap.max() == 1.0
    logits = np.log(np.clip(heatmap, 1e-6, 1 - 1e-6) / np.clip(1 - heatmap, 1e-6, 1))
    found, scores = decode(logits, size, offset, 4, 10)
    overlaps = iou_matrix(found[:2], boxes)
    assert np.all(overlaps.max(axis=1) > 0.99)


def test_augment_keeps_boxes_inside_image(rng: np.random.Generator) -> None:
    image = rng.integers(0, 256, (320, 320, 3), dtype=np.uint8)
    boxes = np.array([[10.0, 10.0, 100.0, 300.0], [250.0, 50.0, 319.0, 200.0]])
    for _ in range(20):
        moved_image, moved = augment(image, boxes, rng)
        assert moved_image.shape == image.shape
        assert np.all(moved >= 0) and np.all(moved <= 320)


def test_model_shapes_and_short_training_lowers_loss(rng: np.random.Generator) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = CentreNet(width=32, pretrained=False).to(device)
    images = rng.integers(0, 256, (4, 128, 128, 3), dtype=np.uint8)
    boxes = [np.array([[20.0, 20.0, 60.0, 110.0]])] * 4
    inputs, targets = make_batch(images, boxes, None, device)
    heatmap, size, offset = model(inputs)
    assert heatmap.shape == (4, 1, 32, 32) and size.shape == (4, 2, 32, 32)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(30):
        loss = detection_loss(model(inputs), targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    assert np.isfinite(losses).all() and losses[-1] < 0.5 * losses[0]
