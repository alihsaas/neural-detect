import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torchvision.models import mobilenet_v3_large

from neural_detect.boxes import Letterbox, letterbox, nms
from neural_detect.config import (
    BACKBONE_URL,
    INPUT_SIZE,
    MAX_DETECTIONS,
    NMS_IOU,
    STRIDE,
    WEIGHTS_DIR,
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FEATURE_TAPS = (6, 12, 16)
SIZE_LOSS_WEIGHT = 0.1
GAUSSIAN_ALPHA = 0.54


@dataclass
class RunSettings:
    name: str
    seed: int
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    width: int = 96
    input_size: int = INPUT_SIZE


def conv_block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
        nn.BatchNorm2d(outputs),
        nn.ReLU(inplace=True),
    )


class CentreNet(nn.Module):
    def __init__(self, width: int = 96, pretrained: bool = True) -> None:
        super().__init__()
        backbone = mobilenet_v3_large()
        if pretrained:
            backbone.load_state_dict(torch.load(backbone_path(), weights_only=True))
        self.features = backbone.features[: FEATURE_TAPS[-1] + 1]
        channels = [self.features[index].out_channels for index in FEATURE_TAPS[:-1]]
        channels.append(self.features[FEATURE_TAPS[-1]].out_channels)
        self.lateral = nn.ModuleList(nn.Conv2d(count, width, 1) for count in channels)
        self.smooth = nn.ModuleList(conv_block(width, width) for _ in channels[:-1])
        self.up = conv_block(width, width)
        self.heatmap = self.head(width, 1)
        self.size = self.head(width, 2)
        self.offset = self.head(width, 2)
        nn.init.constant_(self.heatmap[-1].bias, -math.log((1 - 0.01) / 0.01))

    @staticmethod
    def head(width: int, outputs: int) -> nn.Sequential:
        return nn.Sequential(conv_block(width, width), nn.Conv2d(width, outputs, 1))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        taps = []
        values = inputs
        for index, layer in enumerate(self.features):
            values = layer(values)
            if index in FEATURE_TAPS:
                taps.append(values)
        merged = self.lateral[-1](taps[-1])
        for level in range(len(taps) - 2, -1, -1):
            lateral = self.lateral[level](taps[level])
            merged = lateral + functional.interpolate(
                merged, size=lateral.shape[-2:], mode="nearest"
            )
            merged = self.smooth[level](merged)
        merged = self.up(functional.interpolate(merged, scale_factor=2.0, mode="nearest"))
        return self.heatmap(merged), self.size(merged), self.offset(merged)


def backbone_path() -> Path:
    path = WEIGHTS_DIR / BACKBONE_URL.rsplit("/", 1)[1]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run scripts/download_data.py first.")
    return path


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalise(images: np.ndarray) -> np.ndarray:
    values = images.astype(np.float32) / 255.0
    return ((values - MEAN) / STD).transpose(0, 3, 1, 2).copy()


def prepare(image: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, Letterbox]:
    canvas, geometry = letterbox(image, size)
    return normalise(canvas[None]), geometry


def gaussian_targets(
    boxes: np.ndarray, size: int = INPUT_SIZE, stride: int = STRIDE
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cells = size // stride
    heatmap = np.zeros((cells, cells), dtype=np.float32)
    box_size = np.zeros((2, cells, cells), dtype=np.float32)
    offset = np.zeros((2, cells, cells), dtype=np.float32)
    mask = np.zeros((cells, cells), dtype=np.float32)
    rows, columns = np.mgrid[0:cells, 0:cells]
    order = np.argsort(-((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
    for x1, y1, x2, y2 in boxes[order] / stride:
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
        column, row = min(int(centre_x), cells - 1), min(int(centre_y), cells - 1)
        if column < 0 or row < 0:
            continue
        sigma_x = max(GAUSSIAN_ALPHA * width / 6, 0.5)
        sigma_y = max(GAUSSIAN_ALPHA * height / 6, 0.5)
        blob = np.exp(
            -((columns - column) ** 2) / (2 * sigma_x**2) - (rows - row) ** 2 / (2 * sigma_y**2)
        )
        heatmap = np.maximum(heatmap, blob.astype(np.float32))
        box_size[:, row, column] = width, height
        offset[:, row, column] = centre_x - column, centre_y - row
        mask[row, column] = 1.0
    return heatmap, box_size, offset, mask


def focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    positive = target.eq(1).float()
    negative = 1.0 - positive
    positive_loss = torch.log(probability) * (1 - probability) ** 2 * positive
    negative_loss = torch.log(1 - probability) * probability**2 * (1 - target) ** 4 * negative
    count = positive.sum().clamp(min=1.0)
    return -(positive_loss.sum() + negative_loss.sum()) / count


def detection_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    heatmap_logits, size, offset = outputs
    heatmap, box_size, box_offset, mask = targets
    count = mask.sum().clamp(min=1.0)
    weight = mask.unsqueeze(1)
    size_loss = (torch.abs(size.float() - box_size) * weight).sum() / count
    offset_loss = (torch.abs(offset.float() - box_offset) * weight).sum() / count
    return focal_loss(heatmap_logits[:, 0], heatmap) + SIZE_LOSS_WEIGHT * size_loss + offset_loss


def decode(
    heatmap_logits: np.ndarray,
    size: np.ndarray,
    offset: np.ndarray,
    stride: int = STRIDE,
    limit: int = MAX_DETECTIONS,
) -> tuple[np.ndarray, np.ndarray]:
    probability = 1.0 / (1.0 + np.exp(-heatmap_logits.astype(np.float32)))
    peaks = cv2.dilate(probability, np.ones((3, 3), np.uint8))
    candidates = np.flatnonzero((probability >= peaks).ravel())
    scores = probability.ravel()[candidates]
    top = candidates[np.argsort(-scores, kind="stable")[:limit]]
    rows, columns = np.unravel_index(top, probability.shape)
    centre_x = (columns + offset[0, rows, columns]) * stride
    centre_y = (rows + offset[1, rows, columns]) * stride
    width = np.clip(size[0, rows, columns], 0, None) * stride
    height = np.clip(size[1, rows, columns], 0, None) * stride
    boxes = np.stack(
        [centre_x - width / 2, centre_y - height / 2, centre_x + width / 2, centre_y + height / 2],
        axis=1,
    ).astype(np.float64)
    scores = probability[rows, columns].astype(np.float64)
    keep = nms(boxes, scores, NMS_IOU, limit)
    return boxes[keep], scores[keep]


def random_affine(
    image: np.ndarray, boxes: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    size = image.shape[0]
    scale = rng.uniform(0.75, 1.25)
    shift_x, shift_y = rng.uniform(-0.1, 0.1, 2) * size
    matrix = np.array(
        [
            [scale, 0.0, (1 - scale) * size / 2 + shift_x],
            [0.0, scale, (1 - scale) * size / 2 + shift_y],
        ]
    )
    warped = cv2.warpAffine(
        image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114)
    )
    moved = boxes.copy()
    moved[:, [0, 2]] = moved[:, [0, 2]] * scale + matrix[0, 2]
    moved[:, [1, 3]] = moved[:, [1, 3]] * scale + matrix[1, 2]
    clipped = np.clip(moved, 0, size)
    kept_area = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
    full_area = (moved[:, 2] - moved[:, 0]) * (moved[:, 3] - moved[:, 1])
    keep = kept_area > 0.3 * np.maximum(full_area, 1e-6)
    return warped, clipped[keep]


def augment(
    image: np.ndarray, boxes: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = image[:, ::-1].copy()
        boxes = boxes.copy()
        boxes[:, [0, 2]] = image.shape[1] - boxes[:, [2, 0]]
    image, boxes = random_affine(image, boxes, rng)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.7, 1.3)
    hsv[..., 2] *= rng.uniform(0.7, 1.3)
    image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return image, boxes


def make_batch(
    images: np.ndarray,
    boxes: list[np.ndarray],
    rng: np.random.Generator | None,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    pictures, targets = [], []
    for image, image_boxes in zip(images, boxes, strict=True):
        if rng is not None:
            image, image_boxes = augment(image, image_boxes, rng)
        pictures.append(image)
        targets.append(gaussian_targets(image_boxes, image.shape[0]))
    inputs = torch.from_numpy(normalise(np.stack(pictures))).to(device)
    stacked = tuple(
        torch.from_numpy(np.stack([target[index] for target in targets])).to(device)
        for index in range(4)
    )
    return inputs, stacked


@torch.no_grad()
def predict_batch(
    model: nn.Module, images: np.ndarray, device: torch.device
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    model.eval()
    inputs = torch.from_numpy(normalise(images)).to(device)
    heatmap, size, offset = (output.float().cpu().numpy() for output in model(inputs))
    return [(heatmap[index, 0], size[index], offset[index]) for index in range(len(images))]


def detect_cached(
    model: nn.Module,
    images: np.ndarray,
    geometries: list[Letterbox],
    device: torch.device,
    batch_size: int = 32,
) -> list[tuple[np.ndarray, np.ndarray]]:
    results = []
    for start in range(0, len(images), batch_size):
        outputs = predict_batch(model, images[start : start + batch_size], device)
        for offset_index, raw in enumerate(outputs):
            boxes, scores = decode(*raw)
            geometry = geometries[start + offset_index]
            results.append((geometry.to_original(boxes), scores))
    return results


def train_model(
    settings: RunSettings,
    train_images: np.ndarray,
    train_boxes: list[np.ndarray],
    evaluate: Callable[[nn.Module], float],
    device: torch.device,
    evaluate_every: int = 5,
) -> tuple[nn.Module, list[dict], int]:
    seed_everything(settings.seed)
    rng = np.random.default_rng(settings.seed)
    model = CentreNet(settings.width).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    steps_per_epoch = math.ceil(len(train_images) / settings.batch_size)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=settings.learning_rate,
        total_steps=settings.epochs * steps_per_epoch,
        pct_start=0.05,
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    history: list[dict] = []
    best_score, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, settings.epochs + 1):
        model.train()
        order = rng.permutation(len(train_images))
        total = 0.0
        for start in range(0, len(order), settings.batch_size):
            chosen = np.sort(order[start : start + settings.batch_size])
            inputs, targets = make_batch(
                train_images[chosen], [train_boxes[index] for index in chosen], rng, device
            )
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(inputs)
            loss = detection_loss(outputs, targets)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            total += loss.item() * len(chosen)
        entry = {"epoch": epoch, "train_loss": total / len(order), "validation_ap50": ""}
        if epoch % evaluate_every == 0 or epoch == settings.epochs:
            score = evaluate(model)
            entry["validation_ap50"] = score
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
        history.append(entry)
        print(
            f"{settings.name} epoch {epoch:3d}  loss {entry['train_loss']:.4f}"
            f"  validation AP50 {entry['validation_ap50']}"
        )
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def save_checkpoint(
    path: Path, model: nn.Module, settings: RunSettings, history: list[dict], best_epoch: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "settings": asdict(settings),
            "history": history,
            "best_epoch": best_epoch,
            "state": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = CentreNet(payload["settings"]["width"], pretrained=False).to(device)
    model.load_state_dict(payload["state"])
    model.eval()
    return model, payload
