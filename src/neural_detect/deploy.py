import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from neural_detect.config import INPUT_SIZE
from neural_detect.neural import decode, prepare

OUTPUT_NAMES = ["heatmap", "size", "offset"]


def export_onnx(model: torch.nn.Module, path: Path, size: int = INPUT_SIZE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example = torch.zeros(1, 3, size, size)
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=["image"],
        output_names=OUTPUT_NAMES,
        opset_version=18,
        dynamo=True,
        external_data=False,
    )


class ArrayReader(CalibrationDataReader):
    def __init__(self, batches: list[np.ndarray]) -> None:
        self.data = batches
        self.set_range(0, len(batches))

    def __len__(self) -> int:
        return len(self.data)

    def set_range(self, start_index: int, end_index: int) -> None:
        self.batches = iter(self.data[start_index:end_index])

    def get_next(self) -> dict | None:
        batch = next(self.batches, None)
        return None if batch is None else {"image": batch}


def quantize_int8(
    source: Path,
    target: Path,
    calibration: list[np.ndarray],
    method: CalibrationMethod,
    chunk: int = 20,
) -> None:
    prepared = target.with_name(target.stem + "_prepared.onnx")
    quant_pre_process(str(source), str(prepared))
    quantize_static(
        str(prepared),
        str(target),
        ArrayReader(calibration),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=method,
        extra_options={"CalibStridedMinMax": chunk},
    )
    prepared.unlink()


class OnnxDetector:
    def __init__(self, path: Path, threads: int = 0) -> None:
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.size = self.session.get_inputs()[0].shape[-1]

    def raw(self, inputs: np.ndarray) -> list[np.ndarray]:
        return self.session.run(OUTPUT_NAMES, {"image": inputs})

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inputs, geometry = prepare(image, self.size)
        heatmap, size, offset = self.raw(inputs)
        boxes, scores = decode(heatmap[0, 0], size[0], offset[0])
        return geometry.to_original(boxes), scores


def benchmark(
    run: Callable[[np.ndarray], object], images: list[np.ndarray], warmup: int = 10
) -> dict[str, float]:
    for image in images[:warmup]:
        run(image)
    times = []
    for image in images:
        start = time.perf_counter()
        run(image)
        times.append(1000.0 * (time.perf_counter() - start))
    return {
        "ms_median": float(np.median(times)),
        "ms_p90": float(np.percentile(times, 90)),
    }


def file_megabytes(path: Path) -> float:
    return path.stat().st_size / 1e6
