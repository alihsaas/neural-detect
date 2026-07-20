from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import CalibrationMethod

from neural_detect.deploy import OnnxDetector, benchmark, export_onnx, quantize_int8
from neural_detect.neural import CentreNet, normalise


def test_onnx_matches_torch_and_int8_runs(tmp_path: Path, rng: np.random.Generator) -> None:
    torch.manual_seed(0)
    model = CentreNet(width=32, pretrained=False).eval()
    fp32 = tmp_path / "model.onnx"
    export_onnx(model, fp32, size=128)
    images = rng.integers(0, 256, (6, 128, 128, 3), dtype=np.uint8)
    inputs = normalise(images[:1])
    with torch.no_grad():
        expected = [output.numpy() for output in model(torch.from_numpy(inputs))]
    detector = OnnxDetector(fp32, threads=1)
    for found, wanted in zip(detector.raw(inputs), expected, strict=True):
        assert np.allclose(found, wanted, atol=1e-3)
    calibration = [normalise(images[index : index + 1]) for index in range(6)]
    int8 = tmp_path / "model_int8.onnx"
    quantize_int8(fp32, int8, calibration, CalibrationMethod.MinMax, chunk=3)
    quantized = OnnxDetector(int8, threads=1)
    assert [output.shape for output in quantized.raw(inputs)] == [item.shape for item in expected]
    boxes, scores = quantized.detect(images[0])
    assert boxes.shape == (len(scores), 4)
    timing = benchmark(quantized.detect, list(images), warmup=2)
    assert timing["ms_median"] > 0
