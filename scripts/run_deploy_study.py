import json
import os
import platform
import shutil

import cv2
import numpy as np
import onnxruntime
import torch
from onnxruntime.quantization import CalibrationMethod

from neural_detect.cache import load_cache
from neural_detect.classical import HogDetector, default_detector
from neural_detect.config import (
    CHECKPOINT_DIR,
    DETECTIONS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    SPLIT_PATH,
)
from neural_detect.datasets import load_annotations, load_image
from neural_detect.deploy import OnnxDetector, benchmark, export_onnx, file_megabytes, quantize_int8
from neural_detect.evaluation import (
    ap50,
    print_summary,
    run_detector,
    save_detections,
    summarize,
    write_csv,
    write_json,
)
from neural_detect.neural import load_checkpoint, normalise
from neural_detect.splits import load_split

CALIBRATION_IMAGES = 200
SPEED_IMAGES = 200
CALIBRATION_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "percentile": CalibrationMethod.Percentile,
}


def main() -> None:
    split = load_split(SPLIT_PATH)
    validation_annotations = load_annotations(split["validation"])
    test_annotations = load_annotations(split["test"])
    runs = json.loads((RESULTS_DIR / "learned" / "runs.json").read_text(encoding="ascii"))["runs"]
    chosen = max(runs, key=lambda run: run["best_validation_ap50"])["name"]
    print(f"Exporting {chosen}, the seed with the best validation AP50")
    model, _ = load_checkpoint(CHECKPOINT_DIR / f"{chosen}.pt", torch.device("cpu"))
    fp32_path = MODELS_DIR / "centrenet_fp32.onnx"
    export_onnx(model, fp32_path)

    rng = np.random.default_rng(0)
    _, train_images, _, _ = load_cache("train")
    picks = np.sort(rng.choice(len(train_images), CALIBRATION_IMAGES, replace=False))
    calibration = [normalise(train_images[index : index + 1]) for index in picks]

    calibration_table = []
    for name, method in CALIBRATION_METHODS.items():
        path = MODELS_DIR / f"centrenet_int8_{name}.onnx"
        quantize_int8(fp32_path, path, calibration, method)
        found, _ = run_detector(OnnxDetector(path).detect, split["validation"], load_image)
        score = ap50(found, validation_annotations)
        calibration_table.append({"calibration": name, "validation_ap50": score})
        print(f"INT8 {name}: validation AP50 {score:.4f}")
    best_calibration = max(calibration_table, key=lambda row: row["validation_ap50"])["calibration"]
    int8_path = MODELS_DIR / "centrenet_int8.onnx"
    shutil.copyfile(MODELS_DIR / f"centrenet_int8_{best_calibration}.onnx", int8_path)

    rows = []
    for name, path in (("onnx_fp32", fp32_path), ("onnx_int8", int8_path)):
        detections, seconds = run_detector(
            OnnxDetector(path).detect, split["test"], load_image, name
        )
        save_detections(detections, DETECTIONS_DIR / f"{name}.npz")
        rows.append(summarize(name, detections, test_annotations))
        print(f"{name}: test run took {seconds:.0f} s")

    selection = json.loads(
        (RESULTS_DIR / "classical" / "selection.json").read_text(encoding="ascii")
    )["upscale"]
    hog_trained = np.load(MODELS_DIR / "hog_trained.npy")
    speed_ids = sorted(rng.choice(split["test"], SPEED_IMAGES, replace=False).tolist())
    speed_images = [load_image(image_id) for image_id in speed_ids]
    speed_rows = []
    for threads in (1, os.cpu_count() or 1):
        cv2.setNumThreads(threads)
        methods = {
            "hog_opencv_default": default_detector(selection["hog_opencv_default"]).detect,
            "hog_trained": HogDetector("hog_trained", hog_trained, selection["hog_trained"]).detect,
            "onnx_fp32": OnnxDetector(fp32_path, threads).detect,
            "onnx_int8": OnnxDetector(int8_path, threads).detect,
        }
        for name, detect in methods.items():
            timing = benchmark(detect, speed_images)
            speed_rows.append({"method": name, "threads": threads, **timing})
            print(f"{name} threads {threads}: median {timing['ms_median']:.1f} ms")
    cv2.setNumThreads(-1)

    write_csv(calibration_table, RESULTS_DIR / "deploy" / "calibration_selection.csv")
    write_csv(rows, RESULTS_DIR / "deploy" / "test_summary.csv")
    write_csv(speed_rows, RESULTS_DIR / "deploy" / "speed.csv")
    write_json(
        {
            "exported_run": chosen,
            "int8_calibration": best_calibration,
            "calibration_images": CALIBRATION_IMAGES,
            "speed_images": SPEED_IMAGES,
            "fp32_megabytes": file_megabytes(fp32_path),
            "int8_megabytes": file_megabytes(int8_path),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "onnxruntime": onnxruntime.__version__,
            "opencv": cv2.__version__,
        },
        RESULTS_DIR / "deploy" / "deploy.json",
    )
    print_summary(rows)


if __name__ == "__main__":
    main()
