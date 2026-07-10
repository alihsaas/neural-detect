from collections.abc import Callable

import numpy as np

from neural_detect.classical import HogDetector, default_detector, train_hog_svm
from neural_detect.config import DETECTIONS_DIR, HOG_UPSCALES, MODELS_DIR, RESULTS_DIR, SPLIT_PATH
from neural_detect.datasets import load_annotations, load_image
from neural_detect.evaluation import (
    ap50,
    print_summary,
    run_detector,
    save_detections,
    summarize,
    write_csv,
    write_json,
)
from neural_detect.splits import load_split


def choose_upscale(
    make: Callable[[float], HogDetector], ids: list[str], annotations: dict, name: str
) -> tuple[float, list[dict]]:
    table = []
    for upscale in HOG_UPSCALES:
        detections, seconds = run_detector(make(upscale).detect, ids, load_image)
        score = ap50(detections, annotations)
        table.append(
            {"detector": name, "upscale": upscale, "validation_ap50": score, "seconds": seconds}
        )
        print(f"{name} upscale {upscale}: validation AP50 {score:.4f} ({seconds:.0f} s)")
    best = max(table, key=lambda row: row["validation_ap50"])
    return best["upscale"], table


def main() -> None:
    split = load_split(SPLIT_PATH)
    train_annotations = load_annotations(split["train"])
    validation_annotations = load_annotations(split["validation"])
    test_annotations = load_annotations(split["test"])
    output = RESULTS_DIR / "classical"

    print("Training HOG + linear SVM on the training images")
    coefficients, training = train_hog_svm(split["train"], train_annotations, load_image)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MODELS_DIR / "hog_trained.npy", coefficients)
    print(training)

    makers = {
        "hog_opencv_default": default_detector,
        "hog_trained": lambda upscale: HogDetector("hog_trained", coefficients, upscale),
    }
    selection, table, rows = {}, [], []
    for name, make in makers.items():
        upscale, rows_for_name = choose_upscale(
            make, split["validation"], validation_annotations, name
        )
        selection[name] = upscale
        table += rows_for_name
        detections, seconds = run_detector(make(upscale).detect, split["test"], load_image, name)
        save_detections(detections, DETECTIONS_DIR / f"{name}.npz")
        rows.append(summarize(name, detections, test_annotations))
        print(f"{name}: test run took {seconds:.0f} s")

    write_csv(table, output / "upscale_selection.csv")
    write_json({"upscale": selection, "training": training}, output / "selection.json")
    write_csv(rows, output / "test_summary.csv")
    print_summary(rows)


if __name__ == "__main__":
    main()
