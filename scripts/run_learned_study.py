import argparse
from dataclasses import asdict

import torch

from neural_detect.cache import build_cache, cache_exists, load_cache
from neural_detect.config import CHECKPOINT_DIR, DETECTIONS_DIR, RESULTS_DIR, SEEDS, SPLIT_PATH
from neural_detect.datasets import load_annotations, load_image
from neural_detect.evaluation import (
    ap50,
    print_summary,
    save_detections,
    summarize,
    write_csv,
    write_json,
)
from neural_detect.neural import (
    RunSettings,
    detect_cached,
    load_checkpoint,
    parameter_count,
    save_checkpoint,
    train_model,
)
from neural_detect.splits import load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=RunSettings.epochs)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--reuse", action="store_true")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    split = load_split(SPLIT_PATH)
    annotations = {name: load_annotations(ids) for name, ids in split.items()}
    for name, ids in split.items():
        if not cache_exists(name, ids):
            print(f"Caching {name} images")
            build_cache(name, ids, annotations[name], load_image)
    _, train_images, _, train_boxes = load_cache("train")
    validation_ids, validation_images, validation_geometry, _ = load_cache("validation")
    test_ids, test_images, test_geometry, _ = load_cache("test")

    def evaluate(model: torch.nn.Module) -> float:
        found = detect_cached(model, validation_images, validation_geometry, device)
        return ap50(dict(zip(validation_ids, found, strict=True)), annotations["validation"])

    output = RESULTS_DIR / "learned"
    rows, runs = [], []
    for seed in arguments.seeds:
        settings = RunSettings(name=f"centrenet_seed{seed}", seed=seed, epochs=arguments.epochs)
        path = CHECKPOINT_DIR / f"{settings.name}.pt"
        if arguments.reuse and path.exists():
            model, payload = load_checkpoint(path, device)
            if payload["settings"] != asdict(settings):
                raise SystemExit(f"{path} was trained with different settings")
            history, best_epoch = payload["history"], payload["best_epoch"]
            print(f"{settings.name}: reused {path.name}")
        else:
            model, history, best_epoch = train_model(
                settings, train_images, train_boxes, evaluate, device
            )
            save_checkpoint(path, model, settings, history, best_epoch)
        write_csv(history, output / f"{settings.name}_history.csv")
        found = detect_cached(model, test_images, test_geometry, device)
        detections = dict(zip(test_ids, found, strict=True))
        save_detections(detections, DETECTIONS_DIR / f"{settings.name}.npz")
        rows.append(summarize(settings.name, detections, annotations["test"]))
        scored = [entry["validation_ap50"] for entry in history if entry["validation_ap50"] != ""]
        runs.append(
            {
                "name": settings.name,
                "best_epoch": best_epoch,
                "best_validation_ap50": max(scored),
                "parameters": parameter_count(model),
            }
        )
        del model
        torch.cuda.empty_cache()

    write_csv(rows, output / "test_summary.csv")
    write_json(
        {"device": str(device), "torch": torch.__version__, "runs": runs}, output / "runs.json"
    )
    print_summary(rows)


if __name__ == "__main__":
    main()
