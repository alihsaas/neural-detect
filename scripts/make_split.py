from neural_detect.config import SPLIT_PATH
from neural_detect.datasets import image_ids, load_annotations
from neural_detect.splits import make_split, save_split


def main() -> None:
    split = make_split(image_ids("trainval"), image_ids("test"))
    person_boxes = {}
    for name, ids in split.items():
        annotations = load_annotations(ids)
        person_boxes[name] = {
            "all": int(sum(len(item.boxes) for item in annotations.values())),
            "not_difficult": int(sum((~item.difficult).sum() for item in annotations.values())),
            "images_with_people": int(sum(len(item.boxes) > 0 for item in annotations.values())),
        }
    save_split(split, person_boxes, SPLIT_PATH)
    for name, ids in split.items():
        print(f"{name}: {len(ids)} images, {person_boxes[name]}")


if __name__ == "__main__":
    main()
