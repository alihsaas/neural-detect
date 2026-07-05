from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
VOC_DIR = DATA_DIR / "VOCdevkit" / "VOC2007"
WEIGHTS_DIR = DATA_DIR / "weights"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODELS_DIR = PROJECT_ROOT / "models"
DETECTIONS_DIR = PROJECT_ROOT / "detections"
SPLIT_PATH = RESULTS_DIR / "split.json"

VOC_BASE_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/"
VOC_ARCHIVES = {
    "VOCtrainval_06-Nov-2007.tar": "c52e279531787c972589f7e41ab4ae64",
    "VOCtest_06-Nov-2007.tar": "b6e924de25625d8de591ea690078ad9f",
}
BACKBONE_URL = "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth"

CLASS_NAME = "person"
SPLIT_SEED = 2026
VALIDATION_COUNT = 500

IOU_THRESHOLD = 0.5
STRICT_IOU_THRESHOLD = 0.75
MAX_DETECTIONS = 100
NMS_IOU = 0.45
BOOTSTRAP_SAMPLES = 1000

HOG_WINDOW = (64, 128)
HOG_BOX_MARGIN = (8, 16)
HOG_WIN_STRIDE = (8, 8)
HOG_SCALE_STEP = 1.05
HOG_HIT_THRESHOLD = -1.0
HOG_UPSCALES = (1.0, 1.5, 2.0)

INPUT_SIZE = 320
STRIDE = 4
SEEDS = (0, 1, 2)
