import argparse
import time

import cv2

from neural_detect.config import MODELS_DIR
from neural_detect.deploy import OnnxDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default=str(MODELS_DIR / "centrenet_int8.onnx"))
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--threads", type=int, default=0)
    arguments = parser.parse_args()

    source = int(arguments.source) if arguments.source.isdigit() else arguments.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open {arguments.source}")
    detector = OnnxDetector(arguments.model, arguments.threads)
    smoothed = 0.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        started = time.perf_counter()
        boxes, scores = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elapsed = 1000.0 * (time.perf_counter() - started)
        smoothed = elapsed if smoothed == 0.0 else 0.9 * smoothed + 0.1 * elapsed
        for (x1, y1, x2, y2), score in zip(boxes, scores, strict=True):
            if score < arguments.threshold:
                continue
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 160, 0), 2)
            cv2.putText(
                frame,
                f"{score:.2f}",
                (int(x1), max(12, int(y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 160, 0),
                1,
            )
        cv2.putText(
            frame,
            f"{smoothed:.1f} ms per frame",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow("NeuralDetect", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
