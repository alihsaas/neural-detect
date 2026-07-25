# NeuralDetect

> I measured the reference results using the original benchmark environment.

**Person detection on a CPU: how far does a classical HOG detector get, what does
a small learned detector add, and what does INT8 quantisation cost in accuracy
and buy in speed?**

I compare classical and learned detectors on the same test images, then deploy
the learned one the way it would run on a small device: exported to ONNX,
quantised to 8-bit integers (INT8), and timed on the CPU.

```text
image -> classical: HOG + linear SVM sliding window (OpenCV's pedestrian model,
                    and my own trained on VOC), non-maximum suppression
      -> learned:   letterbox to 320 x 320 -> CentreNet (MobileNetV3 backbone,
                    predicts object centres and sizes), 3 seeds, GPU training
      -> deploy:    best seed -> ONNX FP32 -> ONNX INT8 (calibrated on train)
      -> AP50 on the VOC 2007 test set, and CPU milliseconds per image
```

## Setup

Python 3.12, [uv](https://docs.astral.sh/uv/), PyTorch for CUDA 12.8.

```powershell
uv sync --extra neural
uv run pytest -q --basetemp=.pytest-tmp
```

`run_study.bat` runs the tests, the download, the split and the three studies.
`scripts/live_demo.py --source 0` runs the INT8 model on a webcam, and
`--source <url>` on an MJPEG stream such as an ESP32-CAM.

## Data and split

Pascal VOC 2007 (Everingham et al., 2010), person class only. I train on the
official trainval set minus 500 images held out for validation, and test on the
official test set (4952 images). Boxes marked "difficult" count neither as hits
nor misses, as in the VOC rules.

## Metrics

- **AP50**: average precision, the area under the precision-recall curve, where
  a detection counts as correct if it overlaps a true box by at least 50%
  (intersection over union). 95% bootstrap interval over test images.
- **AP75**: the same with a 75% overlap, which rewards tight boxes.
- **Speed**: median milliseconds per image on the CPU, at 1 thread and all
  threads, including resizing and decoding.

## Results

VOC 2007 test set, 4952 images, 4528 person boxes. Speed is the median over 200
test images on an Intel laptop CPU (12 logical cores), at 1 and 12 threads.

| Method | AP50 (95% CI) | AP75 | Recall | ms, 1 thread | ms, 12 threads | Size |
|---|---|---|---|---|---|---|
| HOG, OpenCV model | 0.060 (0.050-0.071) | 0.004 | 0.351 | 334.9 | 61.9 | |
| HOG, my VOC model | 0.066 (0.057-0.077) | 0.004 | 0.498 | 362.6 | 90.4 | |
| CentreNet seed 0 | 0.692 (0.677-0.709) | 0.276 | 0.784 | | | |
| CentreNet seed 1 | 0.681 (0.665-0.698) | 0.264 | 0.782 | | | |
| CentreNet seed 2 | 0.677 (0.660-0.693) | 0.258 | 0.784 | | | |
| ONNX FP32 (seed 0) | 0.692 (0.677-0.708) | 0.276 | 0.784 | 78.2 | 34.6 | 14.6 MB |
| ONNX INT8 (seed 0) | 0.669 (0.654-0.686) | 0.247 | 0.773 | 43.0 | 31.7 | 4.0 MB |

- CentreNet is about ten times better than HOG on AP50, and 4 to 8 times faster
  on one thread. The three seeds average 0.683.
- Training my own HOG model on VOC added recall (0.351 to 0.498) but little AP50.
  My guess is that one upright 64 x 128 window cannot fit the many small,
  sitting or partly hidden people in VOC.
- INT8 cost 0.023 AP50 and made the model 3.6 times smaller. It was 1.8 times
  faster on one thread, but only 9% faster on 12 threads.
- Percentile calibration beat MinMax on validation (AP50 0.6748 against 0.6435), so
  it is the INT8 model I kept.
- All CentreNet seeds peaked on validation at epoch 30 or 35 of 60, so a shorter
  schedule would likely do as well.

Raw numbers are in `results/`.

## References

- M. Everingham et al. "The Pascal Visual Object Classes (VOC) Challenge." IJCV 2010.
- N. Dalal, B. Triggs. "Histograms of Oriented Gradients for Human Detection." CVPR 2005.
- X. Zhou, D. Wang, P. Kraehenbuehl. "Objects as Points." arXiv 2019.
- A. Howard et al. "Searching for MobileNetV3." ICCV 2019.
