"""Classify a photo with DINOv2: on the Hexagon NPU by default, or on the laptop CPU with --cpu."""

import argparse
import functools
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODEL_OUTPUT = Path(os.environ.get("MODEL_OUTPUT", Path.home() / "model_output"))
QNN_RUNNER = os.environ.get(
    "QNN_RUNNER",
    str(Path.home() / "executorch/cmake-out/examples/qualcomm/executor_runner/qnn_executor_runner"),
)
DINO_PTE = os.environ.get("DINO_PTE", str(MODEL_OUTPUT / "dino_v2" / "dino_v2.pte"))
HF_MODEL = "facebook/dinov2-small-imagenet1k-1-layer"  # same model the upstream exporter uses
LABELS = Path(__file__).with_name("imagenet_labels.txt").read_text().splitlines()

# Must match the export: resize to 256x256, center-crop 224, ImageNet mean/std, NCHW float32.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image):
    """image: a file path or file-like object."""
    img = ImageOps.exif_transpose(Image.open(image)).convert("RGB")  # phone photos carry EXIF rotation
    img = img.resize((256, 256), Image.BILINEAR).crop((16, 16, 240, 240))
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[np.newaxis])  # (1, 3, 224, 224)


def run_npu(x):
    """Run one input through qnn_executor_runner; return (logits, model_ms)."""
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        x.tofile(work / "input_0.raw")
        (work / "input_list.txt").write_text("input_0.raw\n")
        (work / "outputs").mkdir()
        run = subprocess.run(
            [QNN_RUNNER, "--model_path", DINO_PTE,
             "--input_list_path", "input_list.txt", "--output_folder_path", "outputs"],
            cwd=work, capture_output=True, text=True,
        )
        if run.returncode != 0:
            raise RuntimeError(f"qnn_executor_runner failed:\n{run.stderr[-2000:]}")
        logits = np.fromfile(work / "outputs" / "output_0_0.raw", dtype=np.float32)
        return logits, float((work / "inference_speed.txt").read_text())


@functools.lru_cache(maxsize=1)
def load_cpu_model():
    from transformers import Dinov2ForImageClassification
    return Dinov2ForImageClassification.from_pretrained(HF_MODEL).eval()


def run_cpu(x):
    """Same model on the laptop CPU, for developing without the board."""
    import torch
    model = load_cpu_model()
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(pixel_values=torch.from_numpy(x)).logits[0].numpy()
    return logits, (time.perf_counter() - start) * 1000


def classify(image, top_k=5, backend="npu"):
    """Return the top-k (label, probability) pairs plus model and end-to-end latency in ms."""
    start = time.perf_counter()
    logits, model_ms = (run_cpu if backend == "cpu" else run_npu)(preprocess(image))
    if logits.size != len(LABELS):
        raise ValueError(f"Expected {len(LABELS)} logits, got {logits.size}; check the model's output")
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    top = probs.argsort()[::-1][:top_k]
    return {
        "top": [(LABELS[i], float(probs[i])) for i in top],
        "backend": backend,
        "model_ms": model_ms,
        "total_ms": (time.perf_counter() - start) * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--cpu", action="store_true", help="run on the laptop CPU instead of the NPU")
    args = parser.parse_args()
    result = classify(args.image, backend="cpu" if args.cpu else "npu")
    for label, prob in result["top"]:
        print(f"{prob:6.1%}  {label}")
    print(f"{result['backend'].upper()} {result['model_ms']:.1f} ms, end-to-end {result['total_ms']:.1f} ms")


if __name__ == "__main__":
    main()
