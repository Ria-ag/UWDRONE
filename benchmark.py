import sys
import time
from pathlib import Path

import easyocr
import tensorrt
import torch
from rapidocr import EngineType, OCRVersion, RapidOCR

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "sample.png"
RUNS = 10

# Directory containing your local model files
MODEL_DIR = Path("/home/uwdrones/")

DET_MODEL = MODEL_DIR / "det.onnx"
REC_MODEL = MODEL_DIR / "rec.onnx"
REC_DICT = MODEL_DIR / "ppocrv6_dict.txt"
TRT_CACHE = MODEL_DIR / "trt_cache"

print("TensorRT version:", tensorrt.__version__)
print("PyTorch CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "EasyOCR would run on CPU"

# Check files before initializing RapidOCR
for path in (DET_MODEL, REC_MODEL, REC_DICT):
    if not path.is_file():
        raise FileNotFoundError(f"Missing local model file: {path}")


def benchmark(name, load_fn, run_fn):
    t = time.perf_counter()
    model = load_fn()
    load_time = time.perf_counter() - t

    # Warm-up; TensorRT may build its engine on the first call.
    t = time.perf_counter()
    run_fn(model)
    warmup_time = time.perf_counter() - t

    times = []
    for _ in range(RUNS):
        t = time.perf_counter()
        run_fn(model)
        times.append(time.perf_counter() - t)

    print(f"\n{name}")
    print(f"  Model load: {load_time:.2f}s | First call: {warmup_time:.2f}s")
    print(
        f"  Over {RUNS} runs -> "
        f"avg: {sum(times) / RUNS:.3f}s | "
        f"min: {min(times):.3f}s | "
        f"max: {max(times):.3f}s"
    )


benchmark(
    "RapidOCR (TensorRT, local models)",
    lambda: RapidOCR(
        params={
            "Det.engine_type": EngineType.TENSORRT,
            "Det.model_path": str(DET_MODEL),
            "Det.ocr_version": OCRVersion.PPOCRV6,

            "Rec.engine_type": EngineType.TENSORRT,
            "Rec.model_path": str(REC_MODEL),
            "Rec.rec_keys_path": str(REC_DICT),
            "Rec.ocr_version": OCRVersion.PPOCRV6,

            "EngineConfig.tensorrt.use_fp16": True,
            "EngineConfig.tensorrt.cache_dir": str(TRT_CACHE),
        }
    ),
    lambda engine: engine(str(IMAGE), use_cls=False),
)

benchmark(
    "EasyOCR (CUDA)",
    lambda: easyocr.Reader(["en"], gpu=True),
    lambda reader: reader.readtext(str(IMAGE)),
)
