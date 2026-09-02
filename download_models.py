#!/usr/bin/env python3
"""Download PP-OCRv6 tiny ONNX models into ./models (optional; RapidOCR can also auto-fetch)."""
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent / "models"
FILES = {
    "PP-OCRv6_det_tiny.onnx": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_tiny.onnx",
    "PP-OCRv6_rec_tiny.onnx": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_tiny.onnx",
}

def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = ROOT / name
        if dest.is_file() and dest.stat().st_size > 1000:
            print("exists", dest)
            continue
        print("download", name)
        urlretrieve(url, dest)
        print("wrote", dest, dest.stat().st_size)

if __name__ == "__main__":
    main()
