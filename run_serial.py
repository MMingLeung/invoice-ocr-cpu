#!/usr/bin/env python3
"""Serial, disk-backed invoice OCR: one image at a time, RapidOCR CPU ONNX, classifier off."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Keep CPU providers only; never request CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("ORT_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

from pipeline import (
    GT,
    extract_fields,
    field_accuracy,
    load_and_resize,
    make_rapidocr_engine,
    parse_rapidocr_result,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out-dir", default="ocr_out")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = make_rapidocr_engine(use_cls=False, intra_threads=args.threads)
    ledger = []
    for img_path in args.images:
        t0 = time.perf_counter()
        img = load_and_resize(img_path)
        result = engine(img, use_cls=False)
        txts, scores, elapse = parse_rapidocr_result(result)
        extracted = extract_fields(txts)
        rec = {
            "image": img_path,
            "wall_sec": time.perf_counter() - t0,
            "internal_elapse_sec": elapse,
            "extracted": extracted,
            "accuracy_vs_gt": field_accuracy(extracted, GT),
            "n_lines": len(txts),
            "texts": txts,
            "human_verify_amount": True,
        }
        dest = out_dir / (Path(img_path).stem + ".json")
        dest.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        ledger.append({"image": img_path, "json": str(dest), "extracted": extracted})
        print(dest, rec["wall_sec"], extracted, flush=True)
    (out_dir / "ledger_draft.json").write_text(
        json.dumps({"drafts": ledger, "note": "金额必须人工核定"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
