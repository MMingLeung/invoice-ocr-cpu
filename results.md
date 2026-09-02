# Invoice OCR bench: RapidOCR + ONNX Runtime + PP-OCRv4 Chinese mobile

Measured on 2026-09-01 (SGT). This file is the factual report. Do **not** treat OCR amounts as final: 人核定稿.

## Inventory (what was already here)

Reused, not rewritten from scratch:

- Images: `images/invoice_clean.png`, `images/invoice_jpeg_blur.jpg`, `images/invoice_rotated.png`
- Ground truth: `images/ground_truth.json`
- Models already on disk (no download during this run):
  - `models/ch_PP-OCRv4_det_mobile.onnx` (4.6 MiB)
  - `models/ch_PP-OCRv4_rec_mobile.onnx` (11 MiB)
  - `models/ch_ppocr_mobile_v2.0_cls_mobile.onnx` (572 KiB, **not loaded**; classifier off)
- Existing venv: Python 3.13.5, `rapidocr==3.9.2`, `onnxruntime==1.29.0` (CPU wheel, `CPUExecutionProvider`; `onnxruntime-gpu` **not** installed), OpenCV 4.10
- Existing scripts: `bench.py`, `pipeline.py`, `run_limited.sh`, `generate_invoices.py`, probes
- Prior `out_paddleocr.json`: PaddleOCR 3.7 + PaddlePaddle 3.3.1 **crashed** on OneDNN/PIR (`ConvertPirAttribute2RuntimeAttribute`). Full PaddleOCR was **not** re-run on the constrained machine (per spec).

Added for this run:

- Continuous RSS sampler in `bench.py` (`/proc/self/status` VmHWM + 20 ms thread)
- `rss_wrap.py` — parent `wait4` ru_maxrss (GNU `time -v` analog; `/usr/bin/time` is not on this box)
- `run_serial.py` — serial, disk-backed production-style runner (cls off, long side 1280)
- Explicit `Det.model_path` / `Rec.model_path` in `pipeline.py` so local ONNX files are used

Test images are **synthetic** (drawn by `generate_invoices.py` with Noto CJK). Numbers below are real for these three files; phone photos of paper invoices will be harder.

## Method (hardware simulation)

The ~15-year-old laptop is **not** connected. Simulation on this shared Linux box:

| Constraint | How it was applied | Verified |
|---|---|---|
| CPU only, no GPU | `CUDA_VISIBLE_DEVICES=""`, `EngineConfig.onnxruntime.use_cuda=false`; ORT device=`CPU` | `ort.get_device()==CPU`; providers `['AzureExecutionProvider','CPUExecutionProvider']` (Azure EP unused) |
| 2 cores | `taskset -c 0,1`; `OMP/MKL/OPENBLAS/ORT_NUM_THREADS=2`; ORT intra=2, inter=1 | `sched_getaffinity` = `{0,1}`; `nproc=2` inside the jail |
| ~4 GB RAM | `ulimit -v 3670016` (3.5 GiB virtual address cap). Box itself has **15 GiB**; process peak is what matters | `rlimit_as = 3758096384` bytes; run did **not** hit the cap |
| Serial | one process, one image at a time, two timed passes each, JSON persisted | `out_rapidocr.json` |
| Preprocess | `cv2` resize long side **1280** before OCR; RapidOCR `Global.max_side_len=1280` | clean/blur → 880×1280; rotated → 938×1280 |
| Classifier | **off** (`Global.use_cls=false`) | `use_cls: false` in JSON |

Host CPU (for interpreting speed, not RAM): 8× Intel Xeon (KVM), flags include `sse4_2 avx avx2 avx512f`. **This is much newer than a 2011 laptop.** Wall-clock here is a **best-case / lower bound** vs Sandy Bridge (AVX, no AVX2).

Peak memory sources (all agree ~470 MB RSS):

- Parent `wait4` `ru_maxrss` = **469816 KB**
- `/proc/<pid>/status` VmHWM = **469816 KB**
- 20 ms RSS sampler peak = **470040 KB**
- In-process `resource.ru_maxrss` = **469816 KB**
- Steady RSS after infer is lower (~207–267 MB); the **peak** is a transient during ONNX alloc (report the peak)

`VmPeak` (virtual, not RSS) reached **1141936 KB (~1.09 GiB)**. That is address space, not RAM used.

## Per-image measured numbers

Engine init (model load): **0.514 s**, RSS after init **151 MB**. Whole-process wall (init + 3 images × 2 passes): **9.343 s**. Crash: **no**. OOM: **no**. Exit 0.

Ground truth (all three images share the same invoice content):

- invoice number: `25317000000123456789`
- date: `2026年03月15日`
- 价税合计 / 小写 amount: `12880.00`

| Image | Wall first (s) | Wall second / warm (s) | Internal elapse (s) | Mean conf | Peak RSS of process at that point (VmHWM) | Crash | invoice# | date | 价税合计 amount |
|---|---:|---:|---:|---:|---|---|---|---|---|
| invoice_clean.png | 1.801 | 1.346 | 1.341 | 0.9975 | 369 MB after 1st / 402 MB after 2nd | N | got `25317000000123456789` = GT, **Y** | got `2026年03月15日` = GT, **Y** | got `12880.00` from `（小写）￥12880.00`, **Y** |
| invoice_jpeg_blur.jpg | 1.480 | 1.296 | 1.290 | 0.9879 | 429 MB / **470 MB** (run peak) | N | got `25317000000123456789` = GT, **Y** | got `2026年03月15日` = GT, **Y** | got `12880.00` = GT, **Y** |
| invoice_rotated.png (5.2°) | 1.354 | 1.223 | 1.217 | 0.9978 | 470 MB (already at peak) | N | got `25317000000123456789` = GT, **Y** | got `2026年03月15日` = GT, **Y** | got `12880.00` = GT, **Y** |

Process-wide peak RSS: **470040 KB ≈ 459 MiB ≈ 0.45 GiB**.

Notes on accuracy:

- All **three key fields matched GT on all three images**. These are synthetic renders, not camera captures.
- Rotated image: line **order** is scrambled (号码/日期 swapped, table cells mixed) because the direction classifier is off. Regex still found the three fields in the concatenated text. Layout-sensitive parsing would fail; field regex on the full text did not.
- JPEG+blur: confidence dropped (~0.988 vs ~0.998); the three fields still exact.
- **金额仍必须人核.** Matching on three synthetic tickets is not a license to skip review.

## Does it fit 4 GB with OS headroom?

- OCR process peak RSS **0.45 GiB**. Virtual peak **1.09 GiB**. 3.5 GiB `ulimit -v` was not touched.
- 4 GiB machine: OCR itself is **not tight** (~11% of 4 GiB). Leave **~1–1.5 GiB** for Windows + desktop; that still leaves ~2 GiB free if nothing else heavy is open.
- 4 GiB **is** tight as a *whole PC* if Chrome + Excel + AV are already resident. Mitigation: serial OCR, close browsers, pagefile/swap on, do not load PaddlePaddle.
- 32-bit Windows 2 GiB user VAS: virtual peak 1.09 GiB might still boot, but current wheels are **64-bit manylinux/win_amd64**. Plan on 64-bit OS.

## One-sentence conclusion

**可以：RapidOCR + CPU ONNX Runtime + PP-OCRv4 中文 mobile 在 2 核 / ~4GB / 无 GPU 下能稳定出草稿（本合成票三字段全对、峰值 RSS ~470MB），金额必须人核；若无 AVX2 或仍是 Win7，不要上 PaddleOCR 3.x / GPU ORT，改走旧版 CPU ORT 或 OpenVINO/MNN（后两者此处未实测）。**

## Executable plan for the old laptop

OS/RAM not confirmed (Win7 vs Win10). Detect first, then pick a row.

### 0) Inventory the PC (do this first)

```bat
winver
systeminfo | findstr /B /C:"OS Name" /C:"Total Physical Memory" /C:"System Type"
wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors
```

AVX2 check (Sysinternals Coreinfo, or in Python 3):

```python
import sys
print(sys.version)
try:
    import cpuinfo  # optional
except ImportError:
    pass
# crude: if the official onnxruntime import dies with OSError/illegal instruction, you have no usable AVX path
```

Linux equivalent: `grep -o 'avx2' /proc/cpuinfo | head`. 2011 Sandy Bridge = SSE4.2+AVX, **no AVX2**.

### 1) Preferred path (Win10/11 64-bit, ≥4 GB, AVX2 present)

Do **not** install `onnxruntime-gpu`, CUDA, or full PaddlePaddle.

```bat
python -m venv C:\ocr-venv
C:\ocr-venv\Scripts\pip install rapidocr onnxruntime opencv-python-headless
```

Copy these CPU ONNX files next to the script (already in this repo under `models/`):

- `ch_PP-OCRv4_det_mobile.onnx`
- `ch_PP-OCRv4_rec_mobile.onnx`

Env + run (serial, 2 threads, cls off, long side 1280, disk-backed):

```bat
set CUDA_VISIBLE_DEVICES=
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set ORT_NUM_THREADS=2
C:\ocr-venv\Scripts\python run_serial.py invoice1.jpg invoice2.jpg --out-dir ocr_out --threads 2
```

Engine settings that were measured here (`pipeline.py` / `run_serial.py`):

- `Det/Rec.engine_type = onnxruntime`
- `Det/Rec` = Chinese **mobile** **PP-OCRv4**
- `EngineConfig.onnxruntime.use_cuda = false`
- intra_op_num_threads=2, inter_op_num_threads=1, `enable_cpu_mem_arena=false`
- `Global.use_cls = false` (no straighten/direction classifier)
- resize long side 1280; one image at a time; write JSON under `ocr_out/` including `ledger_draft.json`
- Human UI: show image + extracted 号码/日期/价税合计; **amount checkbox required** before posting to the ledger

Linux 64-bit is the same packages; pin with `taskset -c 0,1` if the machine has more cores you want to leave free.

### 2) Worst case: 4 GB RAM, **no AVX2** (and/or Win7)

This host has AVX2/AVX512; **no-AVX2 was not executed here.** Treat the following as a fallback ladder, not a measured guarantee.

**A. Try the CPU `onnxruntime` wheel anyway (runtime ISA dispatch).** Official ORT CPU builds contain AVX2 *and* lower kernels and *may* run on AVX-only CPUs. If `import onnxruntime` or first infer raises `Illegal instruction` / `0xC000001D`, abort this row.

**B. Older CPU ORT (best first fallback if A SIGILLs).** Not installed/run here.

- Win10 64-bit, Python 3.8–3.10: `pip install "onnxruntime==1.16.3"` (last easy py3.8-era CPU wheel family) + an older RapidOCR that still speaks that ORT (`rapidocr-onnxruntime` 1.3.x era, or RapidOCR 1.x/2.x). Keep the same PP-OCRv4 **mobile ONNX** files.
- Win7 64-bit: official CPython stops at **3.8**. Current `onnxruntime==1.29` requires **Python ≥3.11** and will not install. Use Python 3.8.10 + `onnxruntime==1.16.3` (or 1.13.1 if 1.16 still SIGILLs) + old `rapidocr-onnxruntime`. Win7 is also missing modern Universal CRT/API sets; if the wheel won’t load, this PC is a **Linux live USB / a newer Windows disk** job, not a Paddle GPU job.

**C. RapidOCR `EngineType.OPENVINO`.** RapidOCR 3.9 lists `openvino`. `openvino` is **not** in this venv and was **not** run. OpenVINO CPU typically needs SSE4.2 (2011 Core i5 has it). `pip install openvino` then set Det/Rec `engine_type=openvino`, `inference_num_threads=2`. Untested.

**D. RapidOCR `EngineType.MNN`.** Also listed, **not** installed, **not** run. Smaller runtime, often friendlier to old x86. Untested.

**E. Paddle-Lite / PaddlePaddle full.** Full **PaddleOCR 3.7 + PaddlePaddle 3.3.1 already crashed on this modern CPU** (OneDNN PIR). Do **not** ship that stack to a 4 GB box. Paddle-Lite would be a separate mobile/x86 build; **untested**, and not needed if B/C/D works.

**F. Build ORT from source** with AVX2 off (`-DMLAS_DONT_USE_AVX2` / target `sandybridge` or SSE4.2). Untested; last resort.

Stability on 4 GB regardless of ISA:

- 64-bit OS; pagefile ≥ 4 GB
- 2 threads max; serial images; long side 1280; cls off
- `enable_cpu_mem_arena=false` (already)
- persist each result to disk before the next image (`run_serial.py`)
- do not keep PDF renderers, browsers, or PaddlePaddle loaded
- if RSS climbs across files, restart the Python process every N images (here, 3 images did not leak past 470 MB peak)

### 3) What not to do

- `pip install onnxruntime-gpu` / CUDA / TensorRT — no GPU, extra RAM, wrong EP
- `pip install paddleocr paddlepaddle` on the old PC — measured crash even here; huge (libpaddle.so ~254 MB + oneDNN + MKL)
- Batch / multiprocessing OCR
- Trust 价税合计 without a human

## What failed or changed vs the original direction

- **PaddleOCR path abandoned** (already broken in `out_paddleocr.json`; spec said not to run full Paddle on the constrained machine). Root: OneDNN `ConvertPirAttribute2RuntimeAttribute` NotImplementedError.
- **GNU `/usr/bin/time` missing** (apt not permitted). Replaced with `rss_wrap.py` `wait4` + `/proc` VmHWM. These are the same counters `time -v` uses for MaxRSS.
- **No-AVX2 not runnable** on this AVX512 host (no qemu-user). Fallback ladder documented as untested except “current ORT CPU wheel runs *here*”.
- **Win7 vs Win10 unknown**; plan branches on that. Current venv (`onnxruntime 1.29` / Python 3.13) will **not** install on Win7.
- Classifier left **off** as specified; rotated invoice still yielded the three fields via regex, with scrambled line order.
- `pipeline.py` now pins local ONNX paths and `use_cuda=false` (behavior already intended).
- Speed on a real 2011 dual-core will be **worse** than 1.2–1.8 s/image measured here; still plausible for “draft then human” if it stays under ~10–20 s/page. Re-measure on the actual laptop after install.

## Raw artifacts

- `/workspace/ocr-bench/out_rapidocr.json` — per-image texts, times, field match
- `/workspace/ocr-bench/out_rapidocr.json.mem.json` — wait4 MaxRSS
- `/workspace/ocr-bench/run_limited.sh` — how the jail was applied
- `/workspace/ocr-bench/run_serial.py` — laptop runner
