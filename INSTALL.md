# 老电脑发票 OCR 安装清单

基于 `results.md` 的实测（**不要重跑同一套 bench**）。
财务金额必须人工核对。本清单按 **4GB RAM / 无 GPU / 无 AVX2** 的最差情况留后路。
用户机器是 Win7 还是 Win10、确切内存尚未确认：先跑文末「上机自检」，再选对应路径。

## 结论（一句）

可以用于「机器出台账草稿 + 人核定稿」：用 RapidOCR + ONNX Runtime CPU + PP-OCRv4 中文 mobile，不要在老本上装完整 PaddleOCR；金额必须人核。

## 实测摘要（共享电脑，2 核 + 3.5GiB 地址空间，无 GPU）

| 引擎 | 稳态耗时 | 峰值 RSS | 崩溃 | 发票号码/日期/价税合计 |
| --- | --- | --- | --- | --- |
| RapidOCR 3.9.2 + ORT 1.29.0 CPU | 约 1.2–1.3 s/张 | 约 280–425 MiB | 无 | 3/3 |
| PaddleOCR 3.7 默认 MKLDNN | — | — | 崩（oneDNN NotImplementedError） | — |
| PaddleOCR 关 MKLDNN | 约 8–10 s/张 | 约 0.6–0.9 GiB | 无 | 3/3 |

该主机 **有 AVX2**，2011 年左右无 AVX2 的机器会更慢（若干秒/张仍可接受），内存仍远低于 4GB。完整 PaddlePaddle 不要作为主运行时。

## 统一约定（所有路径都遵守）

- 模型：`ch_PP-OCRv4_det_mobile.onnx` + `ch_PP-OCRv4_rec_mobile.onnx`（cls 模型可带上但默认关掉）
- 发票已基本摆正：`use_cls=False`（5.2° 旋转实测仍 3/3；开 cls 大约 +0.2 s）
- 长边缩到 **1280**，单张串行，每张结果立刻落盘
- 线程：`OMP_NUM_THREADS=2`、`ORT_NUM_THREADS=2`、ORT `intra_op_num_threads=2`、`enable_cpu_mem_arena=False`
- **不要** `pip install paddlepaddle paddleocr`
- 金额字段只出草稿，界面/台账必须标「待人核」

现成脚本：`pipeline.py`、`run_serial.py`。Windows 上把 `pipeline.py` 里模型路径改成相对路径 `models/`。

---

## 路径 A（优先）：Windows 10/11 x64，4GB

在一台能上网的电脑下好包，再拷到老本（老本网可能很慢或不稳）。

1. 装 [Python 3.10 或 3.11 **64-bit**](https://www.python.org/downloads/windows/)（不要 3.13；老本上轮子更全）。
2. 装 [VC++ 2015–2022 x64 运行库](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)。
3. 拷贝本目录的 `models\`、`pipeline.py`、`run_serial.py`。
4. 安装（CPU only）：

```bat
py -3.10 -m venv venv
venv\Scripts\activate
python -m pip install -U pip
pip install rapidocr onnxruntime opencv-python-headless
```

5. 自检（见文末脚本）。若 `import onnxruntime` 直接 **Illegal instruction / 0xC000001D**，跳到路径 C，不要继续。
6. 跑：

```bat
set CUDA_VISIBLE_DEVICES=
set OMP_NUM_THREADS=2
set ORT_NUM_THREADS=2
set MKL_NUM_THREADS=2
python run_serial.py 发票1.jpg 发票2.jpg --out-dir ocr_out --threads 2
```

4GB 够用：实测峰值约 0.4GB，给 Windows 留 1.5–2GB。跑 OCR 时尽量别开 Chrome + 大型表格。一次只处理一张。

---

## 路径 B：Windows 7 x64 最差情况

官方栈已经丢掉 Win7，按这个钉死版本，不要装「最新」。

| 组件 | 版本 | 原因 |
| --- | --- | --- |
| Python | **3.8.10 64-bit** | CPython 官方最后支持 Win7 的是 3.8；3.9+ 不行 |
| ONNX Runtime | **1.11.1** | 官方预编译大约到 1.11.1 还能在 Win7 跑；1.12+ 缺 `api-ms-win-core-heap-l2-1-0.dll` |
| OCR 包 | **rapidocr-onnxruntime==1.4.4** | 旧包，Python 3.6–3.12；RapidOCR 3.9 绑的是新 ORT，Win7 上不要用 |
| OpenCV | **opencv-python-headless==4.10.0.84** | Python 3.8 能装的较新 headless 轮 |
| 运行库 | VC++ 2015–2019 x64 | ORT 1.11 需要 |

步骤：

1. 先装 Win7 SP1 + 平台更新（KB2670838）+ [VC++ x64 运行库](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)。缺 DLL 时先补这个，不要换 OCR 引擎。
2. 装 Python 3.8.10 x64，勾选 Add to PATH。
3. 离线拷模型（同上三个 `.onnx`）。PP-OCRv4 mobile ONNX 是 opset 11 左右，ORT 1.11 能跑。
4. 安装：

```bat
py -3.8 -m venv venv
venv\Scripts\activate
python -m pip install -U "pip<24.1"
pip install "numpy<1.25" "opencv-python-headless==4.10.0.84" "onnxruntime==1.11.1" "rapidocr-onnxruntime==1.4.4"
```

5. Win7 不要用 `pipeline.py` 里 RapidOCR 3 的 `EngineType` API。最小调用：

```python
from rapidocr_onnxruntime import RapidOCR
engine = RapidOCR(
    det_model_path=r"models\ch_PP-OCRv4_det_mobile.onnx",
    rec_model_path=r"models\ch_PP-OCRv4_rec_mobile.onnx",
    use_angle_cls=False,
)
result, elapse = engine(img)  # img 已长边缩到 1280
```

6. 若 1.11.1 仍报缺 `api-ms-win-core-*`：先装 KB 和 VC 红包。**不要**去装第三方魔改 ORT 当默认（未在本 bench 验证）。实在不行把整台流程迁到 Win10，Win7 只当扫描/拍照端。

---

## 路径 C：无 AVX2 / 导入 ORT 就 SIGILL

2011 前后笔记本常见：Sandy Bridge **有 AVX、无 AVX2**；再老（Core 2 / 一代 i3）可能连 AVX 都没有。

本 bench **没有**在无 AVX2 的 CPU 上跑过（共享机有 AVX2）。按这个顺序处理：

1. 跑 `check_old_pc.py`。若 flags 里没有 `avx2`，先不要装最新 `onnxruntime`（1.2x 部分预编译会在导入时 SIGILL）。
2. **Win10 + 无 AVX2**：钉 `onnxruntime==1.16.3`（仍走 CPU、运行时分发 SSE/AVX 的可能性更大；仍需上机验证）。

```bat
pip uninstall -y onnxruntime
pip install "onnxruntime==1.16.3"
python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
```

3. 若 1.16.3 仍 SIGILL：退到 **1.11.1**，OCR 包改用路径 B 的 `rapidocr-onnxruntime==1.4.4`。
4. **不要**把 OpenVINO 当无 AVX2 后路——近年 OpenVINO 对老 CPU 更苛刻，本 bench 未测。
5. 再不行：这台老本当拍照/扫描仪，OCR 放到另一台 Win10。不要为了救 Win7 去装完整 PaddlePaddle（实测默认 MKLDNN 崩，关掉后更慢更吃内存）。

---

## 稳定措施

- 一次一张，写完 `ocr_out\<stem>.json` 再读下一张；崩溃也不丢前面的。
- 进程挂了就重启 Python，不要长驻几天。
- JPEG 先缩再识别；扫描件 150–200 dpi 足够，不要 600 dpi 原图直接喂。
- 台账草稿字段：发票号码、开票日期、价税合计；**合计金额永远 `human_verify=true`**。
- 关掉 Windows 视觉效果、别在跑 OCR 时休眠。4GB 机器先关浏览器再跑。

## 上机自检

```bat
python check_old_pc.py
```

看输出里的 OS、物理内存、AVX/AVX2、以及 `onnxruntime` 能否在子进程里 import。有 SIGILL 就走路径 C，不要当「偶发」。
