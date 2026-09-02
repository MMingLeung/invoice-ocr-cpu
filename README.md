# invoice-ocr-cpu

低配 CPU 上的中文发票 OCR。模型用 PP-OCRv4 中文 mobile，运行时用 RapidOCR + ONNX Runtime，不要装完整 PaddleOCR / PaddlePaddle。

金额必须人核。机器只出台账草稿。

## 跑一张

```bash
python -m pip install -r requirements.txt
python run_serial.py 发票.jpg --out-dir ocr_out --threads 2
```

先把长边缩到 1280，关掉方向分类，一次一张，结果写到 `ocr_out/`。

老电脑安装步骤见 `INSTALL.md`。上机自检：`python check_old_pc.py`。

## 仓库里有什么

- `pipeline.py` / `run_serial.py`：识别与落盘
- `models/`：把三个 `.onnx` 放这里（仓库不收录大文件）
- `check_old_pc.py`：查内存、指令集、ORT 能不能 import
- `results.md`：双核限内存实测
- `generate_invoices.py`：合成测试票（不是真票）

不含真实发票图片。
