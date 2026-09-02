#!/usr/bin/env python3
"""Create 3 synthetic Chinese VAT-style invoice images with known ground truth."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = Path("/workspace/ocr-bench/images")
OUT.mkdir(parents=True, exist_ok=True)

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

GT = {
    "invoice_number": "25317000000123456789",
    "date": "2026年03月15日",
    "date_alt": "2026-03-15",
    "amount": "12880.00",
    "amount_cn": "壹万贰仟捌佰捌拾圆整",
    "buyer": "示例科技有限公司",
    "seller": "北斗商贸有限公司",
    "tax_id_buyer": "91110000MA01234567",
    "tax_id_seller": "91310000MA07654321",
}


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size, index=0)


def draw_invoice() -> Image.Image:
    w, h = 1600, 1100
    img = Image.new("RGB", (w, h), (255, 255, 252))
    d = ImageDraw.Draw(img)

    # light red border like VAT invoices
    margin = 28
    d.rectangle([margin, margin, w - margin, h - margin], outline=(180, 40, 40), width=3)
    d.rectangle([margin + 8, margin + 8, w - margin - 8, h - margin - 8], outline=(180, 40, 40), width=1)

    title_f = font(FONT_BOLD, 40)
    sub_f = font(FONT_REG, 22)
    body_f = font(FONT_REG, 20)
    small_f = font(FONT_REG, 16)
    num_f = font(FONT_BOLD, 22)

    d.text((w // 2, 70), "增值税电子普通发票", font=title_f, fill=(180, 30, 30), anchor="mt")
    d.text((w // 2, 118), "电子发票（普通发票）", font=sub_f, fill=(120, 40, 40), anchor="mt")

    d.text((90, 160), f"发票号码：{GT['invoice_number']}", font=num_f, fill=(20, 20, 20))
    d.text((980, 160), f"开票日期：{GT['date']}", font=num_f, fill=(20, 20, 20))

    # buyer / seller boxes
    d.rectangle([70, 210, 1530, 360], outline=(180, 40, 40), width=2)
    d.line([800, 210, 800, 360], fill=(180, 40, 40), width=1)
    d.text((90, 222), "购买方信息", font=small_f, fill=(180, 40, 40))
    d.text((90, 250), f"名称：{GT['buyer']}", font=body_f, fill=(20, 20, 20))
    d.text((90, 285), f"统一社会信用代码：{GT['tax_id_buyer']}", font=body_f, fill=(20, 20, 20))
    d.text((90, 320), "地址、电话：北京市海淀区中关村大街1号 010-88886666", font=body_f, fill=(20, 20, 20))

    d.text((820, 222), "销售方信息", font=small_f, fill=(180, 40, 40))
    d.text((820, 250), f"名称：{GT['seller']}", font=body_f, fill=(20, 20, 20))
    d.text((820, 285), f"统一社会信用代码：{GT['tax_id_seller']}", font=body_f, fill=(20, 20, 20))
    d.text((820, 320), "地址、电话：上海市浦东新区世纪大道88号 021-66668888", font=body_f, fill=(20, 20, 20))

    # table
    table_top = 380
    table_bot = 720
    cols = [70, 420, 560, 700, 860, 1040, 1200, 1530]
    headers = ["货物或应税劳务、服务名称", "规格型号", "数量", "单价", "金额", "税率", "税额"]
    d.rectangle([70, table_top, 1530, table_bot], outline=(180, 40, 40), width=2)
    header_y = table_top + 36
    d.line([70, table_top + 56, 1530, table_top + 56], fill=(180, 40, 40), width=1)
    for x in cols[1:-1]:
        d.line([x, table_top, x, table_bot], fill=(180, 40, 40), width=1)
    # header labels
    mids = [(cols[i] + cols[i + 1]) / 2 for i in range(len(cols) - 1)]
    for mid, hd in zip(mids, headers):
        d.text((mid, header_y), hd, font=small_f, fill=(20, 20, 20), anchor="mm")

    rows = [
        ["办公用品*复印纸", "A4", "100", "45.00", "4500.00", "13%", "585.00"],
        ["计算机配套设备*键盘", "标准", "20", "80.00", "1600.00", "13%", "208.00"],
        ["软件服务*技术服务费", "/", "1", "5300.00", "5300.00", "6%", "318.00"],
    ]
    for i, row in enumerate(rows):
        y = table_top + 90 + i * 50
        for mid, val in zip(mids, row):
            d.text((mid, y), val, font=body_f, fill=(20, 20, 20), anchor="mm")

    d.line([70, table_bot - 70, 1530, table_bot - 70], fill=(180, 40, 40), width=1)
    d.text((90, table_bot - 42), "合计", font=body_f, fill=(20, 20, 20))
    d.text((950, table_bot - 42), "11400.00", font=body_f, fill=(20, 20, 20), anchor="mm")
    d.text((1365, table_bot - 42), "1111.00", font=body_f, fill=(20, 20, 20), anchor="mm")

    # 价税合计
    d.rectangle([70, 720, 1530, 820], outline=(180, 40, 40), width=2)
    d.text((90, 750), f"价税合计（大写）：人民币{GT['amount_cn']}", font=num_f, fill=(20, 20, 20))
    d.text((1100, 750), f"（小写）¥{GT['amount']}", font=num_f, fill=(180, 20, 20))
    d.text((90, 790), "备注：合成测试发票，非真实票据。", font=small_f, fill=(90, 90, 90))

    d.rectangle([70, 820, 1530, 980], outline=(180, 40, 40), width=2)
    d.text((90, 840), "开票人：张三    收款人：李四    复核：王五", font=body_f, fill=(20, 20, 20))
    d.text((90, 880), "销售方开户行及账号：中国银行上海分行 6222021234567890123", font=body_f, fill=(20, 20, 20))
    d.text((90, 920), "购买方开户行及账号：工商银行北京分行 6222089876543210987", font=body_f, fill=(20, 20, 20))
    d.text((w // 2, 1040), "本发票为程序合成图像，仅用于OCR基准测试", font=small_f, fill=(140, 140, 140), anchor="mt")
    return img


def to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main() -> None:
    clean = draw_invoice()
    clean_path = OUT / "invoice_clean.png"
    clean.save(clean_path, format="PNG")

    # JPEG compressed + slight blur
    bgr = to_bgr(clean)
    blur = cv2.GaussianBlur(bgr, (5, 5), 1.1)
    jpeg_path = OUT / "invoice_jpeg_blur.jpg"
    cv2.imwrite(str(jpeg_path), blur, [int(cv2.IMWRITE_JPEG_QUALITY), 28])

    # slight rotation (~5 deg) on a white canvas
    rot = clean.rotate(5.2, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 252))
    # crop/pad back to similar size while keeping all content
    rot_path = OUT / "invoice_rotated.png"
    rot.save(rot_path, format="PNG")

    gt_path = OUT / "ground_truth.json"
    meta = {
        "ground_truth": GT,
        "images": {
            "invoice_clean.png": {"variant": "clean", "format": "png"},
            "invoice_jpeg_blur.jpg": {"variant": "jpeg_q28_gaussian_blur", "format": "jpeg"},
            "invoice_rotated.png": {"variant": "rotated_5.2deg", "format": "png"},
        },
    }
    gt_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    for p in [clean_path, jpeg_path, rot_path]:
        im = Image.open(p)
        print(f"wrote {p} size={im.size} bytes={p.stat().st_size}")
    print("gt", GT)


if __name__ == "__main__":
    main()
