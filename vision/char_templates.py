# -*- coding: utf-8 -*-
"""生成棋子汉字模板（角度多版本），供 recognize_chars.py 做模板匹配。

优点：零训练数据，改字体/加写法只要重跑一次。
产出：vision/data/templates/templates.npz  +  preview.png（肉眼检查）

用法：
    python vision/char_templates.py
    python vision/char_templates.py --size 48 --pad 6
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import config  # noqa: E402


def render_char(char, size, font_path):
    """把单个汉字渲染成黑底白字的二值小图（笔画=255）。"""
    big = size * 4                      # 先大后小，边缘更干净
    img = Image.new("L", (big, big), 0)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, int(big * 0.78))
    bbox = d.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((big - w) / 2 - bbox[0], (big - h) / 2 - bbox[1]), char, font=font, fill=255)
    arr = np.asarray(img)
    arr = crop_to_ink(arr)
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description="生成字符模板")
    ap.add_argument("--size", type=int, default=48)
    args = ap.parse_args()
    size = args.size

    chars = sorted(set(config.CHARS_RED) | set(config.CHARS_BLACK))
    names, mats, metas = [], [], []
    for font_path in config.FONT_QI:
        font_name = Path(font_path).stem
        for ch in chars:
            tpl = render_char(ch, size, font_path)
            tpl = common.normalize_patch(tpl, size)
            names.append(f"{font_name}_{ch}")
            mats.append(tpl)
            metas.append({"font": font_name, "char": ch,
                          "side": "red" if ch in config.CHARS_RED else "black"})
        print(f"[模板] {font_name}：{len(chars)} 个字")

    stack = np.stack(mats).astype(np.uint8)
    out = config.DIR_TEMPLATE / "templates.npz"
    np.savez_compressed(out, templates=stack,
                        names=np.array(names),
                        meta=np.array([f"{m['font']}|{m['char']}|{m['side']}" for m in metas]))
    print(f"[保存] {out}  形状={stack.shape}")

    # 预览图：一行字体一张
    cols = len(chars)
    rows = len(config.FONT_QI)
    gap = 6
    canvas = np.full((rows * (size + gap) + gap, cols * (size + gap) + gap), 30, np.uint8)
    for i, m in enumerate(mats):
        r, c = divmod(i, cols)
        y0 = gap + r * (size + gap)
        x0 = gap + c * (size + gap)
        canvas[y0:y0 + size, x0:x0 + size] = m
    common.imwrite_cn(config.DIR_TEMPLATE / "preview.png", canvas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
