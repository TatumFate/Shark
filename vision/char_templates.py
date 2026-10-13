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
import recognize_chars as rc  # noqa: E402


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
    arr = common.crop_to_ink(arr)
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description="生成字符模板")
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--samples", default=str(config.ROOT / "data" / "pieces" / "samples"),
                    help="实拍样本目录：用你的实物棋子生成模板，识别率通常明显提升")
    ap.add_argument("--no-samples", action="store_true", help="只用系统字体模板")
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

    # 用你的实拍棋子生成模板：走的是和识别时**完全相同**的归一化流程，
    # 所以"字体不像"的问题会被直接消掉（真实棋子的字形才是最重要的模板）
    n_real = 0
    sample_dir = Path(args.samples)
    if not args.no_samples and sample_dir.exists():
        valid = set(config.CHARS_RED) | set(config.CHARS_BLACK)
        ratio = rc.read_crop_ratio(sample_dir)
        for f in sorted(sample_dir.glob("*.png")):
            parts = f.stem.split("_")
            if len(parts) < 2 or parts[0] not in valid or parts[1] not in ("red", "black"):
                continue
            ch, side = parts[0], parts[1]
            img = common.imread_cn(f)
            if img is None:
                continue
            h, w = img.shape[:2]
            ink = rc.ink_patch(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                               w / 2.0, h / 2.0, max(6, int(w / (2.0 * ratio) * 0.70)))
            if ink is None:
                continue
            mats.append(common.normalize_patch(ink, size))
            names.append(f"real_{f.stem}")
            metas.append({"font": "real", "char": ch, "side": side})
            n_real += 1
        if n_real:
            print(f"[模板] real（你的实物棋子）：{n_real} 张样本")
        else:
            print(f"[模板] {sample_dir} 里没有已标注样本，跳过（命名格式：车_red_01.png）")
    elif not args.no_samples:
        print(f"[模板] 还没有实拍样本目录 {sample_dir}，先用系统字体模板")

    stack = np.stack(mats).astype(np.uint8)
    out = config.DIR_TEMPLATE / "templates.npz"
    np.savez_compressed(out, templates=stack,
                        names=np.array(names),
                        meta=np.array([f"{m['font']}|{m['char']}|{m['side']}" for m in metas]))
    print(f"[保存] {out}  形状={stack.shape}")

    # 预览图：一行一种字体/来源（加了实拍模板后总数会变多，行数要按实际数量算）
    cols = len(chars)
    rows = max(1, (len(mats) + cols - 1) // cols)
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
