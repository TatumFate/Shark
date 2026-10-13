# -*- coding: utf-8 -*-
"""生成"合成测试图"：没有相机也能先把视觉算法跑通、调参。

产出三样东西（都在 vision/data/synthetic/）：
    field_topdown.png   场地俯视图（理想图）
    field_oblique.png   模拟相机斜视拍到的画面（有透视、噪点、暗角）
    gt_homography.json  真值单应矩阵（斜视图像素 → 场地毫米），拿来验证标定
    gt_pieces.json      真值棋子列表（位置/颜色/字符），拿来算检测与识别准确率

用法：
    python vision/make_synthetic.py
    python vision/make_synthetic.py --seed 7 --show
"""
import argparse
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import config  # noqa: E402

WALL_MM = 60.0          # 挡板俯视厚度（示意图，非官方数据）
SIDE_RED = [("帅", 1), ("仕", 2), ("相", 2), ("车", 2), ("马", 2), ("炮", 2), ("兵", 5)]
SIDE_BLACK = [("将", 1), ("士", 2), ("象", 2), ("车", 2), ("马", 2), ("炮", 2), ("卒", 5)]
TRAD = {"帅": "帥", "将": "將", "车": "車", "马": "馬", "炮": "砲"}   # 部分用繁体，模拟真实棋子


def build_piece_list(rng, trad_ratio=0.3):
    pieces = []
    for side, spec in (("red", SIDE_RED), ("black", SIDE_BLACK)):
        for char, n in spec:
            for _ in range(n):
                face = TRAD.get(char, char) if rng.random() < trad_ratio else char
                pieces.append({"side": side, "char": face})
    # 在堆放区里撒点，保证棋子之间不重叠
    placed = []
    min_d = config.PIECE_DIA * 1.15
    for p in pieces:
        for _ in range(4000):
            x = rng.uniform(config.PILE_CX - config.PILE_L / 2 + config.PIECE_R,
                            config.PILE_CX + config.PILE_L / 2 - config.PIECE_R)
            y = rng.uniform(config.PILE_CY - config.PILE_W / 2 + config.PIECE_R,
                            config.PILE_CY + config.PILE_W / 2 - config.PIECE_R)
            if all((x - q["x"]) ** 2 + (y - q["y"]) ** 2 > min_d ** 2 for q in placed):
                placed.append({**p, "x": round(x, 2), "y": round(y, 2),
                               "rot": round(rng.uniform(0, 360), 1)})
                break
    rng.shuffle(placed)
    return placed


def piece_tile(char, side, rot_deg, dia_px, font_path):
    """渲染一枚棋子顶面（带透明底），方便贴到场地图上。"""
    size = int(dia_px * 1.35) | 1
    c = size // 2
    r = dia_px / 2.0
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    d.ellipse([c - r, c - r, c + r, c + r], fill=(226, 212, 190),
              outline=(120, 96, 70), width=max(1, int(dia_px * 0.07)))
    d.ellipse([c - r * 0.84, c - r * 0.84, c + r * 0.84, c + r * 0.84],
              outline=(170, 145, 110), width=1)
    font = ImageFont.truetype(font_path, int(dia_px * 0.60))
    ink = (196, 34, 40) if side == "red" else (28, 28, 30)
    ch = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dc = ImageDraw.Draw(ch)
    bbox = dc.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    dc.text(((size - w) / 2 - bbox[0], (size - h) / 2 - bbox[1]), char, font=font, fill=ink)
    ch = ch.rotate(rot_deg, resample=Image.BICUBIC, center=(c, c))
    return Image.alpha_composite(tile, ch)


def dashed_rect(draw, box, color, width=2, dash=14, gap=10):
    x0, y0, x1, y1 = box
    for x in range(int(x0), int(x1), dash + gap):
        draw.line([(x, y0), (min(x + dash, x1), y0)], fill=color, width=width)
        draw.line([(x, y1), (min(x + dash, x1), y1)], fill=color, width=width)
    for y in range(int(y0), int(y1), dash + gap):
        draw.line([(x0, y), (x0, min(y + dash, y1))], fill=color, width=width)
        draw.line([(x1, y), (x1, min(y + dash, y1))], fill=color, width=width)


def render_topdown(pieces, mm_per_px=1.0, seed=0):
    rng = np.random.default_rng(seed)
    wall_px = int(WALL_MM / mm_per_px)
    fw, fh = int(config.FIELD_L / mm_per_px), int(config.FIELD_W / mm_per_px)
    W, H = fw + 2 * wall_px, fh + 2 * wall_px

    # 地面：浅灰绿 + 噪声 + 光照渐变
    base = rng.normal(0, 4, (H, W, 3)).astype(np.float32)
    floor = np.array([196, 200, 196], np.float32)[None, None, :] + base
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = 1.0 + 0.10 * (gx / W) - 0.06 * (gy / H)          # 轻微明暗渐变
    floor *= grad[..., None]
    canvas = np.clip(floor, 0, 255).astype(np.uint8)
    # 挡板（深灰）
    canvas[:wall_px, :] = canvas[-wall_px:, :] = (92, 96, 100)
    canvas[:, :wall_px] = canvas[:, -wall_px:] = (92, 96, 100)

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    ox, oy = wall_px, wall_px
    mm = lambda x, y: (ox + x / mm_per_px, oy + y / mm_per_px)   # noqa: E731
    line = (90, 110, 90)

    # 球门区（左右贴墙长条）+ 门口感知线
    for x0 in (0.0, config.FIELD_L - config.GOAL_DEPTH):
        p0, p1 = mm(x0, config.GOAL_Y0), mm(x0 + config.GOAL_DEPTH, config.GOAL_Y1)
        d.rectangle([p0, p1], outline=(70, 120, 190), width=3)
        inner = x0 + config.GOAL_DEPTH if x0 == 0 else config.FIELD_L - config.GOAL_DEPTH
        q0, q1 = mm(inner, config.GOAL_Y0), mm(inner, config.GOAL_Y1)
        d.line([q0, q1], fill=(60, 200, 90), width=3)

    # 棋子堆放区、两侧发车区
    p0 = mm(config.PILE_CX - config.PILE_L / 2, config.PILE_CY - config.PILE_W / 2)
    p1 = mm(config.PILE_CX + config.PILE_L / 2, config.PILE_CY + config.PILE_W / 2)
    dashed_rect(d, [p0[0], p0[1], p1[0], p1[1]], line, 2)
    s0 = mm(0, config.FIELD_W - config.START_W)
    s1 = mm(config.START_L, config.FIELD_W)
    dashed_rect(d, [s0[0], s0[1], s1[0], s1[1]], (70, 120, 190), 2)
    t0 = mm(config.FIELD_L - config.START_L, 0)
    t1 = mm(config.FIELD_L, config.START_W)
    dashed_rect(d, [t0[0], t0[1], t1[0], t1[1]], (170, 90, 80), 2)

    # 棋子（贴图需要带 alpha 通道的画布）
    pil = pil.convert("RGBA")
    dia_px = config.PIECE_DIA / mm_per_px
    for i, pc in enumerate(pieces):
        font_path = config.FONT_QI[i % len(config.FONT_QI)]
        tile = piece_tile(pc["char"], pc["side"], pc["rot"], dia_px, font_path)
        cx, cy = mm(pc["x"], pc["y"])
        pil.alpha_composite(tile, (int(cx - tile.width / 2), int(cy - tile.height / 2)))

    return cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR), wall_px


def to_oblique(canvas, rng, out_size=(2400, 1600)):
    """用随机四边形做透视变换，模拟一台斜着拍的相机。"""
    h, w = canvas.shape[:2]
    ow, oh = out_size
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = lambda a: rng.uniform(-a, a)   # noqa: E731
    dst = np.float32([
        [ow * 0.04 + jit(30), oh * 0.06 + jit(30)],
        [ow * 0.97 + jit(30), oh * 0.13 + jit(30)],
        [ow * 0.90 + jit(30), oh * 0.99 + jit(30)],
        [ow * 0.02 + jit(30), oh * 0.87 + jit(30)],
    ])
    H_c2o = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(canvas, H_c2o, (ow, oh), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(70, 74, 78))
    # 模拟光学：轻微模糊 + 噪声 + 暗角
    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    img = np.clip(img.astype(np.float32) + rng.normal(0, 2.0, img.shape), 0, 255)
    yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float32)
    vig = 1.0 - 0.22 * (((xx - ow / 2) / (ow / 2)) ** 2 + ((yy - oh / 2) / (oh / 2)) ** 2)
    img = np.clip(img * np.clip(vig, 0.5, 1.0)[..., None], 0, 255).astype(np.uint8)
    return img, H_c2o


def main():
    ap = argparse.ArgumentParser(description="生成合成场地/棋子测试图")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--mm-per-px", type=float, default=1.0)
    ap.add_argument("--show", action="store_true", help="保存后弹窗预览")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pieces = build_piece_list(rng)
    canvas, wall_px = render_topdown(pieces, args.mm_per_px, args.seed)
    oblique, H_c2o = to_oblique(canvas, np.random.default_rng(args.seed + 1))

    # 真值单应：斜视图像素 → 场地毫米
    s = 1.0 / args.mm_per_px
    H_mm2canvas = np.array([[s, 0, wall_px], [0, s, wall_px], [0, 0, 1]], np.float64)
    H_mm2ob = H_c2o.astype(np.float64) @ H_mm2canvas
    H_ob2mm = np.linalg.inv(H_mm2ob)

    common.imwrite_cn(config.DIR_SYN / "field_topdown.png", canvas)
    common.imwrite_cn(config.DIR_SYN / "field_oblique.png", oblique)
    common.save_json(
        {"H": H_ob2mm.tolist(), "frame": "field", "source": "synthetic ground truth",
         "mm_per_px_render": args.mm_per_px, "image_size": [oblique.shape[1], oblique.shape[0]]},
        config.DIR_SYN / "gt_homography.json",
    )
    common.save_json(pieces, config.DIR_SYN / "gt_pieces.json")
    print(f"[真值] 棋子 {len(pieces)} 枚，已写入 gt_pieces.json")

    if args.show:
        common.show_or_save("synthetic", oblique)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
