# -*- coding: utf-8 -*-
"""棋子检测：在（校正后的）场地俯视图里找出每一枚棋子，并判断红/黑。

两种检测方式（--method auto 会先用 Hough，找不到再退回轮廓法）：
    hough   ：圆形检测，稳，但被遮挡时容易漏
    contour ：先找"亮色圆盘"，再用圆度筛选，速度快

用法：
    # 用合成图 + 真值标定跑一遍（不需要相机）
    python vision/detect_pieces.py --image vision/data/synthetic/field_oblique.png --mode gt --compare-gt

    # 真实照片（先用 calib_field.py 标定过）
    python vision/detect_pieces.py --image vision/data/raw/shot.png --mode calib
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import config  # noqa: E402


def detect_hough(rect, dia_px):
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    gray = cv2.medianBlur(gray, 3)
    r = dia_px / 2.0
    # 单组参数的 Hough 容易漏检，跑几组再合并（重复的由 nms 去掉）
    found = []
    for dp, p1, p2 in ((1.0, 120, 20), (1.2, 120, 26), (1.5, 150, 32)):
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=dia_px * 0.8,
            param1=p1, param2=p2,
            minRadius=int(r * config.R_MIN_RATIO), maxRadius=int(r * config.R_MAX_RATIO),
        )
        if circles is not None:
            found += [tuple(map(float, c)) for c in circles[0]]
    return found


def detect_contour(rect, dia_px):
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    r_ref = dia_px / 2.0
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < np.pi * (r_ref * 0.6) ** 2:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if not (r_ref * config.R_MIN_RATIO <= r <= r_ref * config.R_MAX_RATIO):
            continue
        if area / (np.pi * r * r) < 0.55:
            continue
        out.append((x, y, r))
    return out


def nms(circles, min_dist):
    """按半径当置信度去重，避免同一枚棋子出现两个框。"""
    kept = []
    for c in sorted(circles, key=lambda t: -t[2]):
        if all((c[0] - k[0]) ** 2 + (c[1] - k[1]) ** 2 > min_dist ** 2 for k in kept):
            kept.append(c)
    return kept


def classify(rect, cx, cy, r):
    """看圆盘内部：红色笔画多算红棋，黑色笔画多算黑棋。

    关键：阈值跟着"这枚棋子自己的盘面亮度"走（自适应），而不是全图固定灰度值，
    否则细笔画 + 低分辨率 + 暗光下的棋子会被整片漏掉。
    返回 (side, red_ratio, dark_ratio, ink_ratio)
    """
    rad = max(3, int(r * 0.85))
    x0, y0 = int(round(cx)) - rad, int(round(cy)) - rad
    x1, y1 = x0 + 2 * rad + 1, y0 + 2 * rad + 1
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(rect.shape[1], x1), min(rect.shape[0], y1)
    patch = rect[y0:y1, x0:x1]
    if patch.size == 0:
        return "black", 0.0, 0.0, 0.0
    h, w = patch.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    inside = ((yy - (cy - y0)) ** 2 + (xx - (cx - x0)) ** 2) <= rad ** 2
    if int(inside.sum()) < 20:
        return "black", 0.0, 0.0, 0.0
    b = patch[..., 0][inside].astype(np.int16)
    g = patch[..., 1][inside].astype(np.int16)
    rch = patch[..., 2][inside].astype(np.int16)
    grayv = 0.114 * b + 0.587 * g + 0.299 * rch
    base = float(np.percentile(grayv, 80))        # 盘面（浅色木质）亮度
    dark = grayv < base - config.INK_DELTA
    red = (rch - np.maximum(g, b)) > config.RED_DIFF
    n = float(inside.sum())
    rr, dr = float(red.sum()) / n, float((dark & ~red).sum()) / n
    side = "red" if rr > dr else "black"
    return side, rr, dr, rr + dr


def detect(rect, mm_per_px, method="auto"):
    dia_px = config.PIECE_DIA / mm_per_px
    circles = []
    used = method
    if method in ("auto", "hough"):
        circles = detect_hough(rect, dia_px)
    if method == "auto" and len(circles) < 5:
        used = "contour"
        circles = detect_contour(rect, dia_px)
    pieces = []
    for cx, cy, r in nms(circles, dia_px * 0.75):
        side, rr, dr, ink = classify(rect, cx, cy, r)
        if ink < 0.02:
            continue
        pieces.append({
            "x": round(cx * mm_per_px, 1), "y": round(cy * mm_per_px, 1),
            "r": round(r * mm_per_px, 1), "side": side,
            "ink": round(ink, 4), "red_ratio": round(rr, 4), "dark_ratio": round(dr, 4),
        })
    pieces.sort(key=lambda p: (p["y"], p["x"]))
    return pieces, used


def annotate(rect, pieces, mm_per_px):
    view = rect.copy()
    for i, p in enumerate(pieces):
        c = (int(p["x"] / mm_per_px), int(p["y"] / mm_per_px))
        col = (0, 0, 255) if p["side"] == "red" else (255, 255, 255)
        cv2.circle(view, c, int(config.PIECE_R / mm_per_px), col, 2)
        cv2.drawMarker(view, c, (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
        view = common.put_text_cn(view, str(i + 1), (c[0] + 10, c[1] - 34), 20, (0, 255, 255))
    return view


def compare_with_gt(pieces, gt_path, tol=25.0):
    """跟真值比：给出查全率/查准率/平均定位误差。"""
    gt = common.load_json(gt_path)
    used = [False] * len(gt)
    errs, hit = [], 0
    for p in pieces:
        best, bd = -1, 1e9
        for i, g in enumerate(gt):
            if used[i]:
                continue
            d = float(np.hypot(p["x"] - g["x"], p["y"] - g["y"]))
            if d < bd:
                bd, best = d, i
        if best >= 0 and bd <= tol:
            used[best] = True
            hit += 1
            errs.append(bd)
    mean_err = float(np.mean(errs)) if errs else float("nan")
    print("-" * 56)
    print(f"真值 {len(gt)} 枚 / 检出 {len(pieces)} 枚 / 正确匹配 {hit} 枚")
    print(f"查全率 {hit / len(gt) * 100:5.1f}%   查准率 {hit / max(len(pieces), 1) * 100:5.1f}%"
          f"   平均定位误差 {mean_err:.1f} mm")
    return hit


def main():
    ap = argparse.ArgumentParser(description="检测场地上的棋子")
    ap.add_argument("--image", required=True)
    ap.add_argument("--calib", default=None, help="单应矩阵 json；不给就找 calib/homography_<frame>.json")
    ap.add_argument("--mode", choices=["calib", "gt"], default="calib", help="gt=用合成图真值标定")
    ap.add_argument("--frame", default=config.FRAME)
    ap.add_argument("--mm-per-px", type=float, default=config.MM_PER_PX)
    ap.add_argument("--method", choices=["auto", "hough", "contour"], default="auto")
    ap.add_argument("--name", default="detect")
    ap.add_argument("--compare-gt", action="store_true", help="与合成真值比对精度")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    img = common.imread_cn(args.image)
    if img is None:
        raise SystemExit(f"读不到图片：{args.image}")
    if args.mode == "gt":
        H, _meta = common.load_homography(path=config.DIR_SYN / "gt_homography.json")
    else:
        H, _meta = common.load_homography(args.frame, args.calib)
    if H is None:
        raise SystemExit("没有标定文件。先跑：python vision/calib_field.py --image <图> --mode auto")

    size_mm = (config.FIELD_L, config.FIELD_W)
    rect, mm_per_px = common.rectify(img, H, args.mm_per_px, size_mm)
    common.imwrite_cn(config.DIR_OUTPUT / f"{args.name}_rectified.png", rect)

    pieces, used = detect(rect, mm_per_px, args.method)
    n_red = sum(p["side"] == "red" for p in pieces)
    print(f"检测方式：{used}    检出 {len(pieces)} 枚（红 {n_red} / 黑 {len(pieces) - n_red}）")
    for i, p in enumerate(pieces[:8]):
        print(f"  #{i + 1:2d}  ({p['x']:7.1f}, {p['y']:7.1f}) mm  "
              f"直径 {p['r'] * 2:5.1f} mm  {p['side']}")
    if len(pieces) > 8:
        print(f"  … 其余 {len(pieces) - 8} 枚见 JSON")

    common.save_json({"pieces": pieces, "mm_per_px": mm_per_px, "method": used},
                     config.DIR_OUTPUT / f"{args.name}.json")
    view = annotate(rect, pieces, mm_per_px)
    common.imwrite_cn(config.DIR_OUTPUT / f"{args.name}_annotated.png", view)

    if args.compare_gt:
        compare_with_gt(pieces, config.DIR_SYN / "gt_pieces.json")
    if args.show:
        common.show_or_save("detect", view)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
