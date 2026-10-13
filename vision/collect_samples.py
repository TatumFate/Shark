# -*- coding: utf-8 -*-
"""把你的**实物象棋**采集进系统：切样本 → 量直径 → 打标签 → 自评识别率。

为什么需要这一步：系统里默认的字符模板是"用系统字体渲染的楷体/黑体"，
你手里那副棋的字形、粗细、油墨、反光都可能不一样。用**你自己的棋子照片**
生成模板，识别率通常能从七八十直接顶到九十以上。

标准流程（详细说明见 vision/README.md 第十节）：
    1) 拍照片：棋子平铺在桌上、互不重叠、字朝上、光线均匀，最好放一把尺子在画面里
    2) 切样本： python vision/collect_samples.py --image vision/data/raw/pieces.png --scale-pick --ref-mm 100
    3) 改文件名：把 unknown 换成棋子上的字，例如 unknown_red_000.png → 车_red_000.png
    4) 生成模板： python vision/char_templates.py
    5) 看识别率： python vision/recognize_chars.py --eval-samples vision/data/pieces/samples

不想用尺子也行：省掉 --scale-pick，脚本照样会切样本，只是量不出毫米尺寸，
那时棋子直径你自己拿尺量一下填进 config.PIECE_DIA 即可。
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import config  # noqa: E402
import detect_pieces as dp  # noqa: E402

CROP_RATIO = 1.5          # 裁出来的边长 = 1.5 × 棋子直径


def grab_camera(index=0):
    """从相机抓一帧（拍实物棋子用）。"""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, 0):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            for _ in range(5):                 # 丢掉前几帧，等自动曝光稳定
                ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                return frame
    raise SystemExit(f"打不开 {index} 号相机")


def detect_discs(img):
    """在一张普通照片里自动找棋子圆盘：多尺度 Hough + 墨色校验。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    gray = cv2.medianBlur(gray, 3)
    h, w = gray.shape
    best = []
    for frac in (0.012, 0.018, 0.026, 0.036, 0.05, 0.07, 0.10):
        r = max(4, int(frac * min(h, w)))
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=2.0 * r,
                                   param1=120, param2=30,
                                   minRadius=int(0.8 * r), maxRadius=int(1.25 * r))
        if circles is not None and len(circles[0]) > len(best):
            best = [tuple(map(float, c)) for c in circles[0]]
    if not best:
        return []
    best = dp.nms(best, 2.0 * float(np.median([c[2] for c in best])))
    # 圆盘里得有"字"（墨色），否则多半是桌面花纹、尺子刻度、硬币上的图案
    keep = []
    for cx, cy, r in best:
        _side, _rr, _dr, ink = dp.classify(img, cx, cy, r)
        if ink >= 0.02:
            keep.append((cx, cy, r, ink))
    return keep


def pick_scale(img, ref_mm):
    """弹出的图上点两个点（例如尺子上 100mm 的两端），算 mm/像素。"""
    pts = []

    def on_mouse(event, x, y, _f, _p):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
            print(f"  第 {len(pts)} 点：({x}, {y})")

    win = "calibrate: 点尺子上已知长度的两端  r=重来  Enter=确认  Esc=取消"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = img.copy()
        for i, p in enumerate(pts):
            cv2.circle(view, p, 5, (0, 255, 255), -1)
            view = common.put_text_cn(view, str(i + 1), (p[0] + 8, p[1] - 26), 22, (0, 255, 255))
        if len(pts) == 2:
            cv2.line(view, pts[0], pts[1], (0, 255, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            pts.clear()
        elif key in (13, 32) and len(pts) == 2:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("已取消")
    cv2.destroyAllWindows()
    px = float(np.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]))
    return ref_mm / px


def main():
    ap = argparse.ArgumentParser(description="采集实物棋子样本")
    ap.add_argument("--image", default=None, help="棋子照片（俯拍、字朝上、不重叠）")
    ap.add_argument("--camera", type=int, default=None, help="直接用相机抓一帧")
    ap.add_argument("--out", default=str(config.ROOT / "data" / "pieces" / "samples"))
    ap.add_argument("--scale-pick", action="store_true", help="弹窗点尺子两端，换算像素→毫米")
    ap.add_argument("--ref-mm", type=float, default=100.0, help="点的两点实际距离(mm)")
    ap.add_argument("--mm-per-px", type=float, default=None,
                    help="已知比例时直接给（例如整场俯拍图是 1.0 mm/像素），省掉点尺子")
    ap.add_argument("--save-photo", action="store_true", help="把原始照片也留一份到 data/raw")
    args = ap.parse_args()

    if args.camera is not None:
        img = grab_camera(args.camera)
        print(f"[相机] 抓到 {img.shape[1]}x{img.shape[0]}")
    elif args.image:
        img = common.imread_cn(args.image)
        if img is None:
            raise SystemExit(f"读不到图片：{args.image}")
        print(f"[照片] {img.shape[1]}x{img.shape[0]}  {args.image}")
    else:
        raise SystemExit("请给 --image <照片> 或 --camera 0")

    mm_per_px = None
    if args.scale_pick:
        mm_per_px = pick_scale(img, args.ref_mm)
        print(f"[比例] 1 像素 = {mm_per_px:.4f} mm")
    elif args.mm_per_px:
        mm_per_px = float(args.mm_per_px)
        print(f"[比例] 1 像素 = {mm_per_px:.4f} mm（命令行给定）")

    discs = detect_discs(img)
    if not discs:
        raise SystemExit("没找到棋子。检查：棋子字朝上、彼此不重叠、光线别太暗、照片别太糊")
    radii = np.array([d[2] for d in discs])
    r_med = float(np.median(radii))
    print(f"[检测] 找到 {len(discs)} 枚候选（半径中位数 {r_med:.1f} px，"
          f"离散度 {radii.std() / max(r_med, 1e-6) * 100:.0f}%）")

    dia_mm = None
    if mm_per_px:
        dia_mm = 2 * r_med * mm_per_px
        print(f"[直径] 棋子直径约 {dia_mm:.1f} mm（Hough 半径估计有 ±10% 误差）")
        print(f"       建议用卡尺量一下实物，然后把 config.PIECE_DIA 设成实测值；"
              f"这个数字只用来先自查量级")
        if abs(dia_mm - config.PIECE_DIA) / config.PIECE_DIA > 0.12:
            print(f"       注意：现在 config.PIECE_DIA = {config.PIECE_DIA}，差了 "
                  f"{abs(dia_mm - config.PIECE_DIA) / config.PIECE_DIA * 100:.0f}%，"
                  f"改完再跑一遍检测")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pad = int(round(CROP_RATIO * r_med)) + 4
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    side = int(round(2 * CROP_RATIO * r_med))
    tiles, names = [], []
    for i, (cx, cy, r, _ink) in enumerate(sorted(discs, key=lambda d: (d[1], d[0]))):
        sz = int(round(2 * CROP_RATIO * r))
        x0 = int(round(cx)) + pad - sz // 2
        y0 = int(round(cy)) + pad - sz // 2
        crop = padded[max(0, y0):y0 + sz, max(0, x0):x0 + sz]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
        color, _rr, _dr, _ink2 = dp.classify(img, cx, cy, r)
        name = f"unknown_{color}_{i:03d}.png"
        common.imwrite_cn(out / name, crop)
        tiles.append(crop)
        names.append(name)

    # 拼一张"索引图"：一眼看清每张样本是哪个文件、什么颜色，方便批量改名
    if tiles:
        cell, cols = 110, 8
        rows = (len(tiles) + cols - 1) // cols
        sheet = np.full((rows * (cell + 26) + 8, cols * (cell + 6) + 6, 3), 24, np.uint8)
        labels = []
        for i, t in enumerate(tiles):
            rr_, cc = divmod(i, cols)
            y0 = 6 + rr_ * (cell + 26)
            x0 = 6 + cc * (cell + 6)
            sheet[y0:y0 + cell, x0:x0 + cell] = cv2.resize(t, (cell, cell),
                                                           interpolation=cv2.INTER_AREA)
            short = names[i].replace("unknown_", "").replace(".png", "")
            labels.append((short, (x0, y0 + cell + 2), 14, (0, 255, 255)))
        common.put_texts_cn(sheet, labels)
        common.imwrite_cn(out / "_index.png", sheet)

    common.save_json({"crop_ratio": CROP_RATIO, "mm_per_px": mm_per_px,
                      "piece_dia_mm": None if dia_mm is None else round(dia_mm, 1),
                      "radius_px_median": round(r_med, 1), "count": len(tiles),
                      "source": args.image or f"camera{args.camera}"},
                     out / "_meta.json")

    print("-" * 64)
    print(f"已切出 {len(tiles)} 张样本 → {out}")
    print(f"索引图：{out / '_index.png'}（点开看一眼，记下每张的编号）")
    print("接下来：")
    print(f"  1) 把文件名里的 unknown 换成棋子上的字（保持两段不变），例如：")
    print(f"       {out.name}/unknown_red_000.png  →  {out.name}/车_red_000.png")
    print("     颜色脚本已经自动判好了（red/black），你只需要填字")
    print("  2) 重新生成模板（会把你的实物字形一起加进去）：")
    print("       python vision/char_templates.py")
    print("  3) 看识别率（拿你自己的棋子测出来的真实数字）：")
    print(f"       python vision/recognize_chars.py --eval-samples {out}")
    if args.save_photo or args.camera is not None:
        common.imwrite_cn(config.DIR_RAW / "pieces_source.png", img)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
