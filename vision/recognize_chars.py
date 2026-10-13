# -*- coding: utf-8 -*-
"""棋子汉字识别：模板匹配 + 角度搜索（棋子在地上朝向是随机的）。

思路：
1. 从检测结果拿到每枚棋子的圆心和半径；
2. 抠出棋子盘面 → 自适应二值化得到笔画 → 归一化到 48x48；
3. 每 15° 试一个朝向，与"同色候选字"的模板做归一化互相关；
4. 取最高分作为识别结果，并记录与次优（不同字）的差距当置信度。

为什么不直接上深度学习：模板匹配零训练数据、可解释、CPU 上就能跑，
先用它把流程打通并拿到基线准确率；等流程顺了再换小 CNN，跟模板法做 A/B 对比。

用法：
    python vision/recognize_chars.py --image vision/data/synthetic/field_oblique.png --mode gt --compare-gt
    python vision/recognize_chars.py --image vision/data/raw/shot.png --detect-json vision/data/output/detect.json
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import config  # noqa: E402
import detect_pieces as dp  # noqa: E402

TPL_SIZE = 48


def load_templates(path=None):
    path = Path(path) if path else config.DIR_TEMPLATE / "templates.npz"
    if not path.exists():
        raise SystemExit("模板不存在，先跑：python vision/char_templates.py")
    z = np.load(path, allow_pickle=False)
    mats = z["templates"]
    names = [str(x) for x in z["names"]]
    meta = [str(x) for x in z["meta"]]
    chars = [m.split("|")[1] for m in meta]
    sides = [m.split("|")[2] for m in meta]
    fonts = [m.split("|")[0] for m in meta]
    return mats, names, chars, sides, fonts


def piece_roi(rect, cx, cy, r, scale=2.0):
    """抠出棋子附近的小图并放大，方便旋转与二值化。"""
    R = int(r * 1.15) + 2
    x0, y0 = max(0, int(cx) - R), max(0, int(cy) - R)
    x1, y1 = min(rect.shape[1], int(cx) + R), min(rect.shape[0], int(cy) + R)
    crop = rect[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return up, (cx - x0) * scale, (cy - y0) * scale, r * scale


def ink_patch(gray, cx, cy, half, border_margin=2, min_area=3):
    """从灰度小图里取中心方块，得到"笔画强度图"（自适应，不依赖固定灰度）。

    为什么要"剔除贴边的连通域"：棋子外圈描边、装饰圆环、盘面外的地面，
    在二值化后都会变成"笔画"，而且它们面积大、都贴着取图边缘，
    结果所有棋子看起来几乎一样，识别率会崩到 50%。字符在盘面中间，不会被误删。

    为什么用"强度图"而不是黑白二值：相机远、字细时，笔画灰度可能只比盘面暗
    30~60 灰阶，硬二值化会把笔画切成一堆碎点，形状信息就丢了。
    保留灰度（比盘面暗多少 → 0~255）能显著提升这种低对比度字的识别率。
    """
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    sub = gray[y0:y1, x0:x1]
    if sub.size < 16:
        return None
    subf = sub.astype(np.float32)
    base = float(np.percentile(subf, 80))
    hard = ((subf < base - config.INK_DELTA).astype(np.uint8)) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hard, 8)
    keep = np.zeros(hard.shape, bool)
    h, w = hard.shape
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area < min_area:
            continue
        if (bx <= border_margin or by <= border_margin
                or bx + bw >= w - border_margin or by + bh >= h - border_margin):
            continue                      # 贴边 → 外圈/装饰环/地面，丢掉
        keep |= (lab == i)
    if keep.sum() < max(20, 0.05 * float((hard > 0).sum())):
        keep = np.ones_like(keep)         # 兜底：万一字太大被删光，就用全部
    out = np.where(keep, hard, 0).astype(np.uint8)
    return out if int(out.sum()) > 0 else None


def recognize(rect, mm_per_px, pieces, tpl, step=None, size=TPL_SIZE, dilate=1):
    mats, _names, chars, sides, fonts = tpl
    step = step or config.ANGLE_STEP
    angles = list(range(0, 360, step))
    out = []
    for p in pieces:
        roi = piece_roi(rect, p["x"] / mm_per_px, p["y"] / mm_per_px, p["r"] / mm_per_px)
        if roi is None:
            continue
        up, ucx, ucy, ur = roi
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        half = max(6, int(ur * 0.70))
        patches = []
        for ang in angles:
            M = cv2.getRotationMatrix2D((ucx, ucy), ang, 1.0)
            rot = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            ink = ink_patch(rot, ucx, ucy, half)
            if ink is None:
                continue
            patch = common.normalize_patch(ink, size)
            if dilate > 0:
                # 相机拍到的笔画通常比字体渲染的细，先膨胀一下再去比，
                # 否则"细字 vs 粗模板"的相关性会明显偏低。
                patch = cv2.dilate(patch, np.ones((3, 3), np.uint8), iterations=dilate)
            patches.append((ang, patch))
        if not patches:
            continue
        Q = np.stack([q for _a, q in patches]).astype(np.float32)
        cand = [i for i, s in enumerate(sides) if s == p["side"]]
        if not cand:
            cand = list(range(len(sides)))
        S = common.ncc_scores(Q, mats[cand])
        flat = np.argsort(S.ravel())[::-1]
        ai, ci = np.unravel_index(flat[0], S.shape)
        best_char = chars[cand[ci]]
        second = 0.0
        for o in flat[1:]:
            a2, c2 = np.unravel_index(o, S.shape)
            if chars[cand[c2]] != best_char:
                second = float(S[a2, c2])
                break
        norm = config.NORMALIZE.get(best_char, best_char)
        out.append({**p,
                    "char_raw": best_char, "char": norm,
                    "value": config.VALUE.get(norm, 0),
                    "score": round(float(S[ai, ci]), 4),
                    "margin": round(float(S[ai, ci]) - second, 4),
                    "angle": patches[ai][0], "font": fonts[cand[ci]]})
    return out


def annotate(rect, pieces, mm_per_px):
    view = rect.copy()
    for p in pieces:
        c = (int(p["x"] / mm_per_px), int(p["y"] / mm_per_px))
        col = (0, 0, 255) if p["side"] == "red" else (60, 60, 60)
        r_px = int(config.PIECE_R / mm_per_px)
        cv2.circle(view, c, r_px, col, 2)
        label = f"{p.get('char', '?')}"
        view = common.put_text_cn(view, label, (c[0] - r_px, c[1] - r_px - 30), 26,
                                  (255, 255, 0) if p["side"] == "red" else (0, 255, 0))
        conf = f"{p.get('score', 0):.2f}"
        view = common.put_text_cn(view, conf, (c[0] - r_px, c[1] + r_px + 4), 18, (0, 200, 255))
    return view


def compare_with_gt(pieces, gt_path, tol=25.0):
    gt = common.load_json(gt_path)
    used = [False] * len(gt)
    hit = char_ok = side_ok = 0
    for p in pieces:
        best, bd = -1, 1e9
        for i, g in enumerate(gt):
            if used[i]:
                continue
            d = float(np.hypot(p["x"] - g["x"], p["y"] - g["y"]))
            if d < bd:
                bd, best = d, i
        if best < 0 or bd > tol:
            continue
        used[best] = True
        hit += 1
        g = gt[best]
        side_ok += (p["side"] == g["side"])
        g_char = config.NORMALIZE.get(g["char"], g["char"])
        char_ok += (p.get("char") == g_char)
    print("-" * 56)
    print(f"匹配上的棋子 {hit} 枚")
    if hit:
        print(f"红黑判断正确率 {side_ok / hit * 100:5.1f}%")
        print(f"字符识别正确率 {char_ok / hit * 100:5.1f}%   （模板匹配基线）")
    return char_ok, hit


def main():
    ap = argparse.ArgumentParser(description="识别棋子上的汉字")
    ap.add_argument("--image", required=True)
    ap.add_argument("--detect-json", default=None, help="不给就现跑一遍检测")
    ap.add_argument("--mode", choices=["calib", "gt"], default="calib")
    ap.add_argument("--frame", default=config.FRAME)
    ap.add_argument("--mm-per-px", type=float, default=config.MM_PER_PX)
    ap.add_argument("--angle-step", type=int, default=config.ANGLE_STEP)
    ap.add_argument("--dilate", type=int, default=1, help="笔画膨胀次数，0=不膨胀")
    ap.add_argument("--name", default="recognize")
    ap.add_argument("--compare-gt", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    img = common.imread_cn(args.image)
    if img is None:
        raise SystemExit(f"读不到图片：{args.image}")
    if args.mode == "gt":
        H, _ = common.load_homography(path=config.DIR_SYN / "gt_homography.json")
    else:
        H, _ = common.load_homography(args.frame)
    if H is None:
        raise SystemExit("没有标定文件，先跑 calib_field.py（或加 --mode gt）")

    rect, mm_per_px = common.rectify(img, H, args.mm_per_px, (config.FIELD_L, config.FIELD_W))
    if args.detect_json and Path(args.detect_json).exists():
        pieces = common.load_json(args.detect_json)["pieces"]
        print(f"从 {args.detect_json} 读取 {len(pieces)} 枚棋子")
    else:
        pieces, used = dp.detect(rect, mm_per_px)
        print(f"现场检测（{used}）得到 {len(pieces)} 枚棋子")

    tpl = load_templates()
    print(f"载入模板 {len(tpl[0])} 个（{len(set(tpl[2]))} 种字 × 多种字体写法）")
    result = recognize(rect, mm_per_px, pieces, tpl, args.angle_step, dilate=args.dilate)
    common.save_json(result, config.DIR_OUTPUT / f"{args.name}.json")

    print(f"识别出 {len(result)} 枚，按分值统计：")
    stat = {}
    for p in result:
        stat[(p["char"], p["side"])] = stat.get((p["char"], p["side"]), 0) + 1
    for (ch, side), n in sorted(stat.items(), key=lambda kv: -config.VALUE.get(kv[0][0], 0)):
        print(f"  {ch}（{side:5s}）× {n:2d}   单枚 {config.VALUE.get(ch, 0):2d} 分")
    red_v = sum(p["value"] for p in result if p["side"] == "red")
    blk_v = sum(p["value"] for p in result if p["side"] == "black")
    print(f"  红方合计 {red_v} 分 / 黑方合计 {blk_v} 分")

    view = annotate(rect, result, mm_per_px)
    common.imwrite_cn(config.DIR_OUTPUT / f"{args.name}_annotated.png", view)
    if args.compare_gt:
        compare_with_gt(result, config.DIR_SYN / "gt_pieces.json")
    if args.show:
        common.show_or_save("recognize", view)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
