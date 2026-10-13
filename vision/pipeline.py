# -*- coding: utf-8 -*-
"""端到端跑一遍：图 → 校正 → 检测 → 识别 → 结果图 + JSON + 统计。

这是每天调试用的主入口，也是后面接到车上的那个函数：
车上只要把"读一张图"换成"读相机帧"，逻辑完全一样。

用法：
    # 合成图（不需要相机）
    python vision/pipeline.py --image vision/data/synthetic/field_oblique.png --mode gt --compare-gt

    # 真实照片
    python vision/pipeline.py --image vision/data/raw/shot.png --mode calib
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import config  # noqa: E402
import detect_pieces as dp  # noqa: E402
import recognize_chars as rc  # noqa: E402


def in_goal_zone(x, y):
    """判断某个点是否落在"甲队球门区"里（站位判断用）。"""
    return (x <= config.GOAL_DEPTH) and (config.GOAL_Y0 <= y <= config.GOAL_Y1)


def summarize(pieces):
    """打印统计 + 给出第一批该抢哪几枚的建议。"""
    red = [p for p in pieces if p["side"] == "red"]
    black = [p for p in pieces if p["side"] == "black"]
    print("=" * 64)
    print(f"检出 {len(pieces)} 枚：红 {len(red)} / 黑 {len(black)}")
    print(f"分值合计：红 {sum(p['value'] for p in red)} 分 / "
          f"黑 {sum(p['value'] for p in black)} 分")

    stat = {}
    for p in pieces:
        stat[(p["side"], p["char"])] = stat.get((p["side"], p["char"]), 0) + 1
    print("明细：", "  ".join(f"{s[:1]}{c}×{n}" for (s, c), n in
                              sorted(stat.items(), key=lambda kv: -config.VALUE.get(kv[0][1], 0))))

    print("-" * 64)
    print("优先目标（按分值排序，距离是到己方球门区中心的直线距离）：")
    gx, gy = config.GOAL_DEPTH / 2, (config.GOAL_Y0 + config.GOAL_Y1) / 2
    top = sorted(pieces, key=lambda p: -p["value"])[:6]
    for p in top:
        d = float(np.hypot(p["x"] - gx, p["y"] - gy))
        flag = "  ← 已经在球门区里！" if in_goal_zone(p["x"], p["y"]) else ""
        print(f"   {p['char']}（{p['side']}，{p['value']:2d} 分）"
              f"  位置 ({p['x']:6.1f}, {p['y']:6.1f})  距球门 {d / 1000:.2f} m  "
              f"识别置信度 {p['score']:.2f}{flag}")
    return red, black


def main():
    ap = argparse.ArgumentParser(description="视觉端到端流程")
    ap.add_argument("--image", required=True)
    ap.add_argument("--mode", choices=["calib", "gt"], default="calib")
    ap.add_argument("--frame", default=config.FRAME)
    ap.add_argument("--mm-per-px", type=float, default=config.MM_PER_PX)
    ap.add_argument("--method", choices=["auto", "hough", "contour"], default="auto")
    ap.add_argument("--angle-step", type=int, default=config.ANGLE_STEP)
    ap.add_argument("--name", default="pipeline")
    ap.add_argument("--compare-gt", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    img = common.imread_cn(args.image)
    if img is None:
        raise SystemExit(f"读不到图片：{args.image}")
    if args.mode == "gt":
        H, _ = common.load_homography(path=config.DIR_SYN / "gt_homography.json")
    else:
        H, _ = common.load_homography(args.frame)
    if H is None:
        raise SystemExit("没有标定文件，先跑 vision/calib_field.py（或加 --mode gt）")
    t_load = time.perf_counter()

    rect, mm_per_px = common.rectify(img, H, args.mm_per_px, (config.FIELD_L, config.FIELD_W))
    pieces, used = dp.detect(rect, mm_per_px, args.method)
    t_det = time.perf_counter()
    print(f"检测：{used} 方法，{len(pieces)} 枚（{(t_det - t_load) * 1000:.0f} ms）")

    tpl = rc.load_templates()
    pieces = rc.recognize(rect, mm_per_px, pieces, tpl, args.angle_step)
    t_rec = time.perf_counter()
    print(f"识别：{len(pieces)} 枚（{(t_rec - t_det) * 1000:.0f} ms，"
          f"平均 {(t_rec - t_det) / max(len(pieces), 1) * 1000:.0f} ms/枚）")

    summarize(pieces)
    common.save_json(pieces, config.DIR_OUTPUT / f"{args.name}.json")
    view = rc.annotate(rect, pieces, mm_per_px)
    common.imwrite_cn(config.DIR_OUTPUT / f"{args.name}_annotated.png", view)

    if args.compare_gt:
        dp.compare_with_gt(pieces, config.DIR_SYN / "gt_pieces.json")
        rc.compare_with_gt(pieces, config.DIR_SYN / "gt_pieces.json")
    print(f"总耗时 {(t_rec - t0) * 1000:.0f} ms（不含显示）")
    if args.show:
        common.show_or_save("pipeline", view)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
