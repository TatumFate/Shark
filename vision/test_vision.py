# -*- coding: utf-8 -*-
"""一键回归测试：改完视觉代码跑这个，看有没有变差。

用法：
    python vision/test_vision.py
    python vision/test_vision.py --fps-only      # 只测速度

打印四项指标并与基线比较；任何一项低于基线就返回非 0（以后接 CI 也方便）：
    查全率 ≥ 95%   /  定位误差 ≤ 3mm  /  字符识别率 ≥ 70%  /  实时帧率 ≥ 20 FPS
（实拍棋子样本录入之后，字符识别率的基线应该往上调，见 config/README）
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
import realtime as rt  # noqa: E402

BASE = {"recall": 95.0, "pos_err_mm": 3.0, "char_acc": 70.0, "fps": 20.0}


def main():
    # Windows 的终端默认是 GBK，emoji 会直接抛 UnicodeEncodeError，这里兜底
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                          # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps-only", action="store_true")
    args = ap.parse_args()

    img_path = config.DIR_SYN / "field_oblique.png"
    gt_path = config.DIR_SYN / "gt_pieces.json"
    if not img_path.exists() or not gt_path.exists():
        raise SystemExit("合成测试图不存在，先跑：python vision/make_synthetic.py")

    img = common.imread_cn(img_path)
    gt = common.load_json(gt_path)
    H, _ = common.load_homography(path=config.DIR_SYN / "gt_homography.json")
    rect, mmpp = common.rectify(img, H, config.MM_PER_PX, (config.FIELD_L, config.FIELD_W))

    t = time.perf_counter()
    pieces, used = dp.detect(rect, mmpp)
    t_det = (time.perf_counter() - t) * 1000

    # 定位：与真值最近邻匹配
    left = [False] * len(gt)
    errs = []
    for p in pieces:
        bi, bd = -1, 1e9
        for i, g in enumerate(gt):
            if left[i]:
                continue
            d = float(np.hypot(p["x"] - g["x"], p["y"] - g["y"]))
            if d < bd:
                bd, bi = d, i
        if bi >= 0 and bd <= 25:
            left[bi] = True
            errs.append(bd)
    recall = len(errs) / len(gt) * 100
    pos_err = float(np.mean(errs)) if errs else 99.0

    print(f"[检测] {used}  {len(pieces)} 枚  查全 {recall:.1f}%  "
          f"定位误差 {pos_err:.1f}mm  耗时 {t_det:.1f}ms")

    if not args.fps_only:
        t = time.perf_counter()
        res = rc.recognize(rect, mmpp, pieces, rc.load_templates())
        t_rec = (time.perf_counter() - t) * 1000
        left = [False] * len(gt)
        ok = hit = 0
        for p in res:
            bi, bd = -1, 1e9
            for i, g in enumerate(gt):
                if left[i]:
                    continue
                d = float(np.hypot(p["x"] - g["x"], p["y"] - g["y"]))
                if d < bd:
                    bd, bi = d, i
            if bi >= 0 and bd <= 25:
                left[bi] = True
                hit += 1
                want = config.NORMALIZE.get(gt[bi]["char"], gt[bi]["char"])
                ok += (p["char"] == want)
        char_acc = ok / max(hit, 1) * 100
        print(f"[识别] 字符准确率 {char_acc:.1f}%（{ok}/{hit}）  耗时 {t_rec:.1f}ms")
    else:
        char_acc = 100.0

    # 实时帧率：视窗塞满 32 枚的最坏情况
    eng = rt.RealtimeVision(H, rc.load_templates(), mm_per_px=1.5, roi_mm=800.0,
                            center_mm=(config.PILE_CX, config.PILE_CY))
    times = []
    for _ in range(25):
        t = time.perf_counter()
        eng.process(img)
        times.append(time.perf_counter() - t)
    fps = 1000.0 / (float(np.mean(times)) * 1000.0)
    print(f"[实时] {np.mean(times) * 1000:.1f} ms/帧 → {fps:.1f} FPS"
          f"（检测 {eng.stats['detect']:.1f}ms / 识别 {eng.stats['rec']:.1f}ms）")

    fails = []
    if recall < BASE["recall"]:
        fails.append(f"查全率 {recall:.1f}% < {BASE['recall']}%")
    if pos_err > BASE["pos_err_mm"]:
        fails.append(f"定位误差 {pos_err:.1f}mm > {BASE['pos_err_mm']}mm")
    if char_acc < BASE["char_acc"]:
        fails.append(f"字符识别 {char_acc:.1f}% < {BASE['char_acc']}%")
    if fps < BASE["fps"]:
        fails.append(f"帧率 {fps:.1f} < {BASE['fps']} FPS")

    print("-" * 62)
    if fails:
        print("[FAIL] 有指标低于基线：")
        for f in fails:
            print("   ·", f)
        return 1
    print("[OK] 全部达标（查全 >=95% / 定位 <=3mm / 字符 >=70% / 帧率 >=20FPS）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
