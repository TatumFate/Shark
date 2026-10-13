# -*- coding: utf-8 -*-
"""实时识别：相机/视频帧 → 校正 → 检测 → 识别 → 输出（给上层决策或串口）。

和 pipeline.py 用的是**同一套算子**（common / detect_pieces / recognize_chars），
区别只在三处工程改造，为了把 1.3 Hz 拉到 10 Hz 以上：

1. 只校正"车前一块视窗"，不是整块场地  → 校正本身 1~3ms，几乎不花时间
2. 逐枚识别改成"带预算"：每帧最多认 N 枚，其余的沿用缓存结果
   （棋盘上的棋子又不会变，没必要每帧重认 32 枚）
3. 结果走回调/JSON 输出，方便直接接串口发给 STM32

用法：
    python vision/realtime.py --source 0                                  # 0 号相机
    python vision/realtime.py --source 0 --center 500 0 --roi 800         # 车体坐标系：车前 0.8m 视窗
    python vision/realtime.py --source vision/data/raw/demo.mp4           # 回放录像调参
    python vision/realtime.py --source vision/data/synthetic/field_oblique.png \
        --mode gt --center 1500 1000 --bench 30                           # 没相机时压测

按键：q 退出 / s 存图 / r 录像 / c 清缓存 / d 切换校正图 / 空格 暂停
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import config  # noqa: E402
import detect_pieces as dp  # noqa: E402
import recognize_chars as rc  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
REC_MIN_SCORE = 0.5          # 低于这个置信度就安排重认
REC_REDO_SEC = 4.0           # 同一枚棋子超过这么久没重认过，就再认一次


# ------------------------------------------------------------------ 取帧
class CameraSource:
    def __init__(self, index, width, height):
        self.cap = None
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, 0):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.cap = cap
                    break
            cap.release()
        if self.cap is None:
            raise SystemExit(f"打不开 {index} 号相机，先跑 check_env.py 看看哪号能用")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class VideoSource:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"打不开视频：{path}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:                      # 放完自动循环
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class ImageSource:
    """一张静止图片当视频源：压测/演示用，不占相机。"""
    def __init__(self, path):
        self.img = common.imread_cn(path)
        if self.img is None:
            raise SystemExit(f"读不到图片：{path}")

    def read(self):
        return self.img

    def release(self):
        pass


def open_source(src, width, height):
    p = Path(str(src))
    if p.exists() and p.suffix.lower() in IMG_EXT:
        print(f"[源] 静止图片 {p.name}（循环当视频用，方便压测）")
        return ImageSource(p)
    if p.exists():
        print(f"[源] 视频 {p.name}")
        return VideoSource(p)
    print(f"[源] 相机 {src}")
    return CameraSource(int(src), width, height)


# ------------------------------------------------------------------ 实时识别核心
def rectify_window(img, H, center_mm, size_mm, mm_per_px):
    """只校正"以 center_mm 为中心、边长 size_mm"的一块视窗。

    返回 (校正图, 视窗左上角在坐标系里的毫米坐标)，后者用于把局部坐标换算成绝对坐标。
    """
    ox = center_mm[0] - size_mm[0] / 2.0
    oy = center_mm[1] - size_mm[1] / 2.0
    out_w = int(round(size_mm[0] / mm_per_px))
    out_h = int(round(size_mm[1] / mm_per_px))
    S = np.array([[1.0 / mm_per_px, 0, -ox / mm_per_px],
                  [0, 1.0 / mm_per_px, -oy / mm_per_px],
                  [0, 0, 1]], np.float64)
    H_out = S @ np.asarray(H, np.float64)
    return cv2.warpPerspective(img, H_out, (out_w, out_h)), (ox, oy)


class RealtimeVision:
    def __init__(self, H, tpl, mm_per_px=1.5, roi_mm=800.0, center_mm=(500.0, 0.0),
                 max_rec_per_frame=3, angle_step=None, cache_radius=None):
        self.H = H
        self.tpl = tpl
        self.mm_per_px = mm_per_px
        self.roi_mm = roi_mm
        self.center = center_mm
        self.max_rec = max_rec_per_frame
        self.angle_step = angle_step
        # 缓存匹配半径：一定要小于"两枚棋子相贴时的中心距"（=棋子直径），
        # 否则相邻棋子会被合并到同一个缓存条目，导致有的棋子永远认不上。
        self.cache_radius = float(cache_radius or config.PIECE_DIA * 0.8)
        self.cache = []          # [{x, y, side, char, value, score, t_rec, t_seen}]
        self.stats = {"rect": 0.0, "detect": 0.0, "rec": 0.0, "total": 0.0}

    # ---- 缓存：棋子不动，认过一次就记着 ----
    def match_cache(self, x, y, side, now, claimed=None):
        best, bd = None, 1e9
        for c in self.cache:
            if c["side"] != side:
                continue
            if claimed is not None and id(c) in claimed:
                continue                      # 这一帧已经被别的棋子占用了
            d = float(np.hypot(c["x"] - x, c["y"] - y))
            if d < bd:
                bd, best = d, c
        return best if (best is not None and bd <= self.cache_radius) else None

    def clear_cache(self):
        self.cache = []

    def forget_stale(self, now, keep=1.0):
        self.cache = [c for c in self.cache if now - c["t_seen"] <= keep]

    def process(self, frame):
        t0 = time.perf_counter()
        now = time.time()

        # 1) 只校正车前视窗
        rect, (ox, oy) = rectify_window(frame, self.H, self.center,
                                        (self.roi_mm, self.roi_mm), self.mm_per_px)
        t1 = time.perf_counter()

        # 2) 检测（局部坐标）
        pieces, _used = dp.detect(rect, self.mm_per_px)
        t2 = time.perf_counter()

        # 3) 决定这一帧要"新认"哪几枚
        #    优先级：没认过的 > 认过但置信度低（限制重试次数） > 太旧需要刷新
        #    不加限制的话，几枚"怎么认都认不准"的棋子会每帧霸占预算，
        #    其他棋子永远排不上队。
        need, prio, claimed = [], [], set()
        matched = {}
        for p in pieces:
            ax, ay = p["x"] + ox, p["y"] + oy
            c = self.match_cache(ax, ay, p["side"], now, claimed)
            if c is not None:
                claimed.add(id(c))
            matched[(round(p["x"], 1), round(p["y"], 1))] = c
        for p in pieces:
            ax, ay = p["x"] + ox, p["y"] + oy
            c = matched[(round(p["x"], 1), round(p["y"], 1))]
            tries = (c or {}).get("tries", 0)
            if c is None or c.get("char") is None:
                need.append(p)
                prio.append(0)
            elif c["score"] < REC_MIN_SCORE and tries < 3:
                need.append(p)
                prio.append(1)
            elif now - c["t_rec"] > REC_REDO_SEC and tries < 6:
                need.append(p)
                prio.append(2)
        c_local = (self.roi_mm / 2.0, self.roi_mm / 2.0)
        order = sorted(range(len(need)),
                       key=lambda i: (prio[i],
                                      (need[i]["x"] - c_local[0]) ** 2
                                      + (need[i]["y"] - c_local[1]) ** 2))
        todo = [need[i] for i in order[: self.max_rec]]

        # 4) 只识别这几枚
        results = {}
        if todo:
            for r in rc.recognize(rect, self.mm_per_px, todo, self.tpl, self.angle_step):
                results[(round(r["x"], 1), round(r["y"], 1))] = r
        t3 = time.perf_counter()

        # 5) 合并到缓存，并把结果换算成绝对坐标
        out, fresh = [], 0
        for p in pieces:
            ax, ay = p["x"] + ox, p["y"] + oy
            r = results.get((round(p["x"], 1), round(p["y"], 1)))
            c = matched[(round(p["x"], 1), round(p["y"], 1))]
            if r is not None:
                fresh += 1
                if c is None:
                    c = {"x": ax, "y": ay, "side": p["side"], "tries": 0}
                    self.cache.append(c)
                    matched[(round(p["x"], 1), round(p["y"], 1))] = c
                c.update({"x": ax, "y": ay, "char": r["char"], "value": r["value"],
                          "score": r["score"], "margin": r["margin"],
                          "t_rec": now, "t_seen": now})
                c["tries"] = c.get("tries", 0) + 1
            elif c is not None:
                c["t_seen"] = now
            out.append({
                "x": round(ax, 1), "y": round(ay, 1), "r": p["r"], "side": p["side"],
                "char": (c or {}).get("char"), "value": (c or {}).get("value", 0),
                "score": (c or {}).get("score", 0.0), "fresh": r is not None,
            })
        self.forget_stale(now)

        t4 = time.perf_counter()
        self.stats["rect"] = (t1 - t0) * 1000
        self.stats["detect"] = (t2 - t1) * 1000
        self.stats["rec"] = (t3 - t2) * 1000
        self.stats["total"] = (t4 - t0) * 1000
        return out, rect, (ox, oy), fresh


# ------------------------------------------------------------------ 叠加显示
def draw_overlay(view, pieces, roi_mm, mm_per_px, fps, stats, paused):
    items = []
    n_known = sum(1 for p in pieces if p["char"])
    red_v = sum(p["value"] for p in pieces if p["side"] == "red" and p["char"])
    blk_v = sum(p["value"] for p in pieces if p["side"] == "black" and p["char"])
    head = (f"{fps:5.1f} FPS | 一帧 {stats['total']:5.1f} ms "
            f"(校正 {stats['rect']:.1f} / 检测 {stats['detect']:.1f} / 识别 {stats['rec']:.1f})")
    items.append((head, (8, 4), 22, (0, 255, 255)))
    items.append((f"棋子 {len(pieces)} 枚（已识别 {n_known}）  红方 {red_v} 分 / 黑方 {blk_v} 分"
                  + ("   [暂停]" if paused else ""), (8, 32), 22, (0, 255, 0)))
    for p in pieces:
        cx, cy = int(p["x"] / mm_per_px), int(p["y"] / mm_per_px)
        is_red = p["side"] == "red"
        col = (0, 0, 255) if is_red else (255, 255, 255)
        cv2.circle(view, (cx, cy), int(config.PIECE_R / mm_per_px), col, 2)
        txt = p["char"] or "?"
        tcol = (0, 0, 255) if is_red else (0, 255, 0)
        if not p["fresh"]:
            tcol = tuple(int(c * 0.6) for c in tcol)      # 缓存结果画暗一点，一眼能区分
        items.append((txt, (cx - 12, cy - 42), 26, tcol))
        if p["char"]:
            items.append((f"{p['score']:.2f}", (cx - 14, cy + 18), 16, (0, 200, 255)))
    return common.put_texts_cn(view, items)


def top_targets(pieces, k=3):
    """给上层决策用的候选目标：已识别的、离车头越近越靠前。"""
    known = [p for p in pieces if p["char"]]
    known.sort(key=lambda p: (p["x"] ** 2 + p["y"] ** 2, -p["value"]))
    return [{"char": p["char"], "side": p["side"], "value": p["value"],
             "x": p["x"], "y": p["y"], "score": p["score"]} for p in known[:k]]


def main():
    ap = argparse.ArgumentParser(description="实时棋子识别")
    ap.add_argument("--source", default="0", help="相机编号 / 视频文件 / 图片文件")
    ap.add_argument("--mode", choices=["calib", "gt"], default="calib")
    ap.add_argument("--frame-name", default=config.FRAME, help="用哪份标定（field/robot）")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--mm-per-px", type=float, default=1.5)
    ap.add_argument("--roi", type=float, default=800.0, help="车前视窗边长(mm)")
    ap.add_argument("--center", type=float, nargs=2, default=(500.0, 0.0),
                    metavar=("X", "Y"), help="视窗中心在标定坐标系里的位置(mm)")
    ap.add_argument("--max-rec-per-frame", type=int, default=3, help="每帧最多新认几枚")
    ap.add_argument("--angle-step", type=int, default=None)
    ap.add_argument("--emit-hz", type=float, default=5.0, help="向外输出结果的频率(Hz)，0=不输出")
    ap.add_argument("--bench", type=int, default=0, help="压测：跑 N 帧不显示，只报帧率")
    ap.add_argument("--frames", type=int, default=0, help="跑够 N 帧就退出（0=一直跑）")
    ap.add_argument("--save-every", type=int, default=0, help="每 N 帧存一张标注图（0=不存）")
    ap.add_argument("--show", action="store_true", help="强制开窗显示")
    args = ap.parse_args()

    if args.mode == "gt":
        H, _ = common.load_homography(path=config.DIR_SYN / "gt_homography.json")
    else:
        H, _ = common.load_homography(args.frame_name)
    if H is None:
        raise SystemExit("没有标定文件，先跑 vision/calib_field.py")

    src = open_source(args.source, args.width, args.height)
    tpl = rc.load_templates()
    engine = RealtimeVision(H, tpl, args.mm_per_px, args.roi, tuple(args.center),
                            args.max_rec_per_frame, args.angle_step)
    print(f"视窗 {args.roi:.0f}mm @ {args.mm_per_px} mm/px，中心 {tuple(args.center)}，"
          f"每帧最多新认 {args.max_rec_per_frame} 枚")

    # ---- 压测模式：不显示，纯算帧率 ----
    if args.bench:
        times = []
        for _ in range(args.bench):
            frame = src.read()
            if frame is None:
                break
            t = time.perf_counter()
            pieces, _rect, _o, fresh = engine.process(frame)
            times.append(time.perf_counter() - t)
        times = np.array(times) * 1000
        src.release()
        print(f"压测 {len(times)} 帧：平均 {times.mean():.1f} ms，最快 {times.min():.1f} ms，"
              f"最慢 {times.max():.1f} ms  → 平均 {1000 / times.mean():.1f} FPS")
        print(f"分阶段（最后一次）：校正 {engine.stats['rect']:.1f} / 检测 "
              f"{engine.stats['detect']:.1f} / 识别 {engine.stats['rec']:.1f} ms")
        return 0

    win = "realtime (q=quit s=save r=record c=clear d=rect)"
    headless = False
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except cv2.error:
        headless = True
        print("[提示] 当前环境没有图形界面，改用无窗口模式："
              "每 N 帧存一张标注图到 data/output/，Ctrl+C 退出")
        if not args.save_every:
            args.save_every = 20
    writer = None
    frames, t_fps, paused, show_rect, last_emit = 0, time.time(), False, False, 0.0
    frame = src.read()
    try:
      while True:
        if not paused:
            frame = src.read()
            if frame is None:
                break
            pieces, rect, origin, fresh = engine.process(frame)
            frames += 1
        fps = frames / max(time.time() - t_fps, 1e-6)
        view = rect if show_rect else frame
        if show_rect:
            items = [(f"{fps:5.1f} FPS | 校正图 {args.mm_per_px} mm/px | 视窗原点 "
                      f"({origin[0]:.0f}, {origin[1]:.0f}) mm", (8, 4), 22, (0, 255, 255))]
            view = common.put_texts_cn(view.copy(), items)
        else:
            view = draw_overlay(view.copy(), pieces, args.roi, args.mm_per_px,
                                fps, engine.stats, paused)
        if not headless:
            cv2.imshow(win, view)
        else:
            time.sleep(0.001)

        # 向外输出（真实车上这里换成串口 / CAN 发送）
        if args.emit_hz > 0 and time.time() - last_emit > 1.0 / args.emit_hz:
            last_emit = time.time()
            print(json.dumps({"n": len(pieces), "fresh": fresh, "cached": len(engine.cache),
                              "ms": round(engine.stats["total"], 1),
                              "targets": top_targets(pieces)}, ensure_ascii=False))
        if args.save_every and frames % args.save_every == 0:
            common.imwrite_cn(config.DIR_OUTPUT / f"live_{frames:05d}.png", view)

        key = (cv2.waitKey(1) & 0xFF) if not headless else 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            common.imwrite_cn(config.DIR_OUTPUT / f"live_{time.strftime('%H%M%S')}.png", view)
        if key == ord("c"):
            engine.clear_cache()
            print("[清空识别缓存]")
        if key == ord("d"):
            show_rect = not show_rect
        if key == ord(" "):
            paused = not paused
        if key == ord("r"):
            if writer is None:
                path = config.DIR_RAW / f"live_{time.strftime('%H%M%S')}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0,
                                         (view.shape[1], view.shape[0]))
                print("[录像开始]", path)
            else:
                writer.release()
                writer = None
                print("[录像结束]")
        if writer is not None:
            writer.write(view)
        if args.frames and frames >= args.frames:
            print(f"[跑满 {args.frames} 帧，退出]")
            break
    except KeyboardInterrupt:
        print("\n[中断退出]")

    if writer is not None:
        writer.release()
    src.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
