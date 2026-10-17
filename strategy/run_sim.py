# -*- coding: utf-8 -*-
"""跑决策层仿真：没有底盘、没有真棋子也能验证"选目标→推→投放"整条链。

用法：
    python strategy/run_sim.py                      # 跑 3 局，打印得分/用时/卡住次数
    python strategy/run_sim.py --episodes 10        # 多跑几局看稳定性
    python strategy/run_sim.py --video              # 同时录一段比赛仿真视频
    python strategy/run_sim.py --seconds 60 --verbose
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vision"))
import common  # noqa: E402
import config  # noqa: E402
import game  # noqa: E402
from sim import run_episode  # noqa: E402

SCALE = 0.3                      # 像素 / 毫米


def _px(x, y):
    return int(round(x * SCALE)) + 20, int(round(y * SCALE)) + 20


class Renderer:
    def __init__(self, out_path, fps=10):
        self.w = int(config.FIELD_L * SCALE) + 40
        self.h = int(config.FIELD_W * SCALE) + 40
        self.out = None
        if out_path:
            path = Path(out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                       float(fps), (self.w, self.h))
            print(f"[录制] {path}")

    def __call__(self, world, car, fsm, t):
        img = np.full((self.h, self.w, 3), 28, np.uint8)
        cv2.rectangle(img, _px(0, 0), _px(config.FIELD_L, config.FIELD_W), (190, 190, 190), 2)
        # 己方球门区（左，红）/ 对方（右，蓝）
        for box, col in ((game.own_goal_box(), (0, 0, 220)), (game.rival_goal_box(), (220, 120, 0))):
            x0, x1, y0, y1 = box
            cv2.rectangle(img, _px(x0, y0), _px(x1, y1), col, -1)
        # 堆放区虚线
        p0 = _px(config.PILE_CX - config.PILE_L / 2, config.PILE_CY - config.PILE_W / 2)
        p1 = _px(config.PILE_CX + config.PILE_L / 2, config.PILE_CY + config.PILE_W / 2)
        cv2.rectangle(img, p0, p1, (110, 150, 110), 1)
        # 棋子
        for p in world.pieces:
            if p["scored"]:
                continue
            c = (0, 0, 230) if p["side"] == "red" else (230, 230, 230)
            cv2.circle(img, _px(p["x"], p["y"]), max(3, int(p["r"] * SCALE)), c, -1)
            cv2.circle(img, _px(p["x"], p["y"]), max(3, int(p["r"] * SCALE)), (60, 60, 60), 1)
        # 车 + 推板
        cx, cy = _px(car.x, car.y)
        th = np.radians(car.h)
        f = np.array([np.cos(th), np.sin(th)])
        l = np.array([-f[1], f[0]])
        tri = np.array([_px(car.x + f[0] * 90, car.y + f[1] * 90),
                        _px(car.x - f[0] * 60 + l[0] * 55, car.y - f[1] * 60 + l[1] * 55),
                        _px(car.x - f[0] * 60 - l[0] * 55, car.y - f[1] * 60 - l[1] * 55)])
        cv2.fillPoly(img, [tri.astype(np.int32)], (60, 200, 60))
        if car.blade_down:
            hw = config.BLADE_WIDTH_MM / 2
            b0 = (car.x + f[0] * config.BLADE_AHEAD_MM + l[0] * hw,
                  car.y + f[1] * config.BLADE_AHEAD_MM + l[1] * hw)
            b1 = (car.x + f[0] * config.BLADE_AHEAD_MM - l[0] * hw,
                  car.y + f[1] * config.BLADE_AHEAD_MM - l[1] * hw)
            cv2.line(img, _px(*b0), _px(*b1), (60, 220, 255), 4)
        # 文案
        st = fsm.status(t)
        items = [
            (f"{st['state']}   得分 {world.score}   剩余 {st['remain']:.0f}s   "
             f"趟数 {st['trips']}   脱困 {st['recoveries']}", (8, 4), 18, (0, 255, 255)),
            (f"目标 {st['target'] or '-'}   指令 {st['cmd']}", (8, 26), 16, (0, 255, 0)),
        ]
        if self.out is not None:
            img = common.put_texts_cn(img, items)
            self.out.write(img)
        else:
            cv2.imshow("sim", common.put_texts_cn(img, items))
            cv2.waitKey(1)

    def close(self):
        if self.out is not None:
            self.out.release()
        else:
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="决策层仿真")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=None, help="单局时长（默认 config.MATCH_SECONDS）")
    ap.add_argument("--video", action="store_true", help="录一段仿真视频到 data/output/sim.mp4")
    ap.add_argument("--noise", type=float, default=3.0, help="视觉位置噪声(mm)")
    ap.add_argument("--dropout", type=float, default=0.05, help="视觉漏检率")
    ap.add_argument("--char-err", type=float, default=0.08, help="字符误判率")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rows = []
    t0 = time.perf_counter()
    for i in range(args.episodes):
        renderer = Renderer(args.out or (str(config.DIR_OUTPUT / "sim.mp4") if args.video and i == 0 else None))
        r = run_episode(seed=1000 + i, seconds=args.seconds, render=renderer,
                        vision={"noise": args.noise, "dropout": args.dropout,
                                "char_err": args.char_err})
        renderer.close()
        rows.append(r)
        print(f"第 {i + 1} 局：得分 {r['score']:3d}（{r['scored_n']:2d} 枚）  "
              f"趟数 {r['trips']:2d}  脱困 {r['recoveries']:2d}  "
              f"结束状态 {r['state']}  分/秒 {r['rate']:.2f}  剩余未进区 {r['left']}")
        if args.verbose:
            print("   状态轨迹：", " → ".join(f"{h[1]}" for h in r["history"][:14]), "…")

    scores = [r["score"] for r in rows]
    print("-" * 66)
    print(f"{len(rows)} 局：平均 {np.mean(scores):.1f} 分，最高 {max(scores)}，最低 {min(scores)}；"
          f"平均趟数 {np.mean([r['trips'] for r in rows]):.1f}；"
          f"平均脱困 {np.mean([r['recoveries'] for r in rows]):.1f}")
    print(f"仿真耗时 {time.perf_counter() - t0:.1f}s（单局 {args.seconds or config.MATCH_SECONDS:.0f}s 比赛时长）")
    if args.video:
        print(f"视频：{args.out or config.DIR_OUTPUT / 'sim.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
