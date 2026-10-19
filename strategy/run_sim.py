# -*- coding: utf-8 -*-
"""跑双队仿真并录视频：红队 vs 黑队，棋子是真棋子图形（木盘 + 汉字）。

用法：
    python strategy/run_sim.py                          # 3 局，打印比分
    python strategy/run_sim.py --video                  # 录到 vision/data/output/sim.mp4
    python strategy/run_sim.py --seconds 60 --seed 7
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vision"))
import config  # noqa: E402
import game  # noqa: E402
from sim import run_episode  # noqa: E402

FONT = r"C:\Windows\Fonts\msyh.ttc"
S = 0.72                      # 像素 / 毫米
PAD_X, PAD_TOP, HUD = 64, 182, 150
W = int(config.FIELD_L * S) + 2 * PAD_X
H = int(config.FIELD_W * S) + PAD_TOP + 46
TEAM_RGB = {"red": (200, 55, 55), "black": (45, 45, 48)}


def px(x, y):
    return (PAD_X + x * S, PAD_TOP + y * S)


def font(size):
    return ImageFont.truetype(FONT, size)


def piece_tile(char, side, dia_px):
    """画一枚棋子（木盘 + 汉字），返回 RGBA 图。"""
    big = dia_px * 4
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c, r = big // 2, big // 2 - 2
    d.ellipse([c - r, c - r, c + r, c + r], fill=(226, 212, 190),
              outline=(118, 92, 66), width=max(2, big // 26))
    d.ellipse([c - r * 0.86, c - r * 0.86, c + r * 0.86, c + r * 0.86],
              outline=(170, 145, 110), width=max(1, big // 60))
    f = ImageFont.truetype(r"C:\Windows\Fonts\simkai.ttf", int(big * 0.58))
    ink = (196, 34, 40) if side == "red" else (30, 30, 32)
    bb = d.textbbox((0, 0), char, font=f)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((big - w) / 2 - bb[0], (big - h) / 2 - bb[1]), char, font=f, fill=ink)
    return img.resize((dia_px, dia_px), Image.LANCZOS)


class Renderer:
    def __init__(self, out_path=None, fps=10, seconds=None, title=None):
        self.out = None
        if out_path:
            path = Path(out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                       float(fps), (W, H))
            print(f"[录制] {path}")
        self.dia = int(round(config.PIECE_DIA * S))
        self.tiles = {}
        self.seconds = seconds
        self.frames = 0
        self.title = title

    def tile(self, char, side):
        key = (char, side)
        if key not in self.tiles:
            self.tiles[key] = piece_tile(char, side, self.dia)
        return self.tiles[key]

    def __call__(self, world, cars, fsms, t):
        img = Image.new("RGB", (W, H), (247, 246, 242))
        d = ImageDraw.Draw(img)
        # 场地
        x0, y0 = px(0, 0)
        x1, y1 = px(config.FIELD_L, config.FIELD_W)
        d.rectangle([x0, y0, x1, y1], fill=(232, 232, 226), outline=(120, 120, 120), width=2)
        # 挡板（80mm 厚的深色边框）
        wall = int(80 * S)
        d.rectangle([x0 - wall, y0 - wall, x1 + wall, y1 + wall], outline=(70, 70, 72), width=wall)
        d.rectangle([x0, y0, x1, y1], fill=(232, 232, 226), outline=(120, 120, 120), width=2)
        # 球门区
        for side, col, txt in (("red", (232, 208, 206), "红队球门"), ("black", (210, 214, 224), "黑队球门")):
            bx0, bx1, by0, by1 = game.own_goal_box(side)
            d.rectangle([*px(bx0, by0), *px(bx1, by1)], fill=col, outline=(150, 150, 150), width=1)
            d.text(px((bx0 + bx1) / 2 - 55, by0 + 20), txt, font=font(15), fill=(90, 60, 60))
        # 堆放区
        d.rectangle([*px(config.PILE_CX - config.PILE_L / 2, config.PILE_CY - config.PILE_W / 2),
                     *px(config.PILE_CX + config.PILE_L / 2, config.PILE_CY + config.PILE_W / 2)],
                    outline=(150, 170, 150), width=1)
        # 棋子（真棋子图形）
        for p in world.pieces:
            cx, cy = px(p["x"], p["y"])
            tile = self.tile(p["char"], p["side"])
            img.paste(tile, (int(cx - tile.width / 2), int(cy - tile.height / 2)), tile)
        # 车 + 推板
        for side, car in cars.items():
            col = TEAM_RGB[side]
            th = np.radians(car.h)
            f = np.array([np.cos(th), np.sin(th)])
            l = np.array([-f[1], f[0]])
            pts = [px(car.x + f[0] * 120, car.y + f[1] * 120),
                   px(car.x - f[0] * 80 + l[0] * 95, car.y - f[1] * 80 + l[1] * 95),
                   px(car.x - f[0] * 80 - l[0] * 95, car.y - f[1] * 80 - l[1] * 95)]
            d.polygon(pts, fill=col, outline=(255, 255, 255), width=2)
            if car.blade_down:
                hw = config.BLADE_WIDTH_MM / 2
                b0 = px(car.x + f[0] * config.BLADE_AHEAD_MM + l[0] * hw,
                        car.y + f[1] * config.BLADE_AHEAD_MM + l[1] * hw)
                b1 = px(car.x + f[0] * config.BLADE_AHEAD_MM - l[0] * hw,
                        car.y + f[1] * config.BLADE_AHEAD_MM - l[1] * hw)
                d.line([b0, b1], fill=col, width=5)
            d.text((px(car.x, car.y)[0] - 18, px(car.x, car.y)[1] + 10),
                   "红队" if side == "red" else "黑队", font=font(16), fill=(255, 255, 255))
        # 当前目标：把这 1~2 枚圈出来，并画出推进方向
        for side, f in fsms.items():
            tgt = f.target
            if not tgt:
                continue
            for m in tgt["members"]:
                cx, cy = px(m["x"], m["y"])
                rr = int(config.PIECE_R * S * 1.6)
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                          outline=(255, 210, 0), width=4)   # 黄色圈，红车上也看得见
            ax, ay = px(tgt["plan"]["aim"][0], tgt["plan"]["aim"][1])
            bx, by = px(tgt["x"], tgt["y"])
            d.line([ax, ay, bx, by], fill=TEAM_RGB[side], width=3)
        # HUD
        remain = max(0.0, (self.seconds or config.MATCH_SECONDS) - t)
        solo = self.title or ("（红队单独演示：只推红棋，一次 1~2 枚）" if len(cars) == 1
                              else "（只能推自己颜色，推错扣分）")
        d.text((PAD_X, 14), "智能车光电象棋对抗 · " + solo,
               font=font(20), fill=(30, 30, 30))
        parts = []
        for s in cars:
            parts.append(f"{'红' if s == 'red' else '黑'}队 {world.score[s]:3d} 分"
                         f"（{world.scored_count[s]} 枚）")
        d.text((PAD_X, 44), "   ".join(parts) + f"   剩余 {remain:4.1f}s",
               font=font(22), fill=(20, 20, 20))
        y = 74
        for side in fsms:
            st = fsms[side].status(t)
            d.text((W - 520, y), f"{'红' if side == 'red' else '黑'}: {st['state']}",
                   font=font(15), fill=TEAM_RGB[side])
            tg = fsms[side].target
            if tg:
                txt = f"推{tg['own_n']}枚 净+{tg['net']}"
                tw = d.textlength(txt, font=font(13))
                d.text((W - 70 - tw, y), txt, font=font(13), fill=(110, 110, 110))
            y += 24
        if world.events:
            e = world.events[-1]
            cn = lambda s: "红" if s == "red" else "黑"  # noqa: E731
            txt = f"最近：{cn(e[1])}队 {'+' if e[4] > 0 else ''}{e[4]} 分（{cn(e[3])}方 {e[2]}）"
            d.text((PAD_X, H - 30), txt, font=font(14), fill=(110, 110, 110))
        frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        if self.out is not None:
            self.out.write(frame)
        self.frames += 1

    def close(self):
        if self.out is not None:
            self.out.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--team", choices=["red", "black", "both"], default="both",
                    help="跑哪支队：red=只红队（默认给视频用）")
    ap.add_argument("--spread", action="store_true",
                    help="棋子散布全场（默认：按规则图挤在中央堆放区）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    t0 = time.perf_counter()
    for i in range(args.episodes):
        path = (args.out or str(config.DIR_OUTPUT / "sim.mp4")) if (args.video and i == 0) else None
        rnd = Renderer(path, seconds=args.seconds)
        sides = ("red",) if args.team == "red" else ("black",) if args.team == "black" else ("red", "black")
        r = run_episode(seed=args.seed + i, seconds=args.seconds, render=rnd,
                        sides=sides, spread=args.spread)
        rnd.close()
        rows.append(r)
        line = " | ".join(
            f"{'红' if s == 'red' else '黑'}队 {r['score'][s]:4d} 分"
            f"（{r['scored'][s]} 枚，{r['trips'][s]} 趟）" for s in r["score"])
        print(f"第 {i + 1} 局：{line}")
        bad = [e for e in r["events"] if e[4] < 0]
        if bad:
            print("   误推扣分：", "；".join(f"{e[1]}队 {e[3]}方{e[2]} {e[4]}" for e in bad[:5]))
    print("-" * 70)
    for side, name in (("red", "红队"), ("black", "黑队")):
        if side not in rows[0]["score"]:
            continue
        sc = [r["score"][side] for r in rows]
        print(f"{name}：平均 {np.mean(sc):.1f} 分，最高 {max(sc)}，最低 {min(sc)}")
    print(f"仿真耗时 {time.perf_counter() - t0:.1f}s")
    if args.video:
        print(f"视频：{args.out or config.DIR_OUTPUT / 'sim.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
