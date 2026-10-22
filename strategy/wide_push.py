# -*- coding: utf-8 -*-
"""宽推扫堆（任意颜色都计分）· 真实棋堆 + 刚体物理

300mm 折叠铲从棋堆右侧整排扫向左侧球门；每趟按“分值 / 预计秒数”选扫掠带。
这是“两套算法”的第一套：all = 红黑都推，追求总分最高。
第二套 own = 只推红或只推黑，见 push_demo.py / scoop_sim.py。
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vision"))
from rigid2d import Body, World  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PILE_JSON = ROOT / "vision" / "data" / "output" / "pipeline.json"
FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_QI = r"C:\Windows\Fonts\simkai.ttf"
FIELD_L, FIELD_W = 3000.0, 2000.0
GOAL_X, GOAL_Y0, GOAL_Y1 = 130.0, 550.0, 1650.0
BODY_L, BODY_W = 190.0, 180.0
PIECE_R = 22.0
BLADE_HALF = 150.0
BLADE_AHEAD = 150.0
BLADE_THICK = 6.0


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def font(sz):
    return ImageFont.truetype(FONT, sz)


def load_pile():
    d = json.loads(PILE_JSON.read_text(encoding="utf-8"))
    return [Body(p["x"], p["y"], p["side"], p["char"], int(p["value"]), i) for i, p in enumerate(d)]


class Car:
    def __init__(self, x, y, h):
        self.x, self.y, self.h = x, y, h
        self.vx = self.vy = self.w = 0.0
        self.task = None
        self.blade = False

    def done(self):
        return self.task is None

    def goto(self, x, y, h, tol=14.0):
        self.task = {"t": "goto", "x": x, "y": y, "h": wrap(h), "phase": "turn", "tol": tol}

    def drive(self, mm, speed=650.0):
        self.task = {"t": "drive", "left": float(mm), "h0": wrap(self.h), "v": speed}

    def turn(self, deg):
        self.task = {"t": "turn", "h": wrap(deg)}

    def step(self, dt):
        x0, y0, h0 = self.x, self.y, self.h
        t = self.task
        if t:
            if t["t"] == "turn":
                err = wrap(t["h"] - self.h)
                if abs(err) < 1.2:
                    self.h, self.task = t["h"], None
                else:
                    self.h = wrap(self.h + max(-170, min(170, err * 6)) * dt)
            elif t["t"] == "goto":
                if t["phase"] == "turn":
                    bearing = math.degrees(math.atan2(t["y"] - self.y, t["x"] - self.x))
                    err = wrap(bearing - self.h)
                    if abs(err) < 2:
                        t["phase"] = "drive"
                    else:
                        self.h = wrap(self.h + max(-170, min(170, err * 5)) * dt)
                elif t["phase"] == "drive":
                    dx, dy = t["x"] - self.x, t["y"] - self.y
                    dist = math.hypot(dx, dy)
                    if dist < t.get("tol", 14.0):
                        t["phase"] = "align"
                    else:
                        bearing = math.degrees(math.atan2(dy, dx))
                        err = wrap(bearing - self.h)
                        if abs(err) > 6:
                            self.h = wrap(self.h + max(-170, min(170, err * 4)) * dt)
                        v = min(1100.0, max(280.0, dist * 4.0)) * max(0.35, 1 - abs(err) / 100.0)
                        step = min(v * dt, dist)
                        self.x += math.cos(math.radians(self.h)) * step
                        self.y += math.sin(math.radians(self.h)) * step
                else:
                    err = wrap(t["h"] - self.h)
                    if abs(err) < 1.2:
                        self.h, self.task = t["h"], None
                    else:
                        self.h = wrap(self.h + max(-170, min(170, err * 5)) * dt)
            elif t["t"] == "drive":
                left = t["left"]
                if abs(left) < 1:
                    self.task = None
                else:
                    e = wrap(t["h0"] - self.h)
                    self.h = wrap(self.h + max(-90, min(90, e * 8)) * dt)
                    v = t.get("v", 650.0)
                    step = math.copysign(min(v * dt, abs(left)), left)
                    self.x += math.cos(math.radians(self.h)) * step
                    self.y += math.sin(math.radians(self.h)) * step
                    t["left"] -= step
        self.vx = (self.x - x0) / dt
        self.vy = (self.y - y0) / dt
        self.w = math.radians(wrap(self.h - h0)) / dt
        self.x = max(30.0, min(FIELD_L - 30.0, self.x))
        self.y = max(30.0, min(FIELD_W - 30.0, self.y))

    def blade_obbs(self):
        if not self.blade:
            return []
        th = math.radians(self.h)
        f = (math.cos(th), math.sin(th))
        l = (-f[1], f[0])
        bx = self.x + f[0] * BLADE_AHEAD
        by = self.y + f[1] * BLADE_AHEAD
        out = [(bx, by, th, BLADE_THICK, BLADE_HALF)]
        for sgn in (-1, 1):
            gx = self.x + f[0] * (BLADE_AHEAD - 30) + l[0] * sgn * BLADE_HALF
            gy = self.y + f[1] * (BLADE_AHEAD - 30) + l[1] * sgn * BLADE_HALF
            out.append((gx, gy, th, 34.0, 5.0))
        return out


class Planner:
    def __init__(self, world, car, pieces):
        self.world, self.car, self.pieces = world, car, pieces
        self.phase = "NEXT"
        self.phase_t = 0.0
        self.t = 0.0
        self.pass_y = 1000.0
        self.pass_value = 0
        self.pass_no = 0
        self.plan = []
        self.way = []
        self.wi = 0

    def set(self, p):
        self.phase = p
        self.phase_t = self.t

    def remaining(self):
        return [p for p in self.pieces if not p.scored]

    def best_pass(self):
        rem = [p for p in self.remaining() if p.x > 200]
        if not rem:
            return None
        best = None
        for y in (1000.0, 800.0, 1200.0, 650.0, 1350.0):
            val = sum(p.value for p in rem if abs(p.y - y) <= BLADE_HALF)
            if val == 0:
                continue
            dist = math.hypot(self.car.x - 2200.0, self.car.y - 1700.0) + abs(1700.0 - y)
            t_est = dist / 1000.0 + 2200.0 / 650.0 + 2.0
            eff = val / t_est
            if best is None or eff > best[0]:
                best = (eff, y, val, t_est)
        return best

    def update(self, t, dt):
        self.t = t
        c = self.car
        if self.phase == "NEXT":
            if not self.remaining() or self.pass_no >= 5:
                self.set("DONE")
                return
            b = self.best_pass()
            if not b:
                self.set("DONE")
                return
            eff, y, val, est = b
            self.pass_y, self.pass_value = y, val
            self.pass_no += 1
            self.way = [(320.0, 1700.0), (2200.0, 1700.0), (2200.0, y)]
            self.wi = 0
            self.set("TRAVEL")
            c.goto(*self.way[0], 0.0)
            self.plan.append((self.pass_no, y, val, round(eff, 2)))
            return
        if self.phase == "TRAVEL":
            if c.done():
                self.wi += 1
                if self.wi >= len(self.way):
                    self.set("AIM")
                    c.turn(180.0)
                else:
                    c.goto(*self.way[self.wi], 180.0 if self.wi == len(self.way) - 1 else 0.0)
            elif t - self.phase_t > 18:
                self.set("NEXT")
            return
        if self.phase == "AIM":
            if c.done():
                c.blade = True
                self.set("PUSH")
                c.drive(max(100.0, c.x - 250.0), speed=650.0)
            return
        if self.phase == "PUSH":
            if c.done():
                c.blade = False
                self.set("BACK")
                c.drive(-600.0, speed=900.0)
            elif t - self.phase_t > 12:
                self.set("BACK")
                c.blade = False
                c.drive(-600.0, speed=900.0)
            return
        if self.phase == "BACK":
            if c.done():
                self.set("NEXT")
            return


S = 0.40
PAD_L, PAD_T = 60, 118
W = int(FIELD_L * S) + 2 * PAD_L + 500
H = int(FIELD_W * S) + PAD_T + 104


class Renderer:
    def __init__(self, out=None, fps=25):
        self.path = Path(out) if out else None
        self.vw = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (W, H))
        self.frames = 0
        self.tiles = {}
        self.bg = self._bg()

    def _bg(self):
        img = Image.new("RGB", (W, H), (246, 245, 240))
        d = ImageDraw.Draw(img)
        x0, y0 = PAD_L, PAD_T
        x1 = x0 + FIELD_L * S
        y1 = y0 + FIELD_W * S
        d.rectangle([x0 - 10, y0 - 10, x1 + 10, y1 + 10], fill=(66, 68, 70))
        d.rectangle([x0, y0, x1, y1], fill=(226, 226, 218))
        for mm in range(0, int(FIELD_L) + 1, 100):
            xx = x0 + mm * S
            d.line([(xx, y0), (xx, y1)], fill=(219, 219, 211), width=1)
        for mm in range(0, int(FIELD_W) + 1, 100):
            yy = y0 + mm * S
            d.line([(x0, yy), (x1, yy)], fill=(219, 219, 211), width=1)
        gy0, gy1 = y0 + GOAL_Y0 * S, y0 + GOAL_Y1 * S
        d.rectangle([x0, gy0, x0 + GOAL_X * S, gy1], fill=(236, 214, 210),
                    outline=(178, 96, 88), width=2)
        d.rectangle([x1 - GOAL_X * S, gy0, x1, gy1], fill=(208, 214, 224),
                    outline=(96, 112, 150), width=2)
        d.text((x0 + 4, gy0 + 6), "己方\n球门", font=font(11), fill=(150, 70, 62), spacing=2)
        d.rectangle([x0 + 1200 * S, y0 + 750 * S, x0 + 1800 * S, y0 + 1250 * S],
                    outline=(150, 172, 152), width=1)
        d.text((x0 + 1204 * S, y0 + 1254 * S), "真实棋堆（32 枚）", font=font(12), fill=(110, 140, 112))
        d.text((PAD_L, 18), "智能车光电象棋 · 宽推扫堆（任意颜色计分）", font=font(24), fill=(28, 28, 30))
        d.text((PAD_L, 52), "折叠铲展开 300mm，从右侧整排扫向左侧球门；按分值/秒选扫掠带",
               font=font(17), fill=(70, 72, 76))
        d.text((PAD_L, 80), "真实棋堆（pipeline.json）｜刚体物理：质量/摩擦/恢复系数/子步进",
               font=font(14), fill=(120, 120, 124))
        return img

    def tile(self, ch, side):
        key = (ch, side)
        if key in self.tiles:
            return self.tiles[key]
        dia = int(PIECE_R * 2 * S)
        big = dia * 4
        img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        c, r = big // 2, big // 2 - 3
        d.ellipse([c - r, c - r, c + r, c + r], fill=(232, 218, 196),
                  outline=(120, 94, 68), width=max(2, big // 28))
        d.ellipse([c - r * 0.87, c - r * 0.87, c + r * 0.87, c + r * 0.87],
                  outline=(176, 148, 112), width=max(1, big // 64))
        f = ImageFont.truetype(FONT_QI, int(big * 0.56))
        ink = (188, 36, 42) if side == "red" else (34, 34, 38)
        bb = d.textbbox((0, 0), ch, font=f)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        d.text(((big - w) / 2 - bb[0], (big - h) / 2 - bb[1]), ch, font=f, fill=ink)
        t = img.resize((dia, dia), Image.LANCZOS)
        self.tiles[key] = t
        return t

    def px(self, x, y):
        return (PAD_L + x * S, PAD_T + y * S)

    def __call__(self, pieces, car, ctrl, score, in_goal, t):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        for p in pieces:
            if p.scored:
                continue
            cx, cy = self.px(p.x, p.y)
            tile = self.tile(p.char, p.side)
            d.ellipse([cx - tile.width * 0.46 + 2, cy - tile.height * 0.44 + 3,
                       cx + tile.width * 0.46 + 2, cy + tile.height * 0.44 + 3],
                      fill=(174, 174, 166))
            img.paste(tile, (int(cx - tile.width / 2), int(cy - tile.height / 2)), tile)
        th = math.radians(car.h)
        f = (math.cos(th), math.sin(th))
        l = (-f[1], f[0])
        hl, hw = BODY_L / 2, BODY_W / 2
        corners = [self.px(car.x + f[0] * a + l[0] * b, car.y + f[1] * a + l[1] * b)
                   for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
        d.polygon(corners, fill=(52, 120, 156), outline=(20, 50, 70), width=2)
        for sgn in (-1, 1):
            wx = car.x - f[0] * 40 + l[0] * sgn * 88
            wy = car.y - f[1] * 40 + l[1] * sgn * 88
            d.line([self.px(wx - f[0] * 30, wy - f[1] * 30),
                    self.px(wx + f[0] * 30, wy + f[1] * 30)], fill=(34, 34, 36), width=10)
        if car.blade:
            b0 = self.px(car.x + f[0] * BLADE_AHEAD + l[0] * BLADE_HALF,
                         car.y + f[1] * BLADE_AHEAD + l[1] * BLADE_HALF)
            b1 = self.px(car.x + f[0] * BLADE_AHEAD - l[0] * BLADE_HALF,
                         car.y + f[1] * BLADE_AHEAD - l[1] * BLADE_HALF)
            d.line([b0, b1], fill=(226, 152, 54), width=8)
            for sgn in (-1, 1):
                g0 = self.px(car.x + f[0] * 90 + l[0] * sgn * BLADE_HALF,
                             car.y + f[1] * 90 + l[1] * sgn * BLADE_HALF)
                g1 = self.px(car.x + f[0] * BLADE_AHEAD + l[0] * sgn * BLADE_HALF,
                             car.y + f[1] * BLADE_AHEAD + l[1] * sgn * BLADE_HALF)
                d.line([g0, g1], fill=(205, 132, 40), width=5)
        x1 = PAD_L + FIELD_L * S
        px0 = int(x1 + 30)
        d.rectangle([x1 + 12, 0, W, H], fill=(251, 250, 246))
        d.text((px0, 14), "任意颜色 · 总分", font=font(13), fill=(110, 112, 116))
        d.text((px0, 32), f"{score}", font=font(46), fill=(176, 44, 44))
        d.text((px0 + 95, 50), "/ 179 分", font=font(19), fill=(110, 112, 116))
        d.text((px0, 92), f"进区 {in_goal} / 32 枚", font=font(16), fill=(70, 72, 76))
        bx0, bx1 = px0, px0 + 430
        d.rectangle([bx0, 120, bx1, 128], fill=(214, 214, 208))
        d.rectangle([bx0, 120, bx0 + (bx1 - bx0) * min(1.0, score / 179.0), 128],
                    fill=(206, 152, 48))
        d.text((px0, 152), f"第 {ctrl.pass_no} 趟 · 扫掠带 y={ctrl.pass_y:.0f}",
               font=font(15), fill=(120, 86, 30))
        d.text((px0, 178), f"本带面值 {ctrl.pass_value} 分", font=font(14), fill=(96, 98, 102))
        d.text((px0, 210), "扫掠计划（分值 / 预计秒数）", font=font(12), fill=(130, 130, 134))
        yy = 232
        for row in ctrl.plan[-5:]:
            d.text((px0, yy), f"第{row[0]}趟 y={row[1]:.0f}  {row[2]}分  {row[3]}分/秒",
                   font=font(12), fill=(96, 98, 102))
            yy += 20
        names = {"NEXT": "选下一扫掠带", "TRAVEL": "绕到棋堆右侧", "AIM": "摆正车头",
                 "PUSH": "宽铲整排推进", "BACK": "退车", "DONE": "本场结束"}
        d.text((px0, 360), f"状态：{names.get(ctrl.phase, ctrl.phase)}", font=font(15), fill=(54, 78, 104))
        d.text((px0, 390), f"时间 {t:5.1f}s", font=font(13), fill=(110, 110, 114))
        d.text((px0, 430), "说明：宽推不区分红黑，", font=font(12), fill=(120, 120, 124))
        d.text((px0, 450), "按分值/秒先扫最值钱的一条带。", font=font(12), fill=(120, 120, 124))
        frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        if self.vw:
            self.vw.write(frame)
        self.frames += 1
        if self.frames % 100 == 0:
            print(f"  渲染 {self.frames}  t={t:.1f}s  {score}分 {in_goal}枚")

    def close(self):
        if self.vw:
            self.vw.release()
            print(f"[视频] {self.path} {self.frames} 帧")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=50.0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pieces = load_pile()
    world = World(pieces)
    car = Car(320.0, 1790.0, 0.0)
    ctrl = Planner(world, car, pieces)
    stats = {"score": 0, "in": 0, "black": 0}

    def score(p):
        stats["score"] += p.value
        stats["in"] += 1
        if p.side == "black":
            stats["black"] += 1

    rnd = Renderer(args.out or str(ROOT / "vision" / "data" / "output" / "wide_push.mp4"),
                   fps=25) if args.video else None
    dt = 0.02
    t = 0.0
    for i in range(int(args.seconds / dt)):
        t = i * dt
        ctrl.update(t, dt)
        car.step(dt)
        world.step(dt, car, car.blade_obbs(), (0.0, GOAL_X, GOAL_Y0, GOAL_Y1), score)
        if rnd and i % 2 == 0:
            rnd(pieces, car, ctrl, stats["score"], stats["in"], t)
        if ctrl.phase == "DONE":
            break
    if rnd:
        for k in range(25):
            rnd(pieces, car, ctrl, stats["score"], stats["in"], args.seconds + k * 0.04)
        rnd.close()
    print(f"结果：{stats['score']}/179 分，进区 {stats['in']}/32 枚，用时 {t:.1f}s，"
          f"效率 {stats['score']/max(t,0.1):.1f} 分/秒")
    for row in ctrl.plan:
        print(f"  第{row[0]}趟 y={row[1]:.0f} 本带{row[2]}分 预计{row[3]}分/秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
