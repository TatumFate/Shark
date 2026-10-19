# -*- coding: utf-8 -*-
"""红队 · 伸缩臂下探夹取（B 方案）· 真实棋堆 + 刚体物理

车体始终在棋堆外；伸缩臂 250→450mm 从上方够到目标，夹爪只罩住 1 枚，
邻子被漏斗导片推开 ≤3mm。只取红棋，黑棋进区记 0 并计为失败。
"""
import argparse
import heapq
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vision"))
import config  # noqa: E402
from rigid2d import Body, World  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PILE_JSON = ROOT / "vision" / "data" / "output" / "pipeline.json"
FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_QI = r"C:\Windows\Fonts\simkai.ttf"

FIELD_L, FIELD_W = 3000.0, 2000.0
GOAL_X, GOAL_Y0, GOAL_Y1 = 130.0, 550.0, 1650.0
BODY_L, BODY_W = 190.0, 180.0
PIECE_R = 22.0
ARM_STOW, ARM_REACH = 150.0, 450.0
ARM_DEPOSIT = 300.0
ARM_SPEED = 340.0
Z_LIFT = 55.0
JAW_OPEN, JAW_CLOSED = 60.0, 46.0
CELL = 40.0
NX = int(FIELD_L / CELL) + 1
NY = int(FIELD_W / CELL) + 1
INFLATE = 106.0


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def font(sz):
    return ImageFont.truetype(FONT, sz)


def load_pile(path=PILE_JSON):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for i, p in enumerate(data):
        out.append(Body(p["x"], p["y"], p["side"], p["char"],
                        int(p.get("value", config.VALUE.get(p["char"], 1))), i))
    return out


class Car:
    def __init__(self, x, y, h):
        self.x, self.y, self.h = x, y, h
        self.vx = self.vy = self.w = 0.0
        self.task = None
        self.reach = ARM_STOW
        self.arm_target = ARM_STOW
        self.jaw_open = JAW_OPEN
        self.jaw_down = False
        self.z = Z_LIFT

    def pose(self):
        return self.x, self.y, wrap(self.h)

    def done(self):
        return self.task is None and abs(self.reach - self.arm_target) < 1.0

    def goto(self, x, y, h, tol=12.0):
        self.task = {"t": "goto", "x": x, "y": y, "h": wrap(h), "phase": "turn", "tol": tol}

    def goto_soft(self, x, y, h):
        self.task = {"t": "goto", "x": x, "y": y, "h": wrap(h), "phase": "drive", "soft": True}

    def drive(self, mm):
        self.task = {"t": "drive", "left": float(mm), "h0": wrap(self.h)}

    def turn(self, deg):
        self.task = {"t": "turn", "h": wrap(deg)}

    def step(self, dt, v_max=1150.0):
        x0, y0, h0 = self.x, self.y, self.h
        t = self.task
        if t:
            if t["t"] == "turn":
                err = wrap(t["h"] - self.h)
                if abs(err) < 1.2:
                    self.h = t["h"]
                    self.task = None
                else:
                    self.h = wrap(self.h + max(-180, min(180, err * 6)) * dt)
            elif t["t"] == "goto":
                if t["phase"] == "turn":
                    bearing = math.degrees(math.atan2(t["y"] - self.y, t["x"] - self.x))
                    err = wrap(bearing - self.h)
                    if abs(err) < 2:
                        t["phase"] = "drive"
                    else:
                        self.h = wrap(self.h + max(-180, min(180, err * 5)) * dt)
                elif t["phase"] == "drive":
                    dx, dy = t["x"] - self.x, t["y"] - self.y
                    dist = math.hypot(dx, dy)
                    lim = 20.0 if t.get("soft") else t.get("tol", 12.0)
                    if dist < lim:
                        if t.get("soft"):
                            self.task = None
                        else:
                            t["phase"] = "align"
                    else:
                        bearing = math.degrees(math.atan2(dy, dx))
                        err = wrap(bearing - self.h)
                        if abs(err) > 6:
                            self.h = wrap(self.h + max(-180, min(180, err * 4)) * dt)
                        v = min(v_max, max(260.0, dist * 4.0)) * max(0.35, 1 - abs(err) / 100.0)
                        step = min(v * dt, dist)
                        self.x += math.cos(math.radians(self.h)) * step
                        self.y += math.sin(math.radians(self.h)) * step
                else:
                    err = wrap(t["h"] - self.h)
                    if abs(err) < 1.2:
                        self.h = t["h"]
                        self.task = None
                    else:
                        self.h = wrap(self.h + max(-180, min(180, err * 5)) * dt)
            elif t["t"] == "drive":
                left = t["left"]
                if abs(left) < 1:
                    self.task = None
                else:
                    e = wrap(t["h0"] - self.h)
                    self.h = wrap(self.h + max(-120, min(120, e * 8)) * dt)
                    v = min(v_max, max(320.0, abs(left) * 5.0))
                    step = math.copysign(min(v * dt, abs(left)), left)
                    self.x += math.cos(math.radians(self.h)) * step
                    self.y += math.sin(math.radians(self.h)) * step
                    t["left"] -= step
        dr = self.arm_target - self.reach
        if abs(dr) > 0.5:
            self.reach += math.copysign(min(ARM_SPEED * dt, abs(dr)), dr)
        self.vx = (self.x - x0) / dt
        self.vy = (self.y - y0) / dt
        self.w = math.radians(wrap(self.h - h0)) / dt
        self.x = max(30.0, min(FIELD_L - 30.0, self.x))
        self.y = max(30.0, min(FIELD_W - 30.0, self.y))

    def gripper(self):
        th = math.radians(self.h)
        return self.x + math.cos(th) * self.reach, self.y + math.sin(th) * self.reach

    def jaws(self):
        if not self.jaw_down:
            return []
        gx, gy = self.gripper()
        th = math.radians(self.h)
        lx, ly = -math.sin(th), math.cos(th)
        off = self.jaw_open / 2 + 2.0
        return [(gx + lx * off, gy + ly * off, th, 20.0, 2.0),
                (gx - lx * off, gy - ly * off, th, 20.0, 2.0)]


class Planner:
    def __init__(self, world, car, pieces):
        self.world = world
        self.car = car
        self.pieces = pieces
        self.phase = "PLAN"
        self.t = 0.0
        self.phase_t = 0.0
        self.target = None
        self.plan = None
        self.path = []
        self.path_i = 0
        self.trip = 0
        self.fail = 0
        self.log = []
        self.cands = []
        self.failed_ids = {}
        self.fail_started = False

    def set(self, p):
        self.phase = p
        self.phase_t = self.t

    def occ_grid(self):
        occ = np.zeros((NX, NY), bool)
        for p in self.pieces:
            if p.scored or p.held:
                continue
            r = INFLATE + p.r
            x0 = max(0, int((p.x - r) / CELL) - 1)
            x1 = min(NX - 1, int((p.x + r) / CELL) + 1)
            y0 = max(0, int((p.y - r) / CELL) - 1)
            y1 = min(NY - 1, int((p.y + r) / CELL) + 1)
            for ix in range(x0, x1 + 1):
                for iy in range(y0, y1 + 1):
                    if math.hypot(ix * CELL - p.x, iy * CELL - p.y) < r:
                        occ[ix, iy] = True
        return occ

    def dijkstra(self, start, occ):
        sx = max(0, min(NX - 1, int(round(start[0] / CELL))))
        sy = max(0, min(NY - 1, int(round(start[1] / CELL))))
        for ix in range(max(0, sx - 2), min(NX, sx + 3)):
            for iy in range(max(0, sy - 2), min(NY, sy + 3)):
                occ[ix, iy] = False
        dist = np.full((NX, NY), 1e18)
        parent = {}
        dist[sx, sy] = 0.0
        pq = [(0.0, sx, sy)]
        while pq:
            d, ix, iy = heapq.heappop(pq)
            if d > dist[ix, iy] + 1e-9:
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ix + dx, iy + dy
                    if not (0 <= nx < NX and 0 <= ny < NY) or occ[nx, ny]:
                        continue
                    nd = d + math.hypot(dx, dy)
                    if nd < dist[nx, ny]:
                        dist[nx, ny] = nd
                        parent[(nx, ny)] = (ix, iy)
                        heapq.heappush(pq, (nd, nx, ny))
        return dist, parent, (sx, sy)

    def path_from(self, parent, startcell, goal):
        g = (max(0, min(NX - 1, int(round(goal[0] / CELL)))),
             max(0, min(NY - 1, int(round(goal[1] / CELL)))))
        if g != startcell and g not in parent:
            return None
        cur = g
        pts = []
        for _ in range(5000):
            pts.append((cur[0] * CELL, cur[1] * CELL))
            if cur == startcell:
                break
            cur = parent[cur]
        pts.reverse()
        return pts[::3] + [pts[-1]]

    def path_to(self, goal):
        occ = self.occ_grid()
        gcell = (max(0, min(NX - 1, int(round(goal[0] / CELL)))),
                 max(0, min(NY - 1, int(round(goal[1] / CELL)))))
        for ix in range(max(0, gcell[0] - 2), min(NX, gcell[0] + 3)):
            for iy in range(max(0, gcell[1] - 2), min(NY, gcell[1] + 3)):
                occ[ix, iy] = False
        dist, parent, sc = self.dijkstra((self.car.x, self.car.y), occ)
        if dist[gcell] > 1e17:
            return None
        return self.path_from(parent, sc, goal)

    def clear_obb(self, x, y, h, ignore):
        th = math.radians(h)
        c, s = math.cos(th), math.sin(th)
        for p in self.pieces:
            if p.scored or p.held or p.id == ignore:
                continue
            dx, dy = p.x - x, p.y - y
            lx = c * dx + s * dy
            ly = -s * dx + c * dy
            if abs(lx) < BODY_L / 2 + p.r + 6 and abs(ly) < BODY_W / 2 + p.r + 6:
                return False
        return True

    def plan_trip(self):
        occ = self.occ_grid()
        dist, parent, sc = self.dijkstra((self.car.x, self.car.y), occ)
        cands = []
        for p in self.pieces:
            if p.scored or p.held or p.side != "red":
                continue
            if p.id in self.failed_ids and self.t - self.failed_ids[p.id] < 15:
                continue
            jam = False
            for q in self.pieces:
                if q is p or q.scored or q.held:
                    continue
                if math.hypot(q.x - p.x, q.y - p.y) < 30.0:
                    jam = True
                    break
            if jam:
                continue
            best = None
            for ang in range(0, 360, 30):
                th = math.radians(ang)
                ux, uy = math.cos(th), math.sin(th)
                cx = p.x + ux * ARM_REACH
                cy = p.y + uy * ARM_REACH
                if not (70 < cx < FIELD_L - 70 and 70 < cy < FIELD_W - 70):
                    continue
                heading = math.degrees(th + math.pi)
                if not self.clear_obb(cx, cy, heading, p.id):
                    continue
                ix = max(0, min(NX - 1, int(round(cx / CELL))))
                iy = max(0, min(NY - 1, int(round(cy / CELL))))
                if dist[ix, iy] > 1e17:
                    continue
                plen = float(dist[ix, iy]) * CELL
                dep_y = min(max(p.y, 600.0), 1600.0)
                carry = math.hypot(cx - 360.0, cy - dep_y)
                t_total = plen / 1150.0 + 3.2 + carry / 1150.0 + 1.6
                eff = p.value / t_total
                if best is None or t_total < best[0]:
                    best = (t_total, cx, cy, heading, plen, carry, eff)
            if best:
                cands.append((best[6], p, best))
        cands.sort(key=lambda z: -z[0])
        self.cands = cands[:4]
        return cands[0] if cands else None

    def funnel(self):
        p = self.target
        moved = 0
        for q in self.pieces:
            if q is p or q.scored or q.held:
                continue
            dx, dy = q.x - p.x, q.y - p.y
            d = math.hypot(dx, dy)
            if 26 <= d < 52:
                ux, uy = dx / d, dy / d
                push = min(3.0, (52 - d) * 0.3)
                q.x += ux * push
                q.y += uy * push
                q.vx += ux * push * 6
                q.vy += uy * push * 6
                moved += 1
        return moved

    def update(self, t, dt):
        self.t = t
        c = self.car
        if self.phase == "PLAN":
            best = self.plan_trip()
            if not best:
                self.set("DONE")
                return
            eff, p, info = best
            t_total, cx, cy, h, plen, carry, _ = info
            self.target = p
            self.plan = {"eff": eff, "t_total": t_total, "standoff": (cx, cy),
                         "heading": h, "plen": plen, "carry": carry}
            pts = self.path_to((cx, cy))
            if not pts:
                self.set("DONE")
                return
            self.path, self.path_i = pts, 0
            self.set("TRAVEL")
            c.goto_soft(pts[0][0], pts[0][1], h)
            return
        if self.phase == "TRAVEL":
            if c.done():
                self.path_i += 1
                if self.path_i >= len(self.path):
                    # 路径点是 40mm 栅格，最后必须走到精确取棋位，否则夹爪会差 2~4cm
                    cx, cy = self.plan["standoff"]
                    self.set("PRECISE")
                    c.goto(cx, cy, self.plan["heading"], tol=3.0)
                else:
                    c.goto_soft(self.path[self.path_i][0], self.path[self.path_i][1],
                                self.plan["heading"])
            elif t - self.phase_t > 32:
                self.set("FAIL")
            return
        if self.phase == "PRECISE":
            if c.done():
                self.set("AIM")
                c.turn(self.plan["heading"])
            elif t - self.phase_t > 8:
                self.set("FAIL")
            return
        if self.phase == "AIM":
            if c.done():
                c.arm_target = ARM_REACH
                self.set("EXTEND")
            return
        if self.phase == "EXTEND":
            if c.done():
                c.jaw_open = JAW_OPEN
                c.jaw_down = True
                self.funnel()
                self.set("DESCEND")
            return
        if self.phase == "DESCEND":
            if t - self.phase_t >= 0.35:
                c.z = 0.0
                c.jaw_open = JAW_CLOSED
                self.set("GRIP")
            return
        if self.phase == "GRIP":
            if t - self.phase_t >= 0.30:
                p = self.target
                gx, gy = c.gripper()
                if math.hypot(p.x - gx, p.y - gy) > 20:
                    self.set("FAIL")
                    return
                p.held = True
                p.vx = p.vy = 0.0
                self.set("LIFT")
            return
        if self.phase == "LIFT":
            if t - self.phase_t >= 0.40:
                c.z = Z_LIFT
                c.jaw_down = False
                c.arm_target = ARM_STOW
                self.set("RETRACT")
            return
        if self.phase == "RETRACT":
            if c.reach <= ARM_STOW + 1:
                gy = min(max(self.target.y, 600.0), 1600.0)
                pts = self.path_to((360.0, gy))
                if not pts:
                    self.set("FAIL")
                    return
                self.path, self.path_i = pts, 0
                self.set("CARRY")
                c.goto_soft(pts[0][0], pts[0][1], 180.0)
            return
        if self.phase == "CARRY":
            if c.done():
                self.path_i += 1
                if self.path_i >= len(self.path):
                    self.set("DEPOSIT")
                    c.turn(180.0)
                else:
                    c.goto_soft(self.path[self.path_i][0], self.path[self.path_i][1], 180.0)
            elif t - self.phase_t > 32:
                self.set("FAIL")
            return
        if self.phase == "DEPOSIT":
            if c.done():
                if c.reach < ARM_DEPOSIT - 1:
                    c.arm_target = ARM_DEPOSIT
                    return
                p = self.target
                gx, gy = c.gripper()
                p.x, p.y = gx, gy
                p.vx = p.vy = 0.0
                p.held = False
                p.z = 0.0
                c.jaw_down = False
                c.arm_target = ARM_STOW
                self.trip += 1
                self.log.append((round(t, 1), self.trip, p.char, p.value))
                self.set("BACK")
                c.drive(-240.0)
            return
        if self.phase == "BACK":
            if c.done():
                self.target = None
                self.set("PLAN")
            return
        if self.phase == "FAIL":
            c.jaw_down = False
            c.arm_target = ARM_STOW
            if not self.fail_started:
                self.fail_started = True
                c.drive(-260.0)
            if c.done():
                if self.target:
                    self.failed_ids[self.target.id] = t
                self.fail += 1
                self.fail_started = False
                self.target = None
                self.set("PLAN")
            return

    def sync_held(self):
        if self.target and self.target.held:
            gx, gy = self.car.gripper()
            self.target.x, self.target.y = gx, gy
            self.target.z = self.car.z
            self.target.vx = self.car.vx
            self.target.vy = self.car.vy


# ------------------------------------------------------------------ 渲染
S = 0.40
PAD_L, PAD_T = 60, 118
W = int(FIELD_L * S) + 2 * PAD_L + 500
H = int(FIELD_W * S) + PAD_T + 104


class Renderer:
    def __init__(self, out=None, fps=25, title="", note=""):
        self.path = Path(out) if out else None
        self.vw = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (W, H))
        self.title = title
        self.note = note
        self.tiles = {}
        self.frames = 0
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
        gy0 = y0 + GOAL_Y0 * S
        gy1 = y0 + GOAL_Y1 * S
        d.rectangle([x0, gy0, x0 + GOAL_X * S, gy1], fill=(236, 214, 210),
                    outline=(178, 96, 88), width=2)
        for k in range(6):
            yy = gy0 + 12 + k * 55
            d.line([(x0 + 4, yy), (x0 + GOAL_X * S - 4, yy)], fill=(226, 196, 192), width=3)
        d.text((x0 + 4, gy0 + 6), "红队\n球门区", font=font(11), fill=(150, 70, 62), spacing=2)
        gxb = x1 - GOAL_X * S
        d.rectangle([gxb, gy0, x1, gy1], fill=(208, 214, 224), outline=(96, 112, 150), width=2)
        d.text((x1 - 8, gy0 + 8), "黑队球门", font=font(11), fill=(70, 84, 120), anchor="ra")
        # 堆放区（实际棋子的外接框）
        d.rectangle([x0 + 1200 * S, y0 + 750 * S, x0 + 1800 * S, y0 + 1250 * S],
                    outline=(150, 172, 152), width=1)
        d.text((x0 + 1204 * S, y0 + 1254 * S), "棋子堆放区 600×500",
               font=font(12), fill=(110, 140, 112))
        d.text((PAD_L, 18), "智能车光电象棋 · 红队伸缩臂夹取演示", font=font(24), fill=(28, 28, 30))
        d.text((PAD_L, 52), self.title, font=font(17), fill=(70, 72, 76))
        d.text((PAD_L, 80), self.note, font=font(14), fill=(120, 120, 124))
        return img

    def tile(self, char, side):
        key = (char, side)
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
        bb = d.textbbox((0, 0), char, font=f)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        d.text(((big - w) / 2 - bb[0], (big - h) / 2 - bb[1]), char, font=f, fill=ink)
        t = img.resize((dia, dia), Image.LANCZOS)
        self.tiles[key] = t
        return t

    def px(self, x, y):
        return (PAD_L + x * S, PAD_T + y * S)

    def __call__(self, pieces, car, ctrl, world, t):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        if ctrl.target and not ctrl.target.scored:
            p = ctrl.target
            cx, cy = self.px(p.x, p.y)
            d.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], outline=(255, 190, 30), width=3)
            gx, gy = self.px(*car.gripper())
            if car.reach > 260:
                d.line([gx, gy, cx, cy], fill=(255, 190, 30), width=1)
        for p in pieces:
            if p.scored:
                continue
            cx, cy = self.px(p.x, p.y)
            tile = self.tile(p.char, p.side)
            if p.held:
                ox = -p.z * 0.12
                oy = p.z * 0.10
                d.ellipse([cx - tile.width * 0.42 + ox + 4, cy - tile.height * 0.42 + oy + 5,
                           cx + tile.width * 0.42 + ox + 4, cy + tile.height * 0.42 + oy + 5],
                          fill=(170, 170, 162))
                img.paste(tile, (int(cx - tile.width / 2 + ox),
                                 int(cy - tile.height / 2 + oy)), tile)
            else:
                d.ellipse([cx - tile.width * 0.46 + 2, cy - tile.height * 0.44 + 3,
                           cx + tile.width * 0.46 + 2, cy + tile.height * 0.44 + 3],
                          fill=(174, 174, 166))
                img.paste(tile, (int(cx - tile.width / 2), int(cy - tile.height / 2)), tile)
        # 车 + 伸缩臂 + 夹爪
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
        # 臂根
        r0 = self.px(car.x + f[0] * 88 + l[0] * 26, car.y + f[1] * 88 + l[1] * 26)
        r1 = self.px(car.x + f[0] * 88 - l[0] * 26, car.y + f[1] * 88 - l[1] * 26)
        d.line([r0, r1], fill=(70, 78, 88), width=12)
        gx, gy = car.gripper()
        a0 = self.px(car.x + f[0] * 92, car.y + f[1] * 92)
        a1 = self.px(gx, gy)
        d.line([a0, a1], fill=(150, 156, 164), width=6)
        # 夹爪两片
        jaw = car.jaw_open / 2
        for sgn in (-1, 1):
            j0 = self.px(gx + l[0] * sgn * jaw - f[0] * 10,
                         gy + l[1] * sgn * jaw - f[1] * 10)
            j1 = self.px(gx + l[0] * sgn * jaw + f[0] * 12,
                         gy + l[1] * sgn * jaw + f[1] * 12)
            d.line([j0, j1], fill=(40, 110, 150), width=5)
        if car.jaw_down:
            for rad in (50,):
                box = [self.px(gx - rad, gy - rad), self.px(gx + rad, gy - rad),
                       self.px(gx + rad, gy + rad), self.px(gx - rad, gy + rad)]
                for k in range(4):
                    d.line([box[k], box[(k + 1) % 4]], fill=(230, 170, 60), width=1)
        # 右侧面板
        x1 = PAD_L + FIELD_L * S
        px0 = int(x1 + 30)
        d.rectangle([x1 + 12, 0, W, H], fill=(251, 250, 246))
        d.text((px0, 12), "红队 · 只取红棋", font=font(13), fill=(170, 60, 60))
        d.text((px0, 30), f"{world.red_in:2d}", font=font(44), fill=(30, 30, 32))
        d.text((px0 + 60, 48), "/ 16 枚", font=font(19), fill=(110, 112, 116))
        d.text((px0 + 235, 34), f"{world.red_score}", font=font(38), fill=(176, 44, 44))
        d.text((px0 + 235, 78), "红方得分", font=font(12), fill=(130, 110, 110))
        bad_col = (200, 40, 40) if world.black_in else (40, 130, 80)
        d.text((px0, 90), f"黑棋进区  {world.black_in}", font=font(15), fill=bad_col)
        d.text((px0 + 130, 90), "（必须为 0）", font=font(12), fill=(150, 120, 120))
        bx0, bx1 = px0, px0 + 430
        d.rectangle([bx0, 114, bx1, 120], fill=(214, 214, 208))
        d.rectangle([bx0, 114, bx0 + (bx1 - bx0) * min(1.0, world.red_in / 16.0), 120],
                    fill=(206, 152, 48))
        # 棋堆特写
        cw, ch = 560.0, 440.0
        if ctrl.target:
            tx, ty = ctrl.target.x, ctrl.target.y
            gx, gy = car.gripper()
            tx, ty = (tx + gx) / 2, (ty + gy) / 2
        else:
            tx, ty = car.gripper()
        ccx = PAD_L + tx * S
        ccy = PAD_T + ty * S
        sw, sh = cw * S, ch * S
        ccx = max(PAD_L + sw / 2 + 2, min(PAD_L + FIELD_L * S - sw / 2 - 2, ccx))
        ccy = max(PAD_T + sh / 2 + 2, min(PAD_T + FIELD_W * S - sh / 2 - 2, ccy))
        crop = img.crop((int(ccx - sw / 2), int(ccy - sh / 2),
                         int(ccx + sw / 2), int(ccy + sh / 2)))
        iw = 430
        crop = crop.resize((iw, int(iw * ch / cw)), Image.LANCZOS)
        iy = 150
        d.text((px0, iy - 22), "棋堆特写（伸缩臂 / 下探夹爪）", font=font(13), fill=(70, 72, 76))
        img.paste(crop, (px0, iy))
        d.rectangle([px0, iy, px0 + iw, iy + crop.height], outline=(84, 86, 90), width=2)
        sy = iy + crop.height + 12
        if ctrl.target:
            p = ctrl.target
            d.text((px0, sy), f"当前：{'红' if p.side == 'red' else '黑'}{p.char} "
                              f"{p.value} 分   效率 {ctrl.plan['eff']:.2f} 分/秒",
                   font=font(15), fill=(120, 86, 30))
        names = {"PLAN": "按效率选目标", "TRAVEL": "绕到取棋位", "PRECISE": "精确对位",
                 "AIM": "摆正车头",
                 "EXTEND": "伸出伸缩臂", "DESCEND": "下探罩住棋子", "GRIP": "夹紧",
                 "LIFT": "提起", "RETRACT": "缩臂", "CARRY": "搬运",
                 "DEPOSIT": "放入球门", "BACK": "退车", "FAIL": "目标丢失",
                 "DONE": "本场结束"}
        d.text((px0, sy + 26), f"状态：{names.get(ctrl.phase, ctrl.phase)}", font=font(15),
               fill=(54, 78, 104))
        d.text((px0, sy + 50), "效率候选（分值 / 预计秒数）", font=font(12), fill=(130, 130, 134))
        yy = sy + 70
        for eff, p, info in ctrl.cands:
            d.text((px0, yy), f"红{p.char} {p.value:2d}分   {eff:.2f} 分/秒   预计 {info[0]:.1f}s",
                   font=font(12), fill=(96, 98, 102))
            yy += 19
        d.text((px0, sy + 160), f"时间 {t:5.1f}s   趟数 {ctrl.trip}   失败 {ctrl.fail}",
               font=font(12), fill=(110, 110, 114))
        if world.events:
            e = world.events[-1]
            d.text((PAD_L, PAD_T + FIELD_W * S + 22),
                   f"t = {t:5.1f}s    最近进区：{'红' if e[0] == 'red' else '黑'}{e[1]} "
                   f"{'+' if e[2] > 0 else ''}{e[2]}",
                   font=font(14), fill=(100, 100, 104))
        frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        if self.vw:
            self.vw.write(frame)
        self.frames += 1
        if self.frames % 100 == 0:
            print(f"  渲染 {self.frames} 帧  t={t:.1f}s  红 {world.red_in}/16  "
                  f"黑进区 {world.black_in}")

    def close(self):
        if self.vw:
            self.vw.release()
            print(f"[视频] {self.path}  {self.frames} 帧")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=200.0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    pieces = load_pile()
    world = World(pieces)
    car = Car(320.0, 1790.0, 0.0)
    ctrl = Planner(world, car, pieces)
    stats = {"red": 0, "black": 0}
    events = []

    def score(p):
        if p.side == "red":
            stats["red"] += p.value
            events.append(("red", p.char, p.value))
        else:
            stats["black"] += 1
            events.append(("black", p.char, -p.value))

    if args.video:
        title = "B 方案 · 车不进堆，伸缩臂 450mm 逐枚夹取；只取红棋"
        note = "真实棋堆（取自你 10-17 实拍识别结果）｜刚体物理：质量/摩擦/恢复系数/子步进"
        out = args.out or str(ROOT / "vision" / "data" / "output" / "scoop_red.mp4")
        rnd = Renderer(out, fps=25, title=title, note=note)
    else:
        rnd = None
    print(f"棋子 {len(pieces)} 枚，红 {sum(1 for p in pieces if p.side == 'red')}，"
          f"黑 {sum(1 for p in pieces if p.side == 'black')}")
    dt = 0.02
    for i in range(int(args.seconds / dt)):
        t = i * dt
        ctrl.update(t, dt)
        ctrl.sync_held()
        car.step(dt)
        world.step(dt, car, car.jaws(), (0.0, GOAL_X, GOAL_Y0, GOAL_Y1), score)
        if rnd and i % 2 == 0:
            st = type("W", (), {"red_in": sum(1 for p in pieces if p.side == "red" and p.scored),
                                "red_score": stats["red"],
                                "black_in": stats["black"],
                                "events": events})()
            rnd(world.bodies, car, ctrl, st, t)
        if ctrl.phase == "DONE":
            print(f"没有更多可达红棋：t={t:.1f}s")
            break
    if rnd:
        for k in range(25):
            st = type("W", (), {"red_in": sum(1 for p in pieces if p.side == "red" and p.scored),
                                "red_score": stats["red"],
                                "black_in": stats["black"],
                                "events": events})()
            rnd(world.bodies, car, ctrl, st, args.seconds + k * 0.04)
        rnd.close()
    print(f"结果：红棋 {sum(1 for p in pieces if p.side == 'red' and p.scored)}/16，"
          f"得分 {stats['red']}，"
          f"黑棋进区 {stats['black']}，趟数 {ctrl.trip}，失败 {ctrl.fail}")
    for row in ctrl.log:
        print(f"  第 {row[1]:2d} 趟 t={row[0]:5.1f}s  {row[2]} +{row[3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
