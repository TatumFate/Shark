# -*- coding: utf-8 -*-
"""窄推机构 · 可展示级演示（物理更可信版）

和 sim.py 的区别：
  1. 棋子之间会互相挤开（圆-圆碰撞），不再允许"车穿过棋堆"；
  2. 推板只有真正扫到棋子才推动，不再把 200mm 外的棋子瞬移到板前；
  3. 车体压到棋子会把棋子顶开、车被顶偏并降速；
  4. 推板带 10mm 侧挡边，进入铲口的棋子会被夹在 100mm 内槽里。

用法：
  python strategy/push_demo.py --layout spread --rule any --video
  python strategy/push_demo.py --layout pile   --rule own --seconds 90

演示定位：这是"机构 + 决策"验证，不是比赛成绩单。视频左上角会写明棋局条件。
"""
import argparse
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vision"))
import config  # noqa: E402

FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_QI = r"C:\Windows\Fonts\simkai.ttf"

RED_SPEC = [("帅", 1), ("仕", 2), ("相", 2), ("车", 2), ("马", 2), ("炮", 2), ("兵", 5)]
BLACK_SPEC = [("将", 1), ("士", 2), ("象", 2), ("车", 2), ("马", 2), ("炮", 2), ("卒", 5)]

GRID_ROWS = (600.0, 950.0, 1300.0, 1600.0)
GRID_COLS = (400.0, 1000.0, 1600.0, 2200.0)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def font(size):
    return ImageFont.truetype(FONT, size)


# ------------------------------------------------------------------ 棋子 / 世界
def make_pieces(layout="spread", seed=0):
    rng = random.Random(seed)
    out = []
    for side, spec in (("red", RED_SPEC), ("black", BLACK_SPEC)):
        for ch, k in spec:
            for _ in range(k):
                out.append({"side": side, "char": ch,
                            "value": config.VALUE.get(config.NORMALIZE.get(ch, ch), 0)})
    rng.shuffle(out)
    placed = []
    if layout == "grid":
        # 4x4 个独立取棋工位，每工位 2 枚；车在工位间走净空通道。
        # 这是"机构节拍 + 重复精度"演示，不代表开局堆态。
        rows, cols = GRID_ROWS, GRID_COLS
        reds = [p for p in out if p["side"] == "red"]
        blacks = [p for p in out if p["side"] == "black"]
        pairs = [[reds[2 * i], reds[2 * i + 1]] for i in range(8)]
        pairs += [[blacks[2 * i], blacks[2 * i + 1]] for i in range(8)]
        rng.shuffle(pairs)
        k = 0
        for y in rows:
            for x in cols:
                for j in (-1, 1):
                    p = pairs[k // 2][k % 2]
                    k += 1
                    placed.append({**p, "x": x, "y": y + j * 22.0,
                                   "r": config.PIECE_R, "scored": False, "id": len(placed)})
    elif layout == "lane":
        # 台架节拍演示：两条供棋通道，每对 2 枚，车在中间通道往返。
        # 这是"机构重复精度 + 节拍"演示，不代表开局堆态。
        lanes = (700.0, 1300.0)
        xs = [550.0 + i * 285.0 for i in range(8)]
        idx = 0
        for y in lanes:
            for x in reversed(xs):
                for j in (-1, 1):
                    p = out[idx]
                    idx += 1
                    placed.append({**p, "x": x, "y": y + j * 19.0,
                                   "r": config.PIECE_R, "scored": False, "id": len(placed)})
        # 同一对同色
        for i in range(0, len(placed), 2):
            a, b = placed[i], placed[i + 1]
            if a["side"] != b["side"]:
                a.update(side=b["side"], char=b["char"], value=b["value"])
    elif layout == "pile":
        # 官方图：32 枚随机投放在 600x500 的中央堆放区
        for p in out:
            for _ in range(20000):
                x = rng.uniform(config.PILE_CX - config.PILE_L / 2 + 24,
                                config.PILE_CX + config.PILE_L / 2 - 24)
                y = rng.uniform(config.PILE_CY - config.PILE_W / 2 + 24,
                                config.PILE_CY + config.PILE_W / 2 - 24)
                if all(math.hypot(x - q["x"], y - q["y"]) > 45.5 for q in placed):
                    placed.append({**p, "x": x, "y": y, "r": config.PIECE_R,
                                   "scored": False, "id": len(placed)})
                    break
    else:
        # 中局散布态：棋子已被撞散，但仍有单枚/两枚一组——用来验证"1~2 枚一趟"
        # 的机构重复精度，不代表开局就能这样摆。
        groups = []
        for i in range(16):
            for _ in range(4000):
                gx = rng.uniform(330, config.FIELD_L - 330)
                gy = rng.uniform(260, config.FIELD_W - 260)
                if all(math.hypot(gx - g[0], gy - g[1]) > 330 for g in groups):
                    groups.append((gx, gy))
                    break
        rng.shuffle(groups)
        for gi, (gx, gy) in enumerate(groups):
            a = rng.uniform(0, math.pi)
            off = rng.choice([0.0, 26.0])
            for j in range(2):
                p = out[gi * 2 + j]
                dx = math.cos(a) * off * (1 if j else -1)
                dy = math.sin(a) * off * (1 if j else -1)
                placed.append({**p, "x": gx + dx, "y": gy + dy, "r": config.PIECE_R,
                               "scored": False, "id": len(placed)})
        # 保证同组同色，方便"只推己方颜色"规则也能演示
        for i in range(0, len(placed), 2):
            a, b = placed[i], placed[i + 1]
            if a["side"] != b["side"]:
                a.update(side=b["side"], char=b["char"], value=b["value"])
    return placed


class PushWorld:
    def __init__(self, pieces, rule="any", penalty=1.0, rng=None):
        self.pieces = pieces
        self.rule = rule                 # any=公开赛题文本；own=只能推己方颜色
        self.penalty = penalty
        self.rng = rng or np.random.default_rng(0)
        self.score = 0
        self.delivered = 0
        self.wrong = 0
        self.events = []
        self.car_hits = 0

    @staticmethod
    def goal_center(side="red"):
        return (config.GOAL_DEPTH * 0.45, (config.GOAL_Y0 + config.GOAL_Y1) / 2)

    @staticmethod
    def in_goal(p):
        x0, x1 = 0.0, config.GOAL_DEPTH
        y0, y1 = config.GOAL_Y0, config.GOAL_Y1
        return (p["x"] - p["r"] >= x0 - 0.5 and p["x"] + p["r"] <= x1
                and p["y"] - p["r"] >= y0 and p["y"] + p["r"] <= y1)

    def _body_collide(self, car):
        """车体 OBB 压到棋子：沿最小穿透方向把棋子推出车体，并让车轻微顶偏。"""
        th = math.radians(car.h)
        c, s = math.cos(th), math.sin(th)
        hl, hw = config.BODY_L_MM / 2.0, config.BODY_W_MM / 2.0
        hits = 0
        for p in self.pieces:
            if p["scored"]:
                continue
            dx, dy = p["x"] - car.x, p["y"] - car.y
            along = c * dx + s * dy
            lat = -s * dx + c * dy
            if abs(along) > hl + p["r"] or abs(lat) > hw + p["r"]:
                continue
            pen_x = (hl + p["r"]) - abs(along)
            pen_y = (hw + p["r"]) - abs(lat)
            if pen_y < pen_x:
                lat = math.copysign(hw + p["r"] + 0.2, lat if lat else 1.0)
            else:
                along = math.copysign(hl + p["r"] + 0.2, along if along else 1.0)
            p["x"] = car.x + c * along - s * lat
            p["y"] = car.y + s * along + c * lat
            hits += 1
        if hits:
            self.car_hits += hits
            car.h += float(self.rng.normal(0, 0.035 * hits))
            car.x += float(self.rng.normal(0, 0.15 * hits))
            car.y += float(self.rng.normal(0, 0.15 * hits))
            car.slow = min(0.35, 0.06 * hits)
        else:
            car.slow = max(0.0, car.slow - 0.1)

    def _blade(self, car):
        """推板 + U 形挡边。只推动板口范围内的棋子，绝不瞬移远处的棋子。"""
        if not car.blade_down:
            return
        th = math.radians(car.h)
        f = (math.cos(th), math.sin(th))
        l = (-f[1], f[0])
        bx = config.BLADE_AHEAD_MM
        half = config.BLADE_WIDTH_MM / 2.0
        inner = half - config.GUIDE_MM
        for p in self.pieces:
            if p["scored"]:
                continue
            dx, dy = p["x"] - car.x, p["y"] - car.y
            along = dx * f[0] + dy * f[1]
            lat = dx * l[0] + dy * l[1]
            if abs(lat) > inner - p["r"]:
                continue
            # 推板前缘：棋子中心进到板前一个半径内，就被板面顶住
            if bx - p["r"] - 1.0 <= along <= bx + p["r"] + 1.0:
                along = bx + p["r"] + 1.0
                lat = max(-(inner - p["r"]), min(inner - p["r"], lat))
            # U 形挡边：已经在铲口里的棋子被两侧夹住
            elif bx - 62.0 <= along < bx - p["r"]:
                lat = max(-(inner - p["r"]), min(inner - p["r"], lat))
            else:
                continue
            p["x"] = car.x + f[0] * along + l[0] * lat
            p["y"] = car.y + f[1] * along + l[1] * lat
            p["touched"] = p.get("touched", 0) + 1

    def _piece_collide(self, iters=3):
        """棋子互相挤开。密集堆里推一枚必然带动邻近棋子——这正是真实约束。"""
        for _ in range(iters):
            moved = False
            n = len(self.pieces)
            for i in range(n):
                a = self.pieces[i]
                if a["scored"]:
                    continue
                for j in range(i + 1, n):
                    b = self.pieces[j]
                    if b["scored"]:
                        continue
                    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
                    d = math.hypot(dx, dy)
                    need = a["r"] + b["r"] + 0.4
                    if d >= need:
                        continue
                    if d < 1e-6:
                        dx, dy, d = 1.0, 0.0, 1.0
                    push = min((need - d) / 2.0, 2.0)
                    ux, uy = dx / d, dy / d
                    a["x"] -= ux * push
                    a["y"] -= uy * push
                    b["x"] += ux * push
                    b["y"] += uy * push
                    moved = True
            if not moved:
                break

    def update(self, car, t):
        car.step(0.02)
        self._body_collide(car)
        self._blade(car)
        self._piece_collide()
        for p in self.pieces:
            if p["scored"]:
                continue
            p["x"] = max(p["r"], min(config.FIELD_L - p["r"], p["x"]))
            p["y"] = max(p["r"], min(config.FIELD_W - p["r"], p["y"]))
            if not self.in_goal(p):
                continue
            own = p["side"] == "red"
            if self.rule == "own":
                pts = p["value"] if own else -p["value"] * self.penalty
            else:
                pts = p["value"]
            pts = int(round(pts))
            self.score += pts
            if pts > 0:
                self.delivered += 1
            else:
                self.wrong += 1
            p["scored"] = True
            p["pts"] = pts
            self.events.append((round(t, 1), p["side"], p["char"], pts))


# ------------------------------------------------------------------ 底盘
class Car:
    def __init__(self, x, y, h=0.0, v_max=1250.0, w_max=190.0, rng=None):
        self.x, self.y, self.h = float(x), float(y), float(h)
        self.v_max = v_max
        self.w_max = w_max
        self.task = None
        self.blade_down = False
        self.slow = 0.0
        self.odo = 0.0
        self.rng = rng or np.random.default_rng(0)
        self.last_cmd = None

    def pose(self):
        return (self.x, self.y, wrap(self.h))

    def done(self):
        return self.task is None

    def goto(self, x, y, h, tol=12.0):
        self.task = {"t": "goto", "x": x, "y": y, "h": wrap(h), "phase": "turn",
                     "tol": tol}

    def goto_soft(self, x, y, h):
        """航点跟随：到点不停车、不摆正，只把车头往航向拉，适合走折线路径。"""
        self.task = {"t": "goto", "x": x, "y": y, "h": wrap(h),
                     "phase": "drive", "soft": True}

    def drive(self, mm):
        self.task = {"t": "drive", "left": float(mm), "h0": wrap(self.h)}

    def turn(self, deg):
        self.task = {"t": "turn", "h": wrap(deg)}

    def blade(self, down):
        self.blade_down = bool(down)
        self.task = None

    def step(self, dt):
        if self.task is None:
            return
        t = self.task
        if t["t"] == "turn":
            err = wrap(t["h"] - self.h)
            if abs(err) < 1.2:
                self.h = t["h"]
                self.task = None
                return
            w = max(-self.w_max, min(self.w_max, err * 5.0))
            self.h = wrap(self.h + w * dt)
        elif t["t"] == "goto":
            if t["phase"] == "turn":
                bearing = math.degrees(math.atan2(t["y"] - self.y, t["x"] - self.x))
                err = wrap(bearing - self.h)
                if abs(err) < 1.5:
                    t["phase"] = "drive"
                    return
                w = max(-self.w_max, min(self.w_max, err * 4.0))
                self.h = wrap(self.h + w * dt)
            elif t["phase"] == "drive":
                dx, dy = t["x"] - self.x, t["y"] - self.y
                dist = math.hypot(dx, dy)
                if dist < (18.0 if t.get("soft") else t.get("tol", 12.0)):
                    if t.get("soft"):
                        self.task = None
                        return
                    t["phase"] = "align"
                    return
                bearing = math.degrees(math.atan2(dy, dx))
                err = wrap(bearing - self.h)
                if abs(err) > 8.0:
                    w = max(-self.w_max, min(self.w_max, err * 4.0))
                    self.h = wrap(self.h + w * dt)
                v = min(self.v_max, max(260.0, dist * 4.0)) * (1.0 - self.slow)
                step = min(v * dt, dist)
                self.x += math.cos(math.radians(self.h)) * step
                self.y += math.sin(math.radians(self.h)) * step
                self.odo += step
            else:
                err = wrap(t["h"] - self.h)
                if abs(err) < 1.2:
                    self.h = t["h"]
                    self.task = None
                    return
                w = max(-self.w_max, min(self.w_max, err * 4.0))
                self.h = wrap(self.h + w * dt)
        elif t["t"] == "drive":
            left = t["left"]
            if abs(left) < 1.0:
                self.task = None
                return
            # 航向保持：真实底盘有编码器/IMU，长推不会靠"运气"走直
            e = wrap(t["h0"] - self.h)
            self.h = wrap(self.h + max(-120.0, min(120.0, e * 8.0)) * dt)
            v = min(self.v_max, max(320.0, abs(left) * 5.0)) * (1.0 - self.slow)
            step = math.copysign(min(v * dt, abs(left)), left)
            self.x += math.cos(math.radians(self.h)) * step
            self.y += math.sin(math.radians(self.h)) * step
            t["left"] -= step
            self.odo += abs(step)
        self.x = max(40.0, min(config.FIELD_L - 40.0, self.x))
        self.y = max(40.0, min(config.FIELD_W - 40.0, self.y))


# ------------------------------------------------------------------ 决策：外缘剥离
class Peeler:
    """只从"车能直线进入、且不蹭到别的棋子"的外缘取 1~2 枚，再直推入球门。"""

    def __init__(self, world, car, side="red", rule="any", debug=False):
        self.world = world
        self.car = car
        self.side = side
        self.rule = rule
        self.phase = "SELECT"
        self.target = []
        self.plan = None
        self.trip = 0
        self.phase_t = 0.0
        self.t = 0.0
        self.last_target_ids = set()
        self.fail_count = 0
        self.debug = debug
        self.log = []
        self.arrived = 0.0
        self.pushed = 0.0
        self.leg_start = (car.x, car.y)
        self.path = []
        self.path_i = 0
        self.retreat_issued = False

    def set_phase(self, ph):
        self.phase = ph
        self.phase_t = self.t
        self.leg_start = (self.car.x, self.car.y)

    def corridor_clear(self, p0, p1, half_w, ignore):
        ux, uy = p1[0] - p0[0], p1[1] - p0[1]
        L = math.hypot(ux, uy)
        if L < 1:
            return True
        ux, uy = ux / L, uy / L
        for q in self.world.pieces:
            if q["scored"] or q["id"] in ignore:
                continue
            dx, dy = q["x"] - p0[0], q["y"] - p0[1]
            along = dx * ux + dy * uy
            if along < -q["r"] or along > L + q["r"]:
                continue
            lat = abs(-dx * uy + dy * ux)
            if lat < half_w + q["r"]:
                return False
        return True

    def astar(self, start, goal, ignore_ids, cell=40.0, inflate=96.0):
        """50mm 栅格 A*：把棋子当成膨胀障碍，绕开而不是一路顶过去。"""
        nx = int(config.FIELD_L / cell) + 1
        ny = int(config.FIELD_W / cell) + 1
        sx = max(1, min(nx - 2, int(round(start[0] / cell))))
        sy = max(1, min(ny - 2, int(round(start[1] / cell))))
        gx = max(1, min(nx - 2, int(round(goal[0] / cell))))
        gy = max(1, min(ny - 2, int(round(goal[1] / cell))))
        occ = np.zeros((nx, ny), dtype=bool)
        for q in self.world.pieces:
            if q["scored"] or q["id"] in ignore_ids:
                continue
            r = inflate + q["r"]
            x0 = max(0, int((q["x"] - r) / cell) - 1)
            x1 = min(nx - 1, int((q["x"] + r) / cell) + 1)
            y0 = max(0, int((q["y"] - r) / cell) - 1)
            y1 = min(ny - 1, int((q["y"] + r) / cell) + 1)
            for ix in range(x0, x1 + 1):
                for iy in range(y0, y1 + 1):
                    if math.hypot(ix * cell - q["x"], iy * cell - q["y"]) < r:
                        occ[ix, iy] = True
        # 起点/终点附近强行放空，否则会"没有路径"
        for ix in range(max(0, sx - 2), min(nx, sx + 3)):
            for iy in range(max(0, sy - 2), min(ny, sy + 3)):
                occ[ix, iy] = False
        for ix in range(max(0, gx - 3), min(nx, gx + 4)):
            for iy in range(max(0, gy - 3), min(ny, gy + 4)):
                occ[ix, iy] = False
        if occ[gx, gy]:
            return None
        import heapq
        openh = [(0.0, (sx, sy))]
        came = {}
        gscore = {(sx, sy): 0.0}
        closed = set()
        while openh:
            _f, cur = heapq.heappop(openh)
            if cur in closed:
                continue
            if cur == (gx, gy):
                path = []
                while cur in came:
                    path.append((cur[0] * cell, cur[1] * cell))
                    cur = came[cur]
                path.append((sx * cell, sy * cell))
                path.reverse()
                # 抽稀：每 3 个点取一个，最后补上终点
                out = path[::3]
                if out[-1] != path[-1]:
                    out.append(path[-1])
                return out
            closed.add(cur)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nn = (cur[0] + dx, cur[1] + dy)
                    if not (0 <= nn[0] < nx and 0 <= nn[1] < ny) or occ[nn]:
                        continue
                    step = math.hypot(dx, dy)
                    ng = gscore[cur] + step
                    if ng >= gscore.get(nn, 1e18):
                        continue
                    gscore[nn] = ng
                    came[nn] = cur
                    h = math.hypot(nn[0] - gx, nn[1] - gy)
                    heapq.heappush(openh, (ng + h, nn))
        return None

    def select(self):
        car = self.car
        goal = PushWorld.goal_center(self.side)
        best = None
        for p in self.world.pieces:
            if p["scored"]:
                continue
            if self.rule == "own" and p["side"] != self.side:
                continue
            gx, gy = goal
            base = math.atan2(gy - p["y"], gx - p["x"])
            # 允许 ±18° 的推送方向微调，绕开同排棋子
            for dth in (-18, -9, 0, 9, 18):
                u = math.radians(math.degrees(base) + dth)
                ux, uy = math.cos(u), math.sin(u)
                contact = (p["x"] - ux * (config.BLADE_AHEAD_MM + p["r"] + 6.0),
                           p["y"] - uy * (config.BLADE_AHEAD_MM + p["r"] + 6.0))
                start = (contact[0] - ux * 430.0, contact[1] - uy * 430.0)
                if not (80 < start[0] < config.FIELD_L - 80
                        and 80 < start[1] < config.FIELD_W - 80):
                    continue
                if not self.corridor_clear(start, contact, 108.0, {p["id"]}):
                    continue
                # 推送走廊里最好只有目标这一枚；多一枚同色也可以（并排两枚）
                extra, bad = [], []
                for q in self.world.pieces:
                    if q["scored"] or q["id"] == p["id"]:
                        continue
                    dx, dy = q["x"] - car.x, q["y"] - car.y  # 近似用整条推到门的走廊
                    # 点到推送直线的距离
                    vx, vy = gx - p["x"], gy - p["y"]
                    vl = math.hypot(vx, vy) or 1.0
                    vx, vy = vx / vl, vy / vl
                    rx, ry = q["x"] - p["x"], q["y"] - p["y"]
                    along = rx * vx + ry * vy
                    lat = abs(-rx * vy + ry * vx)
                    if -30 < along < vl and lat < 46.0 + q["r"]:
                        same = (q["side"] == p["side"])
                        if self.rule == "own" and not same:
                            bad.append(q)
                        else:
                            extra.append(q)
                if bad:
                    continue
                if len(extra) >= 2:
                    continue
                cost = (math.hypot(p["x"] - car.x, p["y"] - car.y)
                        + 2.2 * abs(dth) + 90.0 * len(extra)
                        + (120.0 if p["id"] in self.last_target_ids else 0.0))
                if best is None or cost < best[0]:
                    best = (cost, p, u, contact, start, extra)
        if best is None:
            return None
        _cost, p, u, contact, start, extra = best
        self.target = [p] + extra[:1]
        self.plan = {"u": u, "contact": contact, "start": start,
                     "goal": goal, "ids": {m["id"] for m in self.target}}
        return self.plan

    def update(self, t):
        self.t = t
        car = self.car
        self.pushed = math.hypot(car.x - self.leg_start[0], car.y - self.leg_start[1])
        if self.phase == "SELECT":
            if not self.select():
                self.set_phase("DONE")
                return
            sx, sy = self.plan["start"]
            # 规划去取棋位的路线时，目标棋子本身也是障碍：不能从它身上碾过去
            self.path = self.astar((car.x, car.y), (sx, sy), set())
            if not self.path:
                self.last_target_ids = set(self.plan["ids"])
                self.plan = None
                self.target = []
                self.set_phase("SELECT")
                return
            self.path.append((sx, sy))
            self.path_i = 0
            self.set_phase("FOLLOW")
            if len(self.path) == 1:
                car.goto(self.path[0][0], self.path[0][1], math.degrees(self.plan["u"]))
            else:
                car.goto_soft(self.path[0][0], self.path[0][1], math.degrees(self.plan["u"]))
            return
        if self.phase == "FOLLOW":
            if car.done():
                self.path_i += 1
                if self.path_i >= len(self.path):
                    if abs(wrap(math.degrees(self.plan["u"]) - car.h)) > 4.0:
                        car.turn(math.degrees(self.plan["u"]))
                        self.set_phase("AIM")
                    else:
                        self.set_phase("CREEP")
                        cx, cy = self.plan["contact"]
                        car.goto(cx, cy, math.degrees(self.plan["u"]))
                else:
                    if self.path_i == len(self.path) - 1:
                        car.goto(self.path[self.path_i][0], self.path[self.path_i][1],
                                 math.degrees(self.plan["u"]))
                    else:
                        car.goto_soft(self.path[self.path_i][0], self.path[self.path_i][1],
                                      math.degrees(self.plan["u"]))
            elif t - self.phase_t > 24.0:
                self.set_phase("FAIL")
            return
        if self.phase == "AIM":
            if car.done():
                self.set_phase("CREEP")
                cx, cy = self.plan["contact"]
                car.goto(cx, cy, math.degrees(self.plan["u"]))
            return
        if self.phase == "CREEP":
            if car.done():
                car.blade(True)
                self.set_phase("LOWER")
            return
        if self.phase == "LOWER":
            # 到位后再落板，往前进 70mm 让板面真正贴住棋子（不是瞬移）
            car.blade(True)
            car.drive(70.0)
            self.set_phase("ENGAGE")
            return
        if self.phase == "ENGAGE":
            if car.done():
                live = [q for q in self.target if not q["scored"]]
                # 目标必须真的在铲口里
                ok = False
                th = math.radians(car.h)
                f = (math.cos(th), math.sin(th))
                l = (-f[1], f[0])
                for q in live:
                    dx, dy = q["x"] - car.x, q["y"] - car.y
                    along = dx * f[0] + dy * f[1]
                    lat = abs(dx * l[0] + dy * l[1])
                    if 130 < along < 280 and lat < 52:
                        ok = True
                if not ok:
                    self.set_phase("FAIL")
                    return
                self.set_phase("PUSH")
                car.drive(260.0)
            return
        if self.phase == "PUSH":
            if car.done():
                self.set_phase("CHECK")
            return
        if self.phase == "CHECK":
            # 每 260mm 复核一次：目标还在板口就继续，掉了就倒车重来
            live = [q for q in self.target if not q["scored"]]
            if not live:
                self.set_phase("RETREAT")
                return
            th = math.radians(car.h)
            f = (math.cos(th), math.sin(th))
            l = (-f[1], f[0])
            best = None
            for q in live:
                dx, dy = q["x"] - car.x, q["y"] - car.y
                along = dx * f[0] + dy * f[1]
                lat = abs(dx * l[0] + dy * l[1])
                if 120 < along < 320 and lat < 55:
                    if best is None or along > best[0]:
                        best = (along, q)
            if best is None:
                self.set_phase("FAIL")
                return
            q = best[1]
            dist_to_goal = math.hypot(goal_x(q), goal_y(q))
            if dist_to_goal < 210:
                self.set_phase("DEPOSIT")
                car.drive(min(360.0, dist_to_goal + 80.0))
            else:
                # 直推：初始朝向已经对准球门，中途不再原地转（转会把棋子甩出铲口）
                car.drive(260.0)
                self.set_phase("PUSH")
            return
        if self.phase == "DEPOSIT":
            if car.done():
                self.set_phase("RETREAT")
                self.retreat_issued = True
                car.blade(False)
                car.drive(-240.0)
            return
        if self.phase == "RETREAT":
            if car.done():
                if not self.retreat_issued:
                    self.retreat_issued = True
                    car.blade(False)
                    car.drive(-240.0)
                    return
                self.retreat_issued = False
                self.trip += 1
                got = [q for q in self.target if q["scored"]]
                self.log.append((round(t, 1), self.trip, len(got),
                                 [q["char"] for q in self.target]))
                self.last_target_ids = set(self.plan["ids"])
                self.target = []
                self.plan = None
                self.set_phase("SELECT")
            return
        if self.phase == "FAIL":
            car.blade(False)
            self.fail_count += 1
            if car.done():
                car.drive(-180.0)
                self.set_phase("BACKOFF")
            return
        if self.phase == "BACKOFF":
            if car.done():
                self.plan = None
                self.target = []
                self.retreat_issued = False
                self.set_phase("SELECT")
            return
        if self.phase == "DONE":
            return

    def status(self):
        return {"phase": self.phase, "trip": self.trip, "target": self.target}


def goal_x(p):
    return config.GOAL_DEPTH * 0.45 - p["x"]


def goal_y(p):
    return (config.GOAL_Y0 + config.GOAL_Y1) / 2 - p["y"]


# ------------------------------------------------------------------ 台架节拍演示
class LaneFeeder:
    """双通道供棋：车在 y=1000 的中间通道往返，从右往左逐对推入球门。

    这是"1~2 枚/趟"的机构重复精度 + 节拍演示。棋子按两条供棋通道预置，
    不代表官方开局的密集堆放态；堆态结论见 docs/窄推与夹取对比.md。
    """

    def __init__(self, world, car, side="red", rule="any"):
        self.world = world
        self.car = car
        self.side = side
        self.rule = rule
        self.phase = "NEXT"
        self.phase_t = 0.0
        self.t = 0.0
        self.i = 0
        self.trip = 0
        self.fail_count = 0
        self.target = []
        self.pairs = []
        # 按供棋顺序（先上通道，右→左）两枚一组
        for lane in (700.0, 1300.0):
            row = [p for p in world.pieces if abs(p["y"] - lane) < 40]
            row.sort(key=lambda q: -q["x"])
            for k in range(0, len(row), 2):
                self.pairs.append(row[k:k + 2])
        self.x = 0.0
        self.y = 0.0

    def set_phase(self, ph):
        self.phase = ph
        self.phase_t = self.t

    def update(self, t):
        self.t = t
        car = self.car
        if self.phase == "NEXT":
            if self.i >= len(self.pairs):
                self.set_phase("DONE")
                return
            self.target = self.pairs[self.i]
            if all(p["scored"] for p in self.target):
                self.i += 1
                return
            self.x = sum(p["x"] for p in self.target) / len(self.target)
            self.y = sum(p["y"] for p in self.target) / len(self.target)
            self.set_phase("TO_CORRIDOR")
            car.goto(self.x + 340.0, 1000.0, 0.0)
            return
        if self.phase == "TO_CORRIDOR":
            if car.done():
                self.set_phase("TO_ENTRY")
                car.goto(self.x + 340.0, self.y, 180.0, tol=5.0)
            return
        if self.phase == "TO_ENTRY":
            if car.done():
                self.set_phase("ALIGN")
                car.turn(180.0)
            return
        if self.phase == "ALIGN":
            if car.done():
                car.blade(True)
                self.set_phase("LOWER")
                car.drive(185.0)
            return
        if self.phase == "LOWER":
            if car.done():
                ok = []
                th = math.radians(car.h)
                f = (math.cos(th), math.sin(th))
                l = (-f[1], f[0])
                for q in self.target:
                    dx, dy = q["x"] - car.x, q["y"] - car.y
                    along = dx * f[0] + dy * f[1]
                    lat = abs(dx * l[0] + dy * l[1])
                    if 130 < along < 280 and lat < 50:
                        ok.append(q)
                if len(ok) < 2:
                    self.set_phase("FAIL")
                    return
                self.set_phase("PUSH")
                car.drive(max(100.0, car.x - 278.0))
            return
        if self.phase == "PUSH":
            if car.done():
                car.blade(False)
                self.set_phase("RAISE")
                car.goto(car.x, 1000.0, 0.0)
            return
        if self.phase == "RAISE":
            if car.done():
                self.trip += 1
                self.i += 1
                self.set_phase("NEXT")
            return
        if self.phase == "BACK_CORRIDOR":
            if car.done():
                self.set_phase("NEXT")
            return
        if self.phase == "FAIL":
            car.blade(False)
            self.set_phase("BACKOFF")
            car.drive(-320.0)
            return
        if self.phase == "BACKOFF":
            if car.done():
                self.fail_count += 1
                self.i += 1
                self.set_phase("NEXT")
            return
        if self.phase == "DONE":
            car.blade(False)
            return

    def status(self):
        return {"phase": self.phase, "trip": self.trip, "target": self.target}


class GridFeeder:
    """4x4 独立取棋工位：每工位一对棋子，车从净空通道进入、推完退回通道。

    和 LaneFeeder 的区别：工位之间横向 500mm、纵向 350mm，推一个工位不会
    穿过另一个工位，所以"每趟 2 枚"是可以重复的；代价是需要预先把棋子
    分到工位里（比赛中由对抗撞散或由供棋机构完成）。
    """

    ROWS = GRID_ROWS
    COLS = GRID_COLS

    def __init__(self, world, car, side="red", rule="any"):
        self.world = world
        self.car = car
        self.side = side
        self.rule = rule
        self.phase = "NEXT"
        self.phase_t = 0.0
        self.t = 0.0
        self.cells = [(x, y) for y in self.ROWS for x in self.COLS]
        self.i = 0
        self.trip = 0
        self.fail_count = 0
        self.target = []
        self.x = self.y = 0.0
        self.wps = []
        self.wp_i = 0
        self.log = []
        self.finish_tries = 0

    def set_phase(self, ph):
        self.phase = ph
        self.phase_t = self.t

    def corridor_of(self, y):
        return y - 175.0

    def update(self, t):
        self.t = t
        car = self.car
        if self.phase == "NEXT":
            if self.i >= len(self.cells):
                self.set_phase("DONE")
                return
            self.x, self.y = self.cells[self.i]
            self.target = [p for p in self.world.pieces
                           if not p["scored"] and abs(p["x"] - self.x) < 1
                           and abs(p["y"] - self.y) < 30]
            if self.rule == "own":
                self.target = [p for p in self.target if p["side"] == self.side]
            if len(self.target) < 2:
                self.i += 1
                return
            corr = self.corridor_of(self.y)
            self.wps = [(car.x, corr, 0.0, 12.0),
                        (self.x + 340.0, corr, 0.0, 12.0),
                        (self.x + 340.0, self.y, 180.0, 5.0)]
            self.wp_i = 0
            self.set_phase("TRAVEL")
            car.goto(*self.wps[0])
            return
        if self.phase == "TRAVEL":
            if car.done():
                self.wp_i += 1
                if self.wp_i >= len(self.wps):
                    self.set_phase("ALIGN")
                    car.turn(180.0)
                else:
                    car.goto(*self.wps[self.wp_i])
            elif t - self.phase_t > 22.0:
                self.set_phase("FAIL")
            return
        if self.phase == "ALIGN":
            if car.done():
                car.blade(True)
                self.set_phase("LOWER")
                car.drive(185.0)
            return
        if self.phase == "LOWER":
            if car.done():
                ok = 0
                th = math.radians(car.h)
                f = (math.cos(th), math.sin(th))
                l = (-f[1], f[0])
                for q in self.target:
                    dx, dy = q["x"] - car.x, q["y"] - car.y
                    along = dx * f[0] + dy * f[1]
                    lat = abs(dx * l[0] + dy * l[1])
                    if 130 < along < 280 and lat < 48:
                        ok += 1
                if ok < 2:
                    self.set_phase("FAIL")
                    return
                self.set_phase("PUSH")
                car.drive(max(100.0, car.x - 278.0))
            return
        if self.phase == "PUSH":
            if car.done():
                self.finish_tries = 0
                self.set_phase("VERIFY")
            return
        if self.phase == "VERIFY":
            pending = [q for q in self.target if not q["scored"]]
            if not pending:
                car.blade(False)
                self.set_phase("RETURN")
                car.goto(car.x, self.corridor_of(self.y), 0.0)
                return
            # 没完全进区：只要棋子还在板口，就再补一小段（真车也是视觉复核后再补推）
            if self.finish_tries >= 3:
                car.blade(False)
                self.set_phase("RETURN")
                car.goto(car.x, self.corridor_of(self.y), 0.0)
                return
            if car.done():
                th = math.radians(car.h)
                f = (math.cos(th), math.sin(th))
                l = (-f[1], f[0])
                in_mouth = False
                for q in pending:
                    dx, dy = q["x"] - car.x, q["y"] - car.y
                    along = dx * f[0] + dy * f[1]
                    lat = abs(dx * l[0] + dy * l[1])
                    if 95 < along < 300 and lat < 50:
                        in_mouth = True
                if in_mouth and car.x > 90:
                    self.finish_tries += 1
                    car.drive(110.0)
                else:
                    car.blade(False)
                    self.set_phase("RETURN")
                    car.goto(car.x, self.corridor_of(self.y), 0.0)
            return
        if self.phase == "RETURN":
            if car.done():
                got = sum(1 for q in self.target if q["scored"])
                if got < 2:
                    print(f"  [少收] 工位({self.x:.0f},{self.y:.0f}) 只进 {got}；"
                          + " ".join(f"{q['char']}({q['x']:.0f},{q['y']:.0f})"
                                     for q in self.target)
                          + f" 车=({car.x:.0f},{car.y:.0f},{car.h:.0f})")
                self.trip += 1
                self.log.append((round(t, 1), self.trip, got,
                                 [q["char"] for q in self.target]))
                self.i += 1
                self.set_phase("NEXT")
            return
        if self.phase == "FAIL":
            car.blade(False)
            self.set_phase("BACKOFF")
            car.drive(-300.0)
            return
        if self.phase == "BACKOFF":
            if car.done():
                self.fail_count += 1
                self.i += 1
                self.set_phase("NEXT")
            return
        if self.phase == "DONE":
            car.blade(False)
            return

    def status(self):
        return {"phase": self.phase, "trip": self.trip, "target": self.target}


# ------------------------------------------------------------------ 渲染
S = 0.40
PAD_L, PAD_T = 60, 118
W = int(config.FIELD_L * S) + 2 * PAD_L + 500
H = int(config.FIELD_W * S) + PAD_T + 104


class DemoRenderer:
    def __init__(self, out=None, fps=25, title="", note="", rule="any", layout="grid"):
        self.path = Path(out) if out else None
        self.vw = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (W, H))
        self.title = title
        self.note = note
        self.rule = rule
        self.layout = layout
        self.tiles = {}
        self.frames = 0
        self.bg = self._bg()

    def _bg(self):
        img = Image.new("RGB", (W, H), (246, 245, 240))
        d = ImageDraw.Draw(img)
        x0, y0 = PAD_L, PAD_T
        x1 = x0 + config.FIELD_L * S
        y1 = y0 + config.FIELD_W * S
        d.rectangle([x0 - 10, y0 - 10, x1 + 10, y1 + 10], fill=(66, 68, 70))
        d.rectangle([x0, y0, x1, y1], fill=(226, 226, 218))
        # 垫子纹理：100mm 细网格
        for mm in range(0, int(config.FIELD_L) + 1, 100):
            xx = x0 + mm * S
            d.line([(xx, y0), (xx, y1)], fill=(219, 219, 211), width=1)
        for mm in range(0, int(config.FIELD_W) + 1, 100):
            yy = y0 + mm * S
            d.line([(x0, yy), (x1, yy)], fill=(219, 219, 211), width=1)
        # 球门区
        gx0, gy0, gx1, gy1 = x0, y0 + config.GOAL_Y0 * S, x0 + config.GOAL_DEPTH * S, y0 + config.GOAL_Y1 * S
        d.rectangle([gx0, gy0, gx1, gy1], fill=(236, 214, 210), outline=(178, 96, 88), width=2)
        for k in range(6):
            yy = gy0 + 12 + k * 55
            d.line([(gx0 + 4, yy), (gx1 - 4, yy)], fill=(226, 196, 192), width=3)
        d.text((gx0 + 4, gy0 + 6), "红队\n球门区", font=font(11), fill=(150, 70, 62), spacing=2)
        gx0b = x1 - config.GOAL_DEPTH * S
        d.rectangle([gx0b, gy0, x1, gy1], fill=(208, 214, 224), outline=(96, 112, 150), width=2)
        d.text((x1 - 8, gy0 + 8), "黑队球门", font=font(11), fill=(70, 84, 120), anchor="ra")
        if self.layout == "pile":
            px0 = x0 + (config.PILE_CX - config.PILE_L / 2) * S
            px1 = x0 + (config.PILE_CX + config.PILE_L / 2) * S
            py0 = y0 + (config.PILE_CY - config.PILE_W / 2) * S
            py1 = y0 + (config.PILE_CY + config.PILE_W / 2) * S
            for k in range(0, int(px1 - px0), 12):
                d.line([(px0 + k, py0), (min(px0 + k + 6, px1), py0)], fill=(150, 172, 152), width=1)
                d.line([(px0 + k, py1), (min(px0 + k + 6, px1), py1)], fill=(150, 172, 152), width=1)
            d.line([(px0, py0), (px0, py1)], fill=(150, 172, 152), width=1)
            d.line([(px1, py0), (px1, py1)], fill=(150, 172, 152), width=1)
            d.text((px0 + 8, py1 - 22), "棋子堆放区 600×500", font=font(13), fill=(110, 140, 112))
        else:
            # 16 个取棋工位：虚线框 + 编号（已取的会被棋子盖住，视觉上自然）
            k = 0
            for yy in GRID_ROWS:
                for xx in GRID_COLS:
                    k += 1
                    cx, cy = x0 + xx * S, y0 + yy * S
                    d.rectangle([cx - 42, cy - 42, cx + 42, cy + 42],
                                outline=(196, 200, 192), width=1)
                    d.text((cx - 38, cy + 28), f"{k:02d}", font=font(10), fill=(178, 182, 176))
            d.text((x0 + 10, y0 + (config.FIELD_W) * S - 24),
                   "16 个独立取棋工位（每工位 2 枚）· 车走工位间净空通道",
                   font=font(13), fill=(110, 140, 112))
        # 发车区
        sx0, sy0 = x0, y0 + (config.FIELD_W - config.START_W) * S
        sx1, sy1 = x0 + config.START_L * S, y0 + config.FIELD_W * S
        d.rectangle([sx0, sy0, sx1, sy1], outline=(160, 170, 160), width=1)
        d.text((sx0 + 10, sy0 + 8), "红队发车区", font=font(13), fill=(120, 140, 120))
        # 标题
        d.text((PAD_L, 18), "智能车光电象棋 · 窄推机构演示", font=font(24), fill=(28, 28, 30))
        d.text((PAD_L, 52), self.title, font=font(17), fill=(70, 72, 76))
        d.text((PAD_L, 80), self.note, font=font(14), fill=(120, 120, 124))
        return img

    def tile(self, char, side):
        key = (char, side)
        if key in self.tiles:
            return self.tiles[key]
        dia = int(config.PIECE_DIA * S)
        big = dia * 4
        img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        c, r = big // 2, big // 2 - 3
        d.ellipse([c - r, c - r, c + r, c + r], fill=(232, 218, 196), outline=(120, 94, 68), width=max(2, big // 28))
        d.ellipse([c - r * 0.87, c - r * 0.87, c + r * 0.87, c + r * 0.87],
                  outline=(176, 148, 112), width=max(1, big // 64))
        d.ellipse([c - r * 0.78, c - r * 0.78, c + r * 0.78, c + r * 0.78],
                  outline=(206, 186, 156), width=max(1, big // 90))
        f = ImageFont.truetype(FONT_QI, int(big * 0.56))
        ink = (188, 36, 42) if side == "red" else (34, 34, 38)
        bb = d.textbbox((0, 0), char, font=f)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        d.text(((big - w) / 2 - bb[0], (big - h) / 2 - bb[1]), char, font=f, fill=ink)
        tile = img.resize((dia, dia), Image.LANCZOS)
        self.tiles[key] = tile
        return tile

    def px(self, x, y):
        return (PAD_L + x * S, PAD_T + y * S)

    def __call__(self, world, car, ctrl, t):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        # 目标高亮
        if ctrl and ctrl.target:
            for q in ctrl.target:
                if q["scored"]:
                    continue
                cx, cy = self.px(q["x"], q["y"])
                rr = 20
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(255, 196, 30), width=3)
        # 棋子 + 投影
        for p in world.pieces:
            if p["scored"]:
                continue
            cx, cy = self.px(p["x"], p["y"])
            tile = self.tile(p["char"], p["side"])
            d.ellipse([cx - tile.width * 0.48 + 2, cy - tile.height * 0.46 + 3,
                       cx + tile.width * 0.48 + 2, cy + tile.height * 0.46 + 3],
                      fill=(176, 176, 168))
            img.paste(tile, (int(cx - tile.width / 2), int(cy - tile.height / 2)), tile)
        # 车
        th = math.radians(car.h)
        f = (math.cos(th), math.sin(th))
        l = (-f[1], f[0])
        hl, hw = config.BODY_L_MM / 2, config.BODY_W_MM / 2
        corners = []
        for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
            corners.append(self.px(car.x + f[0] * a + l[0] * b, car.y + f[1] * a + l[1] * b))
        d.polygon(corners, fill=(52, 120, 156), outline=(20, 50, 70), width=2)
        # 车轮
        for sgn in (-1, 1):
            wx = car.x - f[0] * 40 + l[0] * sgn * 88
            wy = car.y - f[1] * 40 + l[1] * sgn * 88
            a0 = self.px(wx - f[0] * 30, wy - f[1] * 30)
            a1 = self.px(wx + f[0] * 30, wy + f[1] * 30)
            d.line([a0, a1], fill=(34, 34, 36), width=10)
        # 推板 + 挡边
        bh = config.BLADE_WIDTH_MM / 2
        if car.blade_down:
            b0 = self.px(car.x + f[0] * config.BLADE_AHEAD_MM + l[0] * bh,
                         car.y + f[1] * config.BLADE_AHEAD_MM + l[1] * bh)
            b1 = self.px(car.x + f[0] * config.BLADE_AHEAD_MM - l[0] * bh,
                         car.y + f[1] * config.BLADE_AHEAD_MM - l[1] * bh)
            d.line([b0, b1], fill=(236, 170, 70), width=6)
            for sgn in (-1, 1):
                g0 = self.px(car.x + f[0] * (config.BLADE_AHEAD_MM - 62) + l[0] * sgn * bh,
                             car.y + f[1] * (config.BLADE_AHEAD_MM - 62) + l[1] * sgn * bh)
                g1 = self.px(car.x + f[0] * (config.BLADE_AHEAD_MM + 6) + l[0] * sgn * bh,
                             car.y + f[1] * (config.BLADE_AHEAD_MM + 6) + l[1] * sgn * bh)
                d.line([g0, g1], fill=(210, 140, 50), width=4)
        else:
            # 收起态：推板缩在车头前，用两根摆臂连到车体，避免看起来"飘在空中"
            b0 = self.px(car.x + f[0] * 112 + l[0] * 34, car.y + f[1] * 112 + l[1] * 34)
            b1 = self.px(car.x + f[0] * 112 - l[0] * 34, car.y + f[1] * 112 - l[1] * 34)
            d.line([b0, b1], fill=(214, 148, 56), width=4)
            for sgn in (-1, 1):
                a0 = self.px(car.x + f[0] * 86 + l[0] * sgn * 30,
                             car.y + f[1] * 86 + l[1] * sgn * 30)
                a1 = self.px(car.x + f[0] * 112 + l[0] * sgn * 30,
                             car.y + f[1] * 112 + l[1] * sgn * 30)
                d.line([a0, a1], fill=(160, 116, 52), width=3)
        # 车头指示
        tip = self.px(car.x + f[0] * 70, car.y + f[1] * 70)
        d.ellipse([tip[0] - 3, tip[1] - 3, tip[0] + 3, tip[1] + 3], fill=(255, 255, 255))
        # ---- 右侧仪表板
        x1 = PAD_L + config.FIELD_L * S
        px0 = int(x1 + 30)
        d.rectangle([x1 + 12, 0, W, H], fill=(251, 250, 246))
        d.text((px0, 14), "回收进度", font=font(13), fill=(110, 112, 116))
        d.text((px0, 32), f"{world.delivered:2d}", font=font(46), fill=(30, 30, 32))
        d.text((px0 + 62, 48), "/ 32 枚", font=font(20), fill=(110, 112, 116))
        d.text((px0 + 240, 34), f"{world.score}", font=font(38), fill=(176, 44, 44))
        d.text((px0 + 240, 78), "累计得分", font=font(12), fill=(130, 110, 110))
        d.text((px0, 88), f"趟数 {ctrl.trip if ctrl else 0}  ·  误推 {world.wrong}  "
                         f"·  {'任意棋子按面值' if self.rule == 'any' else '只计己方颜色'}",
               font=font(12), fill=(100, 102, 106))
        bx0, bx1 = px0, px0 + 440
        d.rectangle([bx0, 108, bx1, 114], fill=(214, 214, 208))
        d.rectangle([bx0, 108, bx0 + (bx1 - bx0) * world.delivered / 32.0, 114],
                    fill=(206, 152, 48))
        # 车头特写：从同一张画面裁一块放大，等于"机器人相机看到的机构细节"
        crop_w, crop_h = 760.0, 500.0
        sw, sh = crop_w * S, crop_h * S
        ccx = PAD_L + (car.x + f[0] * 40) * S
        ccy = PAD_T + (car.y + f[1] * 40) * S
        ccx = max(PAD_L + sw / 2 + 2, min(PAD_L + config.FIELD_L * S - sw / 2 - 2, ccx))
        ccy = max(PAD_T + sh / 2 + 2, min(PAD_T + config.FIELD_W * S - sh / 2 - 2, ccy))
        inset = img.crop((int(ccx - sw / 2), int(ccy - sh / 2),
                          int(ccx + sw / 2), int(ccy + sh / 2)))
        iw = 440
        inset = inset.resize((iw, int(iw * crop_h / crop_w)), Image.LANCZOS)
        iy = 142
        d.text((px0, iy - 22), "车头特写（推板 / 铲口）", font=font(13), fill=(70, 72, 76))
        img.paste(inset, (px0, iy))
        d.rectangle([px0, iy, px0 + iw, iy + inset.height], outline=(84, 86, 90), width=2)
        # 状态与最近几趟
        sy = iy + inset.height + 16
        names = {"SELECT": "挑选外缘目标", "APPROACH": "绕到取棋位", "AIM": "摆正车头",
                 "LOWER": "放下推板", "ENGAGE": "贴住棋子", "PUSH": "直推",
                 "CHECK": "复核铲口", "REAIM": "修正航向", "DEPOSIT": "推入球门",
                 "RETREAT": "退车", "FAIL": "目标丢失", "BACKOFF": "倒车重来",
                 "NEXT": "选下一个工位", "TRAVEL": "走净空通道", "ALIGN": "摆正车头",
                 "VERIFY": "复核是否整体进区", "RETURN": "退回通道",
                 "DONE": "本场结束"}
        st = names.get(ctrl.phase, ctrl.phase) if ctrl else "-"
        d.text((px0, sy), f"t = {t:5.1f}s", font=font(13), fill=(110, 112, 116))
        d.text((px0, sy + 22), f"状态：{st}", font=font(17), fill=(54, 78, 104))
        if ctrl and ctrl.target:
            tg = " / ".join(f"{'红' if q['side'] == 'red' else '黑'}{q['char']}"
                            for q in ctrl.target)
            d.text((px0, sy + 48), f"本趟：{tg}", font=font(14), fill=(130, 96, 40))
        d.text((px0, sy + 76), "最近几趟", font=font(12), fill=(130, 130, 134))
        logs = getattr(ctrl, "log", [])[-5:]
        yy = sy + 96
        for row in logs:
            d.text((px0, yy), f"第 {row[1]:2d} 趟   +{row[2]} 枚   {'/'.join(row[3])}",
                   font=font(12), fill=(96, 98, 102))
            yy += 20
        if world.events:
            e = world.events[-1]
            d.text((PAD_L, PAD_T + config.FIELD_W * S + 22),
                   f"t = {t:5.1f}s    最近进区：{'红' if e[1] == 'red' else '黑'}{e[2]} +{e[3]}",
                   font=font(14), fill=(100, 100, 104))
        frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        if self.vw:
            self.vw.write(frame)
        self.frames += 1
        if self.frames % 50 == 0:
            print(f"  渲染 {self.frames} 帧  t={t:.1f}s  已进 {world.delivered} 枚")

    def close(self):
        if self.vw:
            self.vw.release()
            print(f"[视频] {self.path}  {self.frames} 帧")


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["grid", "lane", "spread", "pile"], default="grid")
    ap.add_argument("--rule", choices=["any", "own"], default="any")
    ap.add_argument("--seconds", type=float, default=170.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    pieces = make_pieces(args.layout, args.seed)
    world = PushWorld(pieces, rule=args.rule)
    if args.layout in ("grid", "lane"):
        car = Car(2900.0, 1000.0, 180.0, rng=np.random.default_rng(args.seed))
        ctrl = (GridFeeder if args.layout == "grid" else LaneFeeder)(
            world, car, side="red", rule=args.rule)
    else:
        car = Car(320.0, config.FIELD_W - 210.0, 0.0, rng=np.random.default_rng(args.seed))
        ctrl = Peeler(world, car, side="red", rule=args.rule)
    if args.video:
        if args.layout == "grid":
            title = "场景 A：4×4 取棋工位 · 32 枚连续回收（16 趟，每趟 2 枚）"
            note = "机构节拍与重复精度演示：每工位 2 枚，车走净空通道；不代表开局堆态"
        elif args.layout == "lane":
            title = "场景 A：双通道供棋 · 32 枚连续回收（16 趟，每趟 2 枚）"
            note = "机构节拍与重复精度演示：棋子按两条取棋通道预置，车在中间通道往返"
        elif args.layout == "pile":
            title = "场景 B：官方开局堆态 · 只从外缘剥 1~2 枚"
            note = "32 枚随机堆在 600×500 中央区；车体不得穿堆，推板必须真实迎住棋子"
        else:
            title = "场景 A：中局散布态 · 32 枚逐趟回收（每趟 1~2 枚）"
            note = "棋子已被对抗撞散，用于验证机构重复精度；不代表开局堆态可直接清空"
        out = args.out or str(Path(__file__).resolve().parent.parent / "vision" / "data" / "output" /
                              {"spread": "push_spread.mp4", "lane": "push_lane.mp4",
                               "grid": "push_grid.mp4",
                               "pile": "push_pile.mp4"}[args.layout])
        rnd = DemoRenderer(out, fps=25, title=title, note=note, rule=args.rule,
                           layout=args.layout)
    else:
        rnd = None
    print(f"布局={args.layout} 规则={args.rule} 棋子={len(pieces)} 枚")
    dt = 0.02
    steps = int(args.seconds / dt)
    for i in range(steps):
        t = i * dt
        ctrl.update(t)
        world.update(car, t)
        if rnd and i % 2 == 0:
            rnd(world, car, ctrl, t)
        if ctrl.phase == "DONE":
            print(f"没有可安全取的目标：停止于 t={t:.1f}s")
            break
    if rnd:
        # 补 1 秒定格
        for k in range(25):
            rnd(world, car, ctrl, args.seconds + k * 0.04)
        rnd.close()
    print(f"结果：进 {world.delivered}/32 枚，得分 {world.score}，"
          f"趟数 {ctrl.trip}，失败 {ctrl.fail_count}，车体顶到棋子 {world.car_hits} 次")
    for row in ctrl.log[:40]:
        print(f"  第 {row[1]:2d} 趟 t={row[0]:5.1f}s 进 {row[2]} 枚 {row[3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
