# -*- coding: utf-8 -*-
"""2D 仿真（双队版）：红队 vs 黑队，用真棋子图形，按"只能推自己颜色"计分。

规则（按你们确认的版本）：
  · 红队守左门、黑队守右门（config.SIDE_GOAL）
  · 己方颜色进己方球门 → +分值
  · 对方颜色进己方球门 → -分值 × PENALTY_FACTOR（推错扣分）
  · 一次推一枚还是多枚未定 → 决策按"净收益/耗时"选，推多枚会自然出现
"""
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vision"))
import config  # noqa: E402
import game  # noqa: E402
from fsm import DecisionFSM  # noqa: E402


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


class SimChassis:
    """模拟下位机：执行 GOTO / TURN / DRIVE / BLADE / STOP（和固件行为对齐）。"""

    def __init__(self, x, y, heading_deg, v_max=None, w_max=110.0, rng=None):
        self.x, self.y, self.h = float(x), float(y), float(heading_deg)
        self.v_max = float(v_max or config.ROBOT_SPEED_MM_S)
        self.w_max = float(w_max)
        self.task = None
        self.last_cmd = None
        self.blade_down = False
        self.rng = rng or np.random.default_rng(0)
        self.obstruct = 0.0          # 0~0.6：车体压到棋子时降速用
        self.arrive_err = None       # 上一次 GOTO 结束时离目标还差多少（调试用）

    def pose(self):
        return (self.x, self.y, wrap_deg(self.h))

    def is_done(self):
        return self.task is None

    def send(self, cmd):
        s = str(cmd)
        # 只在"任务正在执行"时去重；任务已完成（task is None）时，同一条指令
        # 也要重新接受——否则"回到起点再发同样的 GOTO"会被忽略，车以为已经到了。
        if s == self.last_cmd and self.task is not None:
            return
        self.last_cmd = s
        v = cmd.verb
        if v == "GOTO":
            x, y, h = cmd.args
            self.task = {"type": "goto", "x": float(x), "y": float(y),
                         "h": float(h), "phase": "turn"}
        elif v == "TURN":
            self.task = {"type": "turn", "h": float(cmd.args[0])}
        elif v == "DRIVE":
            self.task = {"type": "drive", "left": float(cmd.args[0])}
        elif v == "BLADE":
            self.blade_down = bool(cmd.args[0])
            self.task = None
        elif v in ("STOP", "IDLE"):
            self.task = None

    def step(self, dt):
        if self.task is None:
            return
        t = self.task
        if t["type"] == "turn":
            err = wrap_deg(t["h"] - self.h)
            if abs(err) < 3.0:
                self.task = None
                return
            w = max(-self.w_max, min(self.w_max, err * 3.0))
            self.h = wrap_deg(self.h + w * dt) + float(self.rng.normal(0, 0.03))
        elif t["type"] == "goto":
            if t["phase"] == "turn":
                bearing = math.degrees(math.atan2(t["y"] - self.y, t["x"] - self.x))
                err = wrap_deg(bearing - self.h)
                if abs(err) < 6.0:
                    t["phase"] = "drive"
                    return
                w = max(-self.w_max, min(self.w_max, err * 3.0))
                self.h = wrap_deg(self.h + w * dt)
            elif t["phase"] == "drive":
                dx, dy = t["x"] - self.x, t["y"] - self.y
                dist = math.hypot(dx, dy)
                if dist < 25.0:
                    t["phase"] = "align"
                    return
                bearing = math.degrees(math.atan2(dy, dx))
                err = wrap_deg(bearing - self.h)
                w = max(-self.w_max, min(self.w_max, err * 2.5))
                self.h = wrap_deg(self.h + w * dt)
                v = min(self.v_max, max(120.0, dist * 3.5)) * max(0.25, 1.0 - abs(err) / 120.0)
                v *= (1.0 - self.obstruct)
                step = min(v * dt, dist)
                self.x += math.cos(math.radians(self.h)) * step
                self.y += math.sin(math.radians(self.h)) * step
            else:
                err = wrap_deg(t["h"] - self.h)
                if abs(err) < 3.0:
                    self.arrive_err = math.hypot(t["x"] - self.x, t["y"] - self.y)
                    self.task = None
                    return
                w = max(-self.w_max, min(self.w_max, err * 3.0))
                self.h = wrap_deg(self.h + w * dt)
        elif t["type"] == "drive":
            left = t["left"]
            if abs(left) < 10.0:
                self.task = None
                return
            v = min(self.v_max, max(200.0, abs(left) * 4.0)) * (1.0 - self.obstruct)
            step = math.copysign(min(v * dt, abs(left)), left)
            self.x += math.cos(math.radians(self.h)) * step
            self.y += math.sin(math.radians(self.h)) * step
            t["left"] -= step
        if self.task is not None:
            self.x += float(self.rng.normal(0, 0.6))
            self.y += float(self.rng.normal(0, 0.6))
        self.x = max(30.0, min(config.FIELD_L - 30.0, self.x))
        self.y = max(30.0, min(config.FIELD_W - 30.0, self.y))


class SimWorld:
    """棋子 + 双方推板 + 计分（含推错扣分）。"""

    def __init__(self, pieces, rng=None):
        self.pieces = pieces
        self.rng = rng or np.random.default_rng(1)
        self.score = {"red": 0, "black": 0}
        self.scored_count = {"red": 0, "black": 0}
        self.events = []          # (时间标签, 队伍, 棋子, 得分)

    @staticmethod
    def make_pieces(n=32, rng=None, spread=False):
        rng = rng or random.Random(0)
        spec = {
            "red": [("帅", 1), ("仕", 2), ("相", 2), ("车", 2), ("马", 2), ("炮", 2), ("兵", 5)],
            "black": [("将", 1), ("士", 2), ("象", 2), ("车", 2), ("马", 2), ("炮", 2), ("卒", 5)],
        }
        out = []
        for side in ("red", "black"):
            for ch, k in spec[side]:
                for _ in range(k):
                    out.append({"side": side, "char": ch,
                                "value": config.VALUE.get(config.NORMALIZE.get(ch, ch), 0)})
        out = out[:n]
        placed = []
        min_d = config.PIECE_DIA * (3.4 if spread else 1.1)
        for p in out:
            for _ in range(8000):
                if spread:
                    x = rng.uniform(420.0, config.FIELD_L - 420.0)
                    y = rng.uniform(320.0, config.FIELD_W - 320.0)
                else:
                    x = rng.uniform(config.PILE_CX - config.PILE_L / 2 + config.PIECE_R,
                                    config.PILE_CX + config.PILE_L / 2 - config.PIECE_R)
                    y = rng.uniform(config.PILE_CY - config.PILE_W / 2 + config.PIECE_R,
                                    config.PILE_CY + config.PILE_W / 2 - config.PIECE_R)
                if all(math.hypot(x - q["x"], y - q["y"]) > min_d for q in placed):
                    placed.append({**p, "x": x, "y": y, "r": config.PIECE_R, "scored": False})
                    break
        return placed

    @staticmethod
    def goal_of(p):
        """棋子整体进入哪个门：'left' / 'right' / None。"""
        for name, (x0, x1, y0, y1) in (("left", (0.0, config.GOAL_DEPTH, config.GOAL_Y0, config.GOAL_Y1)),
                                       ("right", (config.FIELD_L - config.GOAL_DEPTH, config.FIELD_L,
                                                  config.GOAL_Y0, config.GOAL_Y1))):
            if (p["x"] - p["r"] >= x0 - 0.5 and p["x"] + p["r"] <= x1
                    and p["y"] - p["r"] >= y0 and p["y"] + p["r"] <= y1):
                return name
        return None

    def _push(self, car):
        f = (math.cos(math.radians(car.h)), math.sin(math.radians(car.h)))
        l = (-f[1], f[0])
        half_w = config.BLADE_WIDTH_MM / 2.0
        for p in self.pieces:
            if p["scored"]:
                continue
            dx, dy = p["x"] - car.x, p["y"] - car.y
            along = dx * f[0] + dy * f[1]
            lat = dx * l[0] + dy * l[1]
            if not car.blade_down or not (0.0 < along < config.BLADE_AHEAD_MM + p["r"] + 12.0):
                continue
            # 侧挡边：棋子必须能进铲口；进去之后被两侧挡边夹住
            mouth = half_w - config.GUIDE_MM - p["r"] * 0.2
            if abs(lat) > mouth:
                continue
            new_along = config.BLADE_AHEAD_MM + p["r"] + 3.0
            slide = 0.0
            if config.GUIDE_MM <= 0 and self.rng.random() < 0.12:
                slide = math.copysign(4.0, lat if lat != 0 else 1.0)
            lat_c = max(-mouth, min(mouth, lat + slide))
            p["x"] = car.x + f[0] * new_along + l[0] * lat_c
            p["y"] = car.y + f[1] * new_along + l[1] * lat_c
            p["pushed"] = p.get("pushed", 0) + 1

    def _collide(self, car):
        """车体压到棋子：把棋子挤到最近的车体边外，并让车被顶偏、降速。

        这是之前仿真漏掉的一环。真实车离地只有约 10mm，而棋子有 10~20mm 厚、44mm 直径：
        车开过去时棋子不会消失，而是被车轮/底盘挤到旁边，同时把车顶歪、拖慢。
        所以"直接穿过棋堆再开始推"在现实里是不成立的——会一路把棋子扒拉散。
        """
        th = math.radians(car.h)
        c, s = math.cos(th), math.sin(th)
        hl, hw = config.BODY_L_MM / 2.0, config.BODY_W_MM / 2.0
        hit = 0
        for p in self.pieces:
            if p["scored"]:
                continue
            dx, dy = p["x"] - car.x, p["y"] - car.y
            along = c * dx + s * dy
            lat = -s * dx + c * dy
            if abs(along) > hl + p["r"] or abs(lat) > hw + p["r"]:
                continue
            hit += 1
            need_lat = (hw + p["r"] + 2.0) - abs(lat)
            need_along = (hl + p["r"] + 2.0) - abs(along)
            if need_lat <= need_along:
                lat2 = math.copysign(hw + p["r"] + 2.0, lat if lat != 0 else 1.0)
                p["x"] = car.x + c * along - s * lat2
                p["y"] = car.y + s * along + c * lat2
            else:
                along2 = math.copysign(hl + p["r"] + 2.0, along if along != 0 else 1.0)
                p["x"] = car.x + c * along2 - s * lat
                p["y"] = car.y + s * along2 + c * lat
        if hit:
            car.x += float(self.rng.normal(0, 0.9 * hit))
            car.y += float(self.rng.normal(0, 0.9 * hit))
            car.h += float(self.rng.normal(0, 0.4 * hit))
            car.obstruct = min(0.6, 0.12 * hit)      # 压到的棋子越多越慢
        else:
            car.obstruct = 0.0

    def update(self, cars, t=0.0):
        for car in cars:
            self._collide(car)
            self._push(car)
        for p in self.pieces:
            if p["scored"]:
                continue
            p["x"] = max(p["r"], min(config.FIELD_L - p["r"], p["x"]))
            p["y"] = max(p["r"], min(config.FIELD_W - p["r"], p["y"]))
            g = self.goal_of(p)
            if g is None:
                continue
            owner = "red" if g == "left" else "black"
            pts = p["value"] if p["side"] == owner else -p["value"] * config.PENALTY_FACTOR
            pts = int(round(pts))
            self.score[owner] += pts
            if pts > 0:
                self.scored_count[owner] += 1
            p["scored"] = True
            p["owner"] = owner
            p["pts"] = pts
            self.events.append((round(t, 1), owner, p["char"], p["side"], pts))


class VisionEmu:
    """模拟车前相机：只看车前一块；带位置噪声、漏检、字符误判。"""

    def __init__(self, forward_mm=520.0, half_w=400.0, noise=3.0,
                 dropout=0.05, char_err=0.08, rng=None):
        self.forward_mm = forward_mm
        self.half_w = half_w
        self.noise = noise
        self.dropout = dropout
        self.char_err = char_err
        self.rng = rng or np.random.default_rng(2)

    def observe(self, world, car):
        th = math.radians(car.h)
        c, s = math.cos(th), math.sin(th)
        out = []
        for p in world.pieces:
            if p["scored"]:
                continue
            dx, dy = p["x"] - car.x, p["y"] - car.y
            rx = c * dx + s * dy
            ry = -s * dx + c * dy
            if not (60.0 < rx < self.forward_mm + 280.0 and abs(ry) < self.half_w):
                continue
            if self.rng.random() < self.dropout:
                continue
            ch = p["char"]
            if self.rng.random() < self.char_err:
                ch = "车" if ch != "车" else "马"
            out.append({"x": rx + float(self.rng.normal(0, self.noise)),
                        "y": ry + float(self.rng.normal(0, self.noise)),
                        "r": p["r"], "side": p["side"],
                        "char": config.NORMALIZE.get(ch, ch),
                        "value": config.VALUE.get(config.NORMALIZE.get(ch, ch), 0),
                        "score": 0.9, "fresh": True})
        return out


def _start_pose(side):
    if side == "red":
        return (config.START_L / 2, config.FIELD_W - config.START_W / 2)
    return (config.FIELD_L - config.START_L / 2, config.START_W / 2)


def run_episode(seed=0, seconds=None, dt=0.02, vision_hz=5.0, render=None, vision=None,
                sides=("red", "black"), spread=False, debug=False):
    """sides 决定跑哪几支队：默认红黑都跑；传 ("red",) 就是只看红队。"""
    seconds = float(seconds or config.MATCH_SECONDS)
    rng = np.random.default_rng(seed)
    world = SimWorld(SimWorld.make_pieces(32, random.Random(seed), spread=spread), rng)
    cars, mems, fsms, emus = {}, {}, {}, {}
    for i, side in enumerate(sides):
        sx, sy = _start_pose(side)
        tx, ty = config.PILE_CX, config.PILE_CY
        cars[side] = SimChassis(sx, sy, math.degrees(math.atan2(ty - sy, tx - sx)),
                                rng=np.random.default_rng(seed + 10 + i))
        mems[side] = game.FieldMemory()
        fsms[side] = DecisionFSM(match_seconds=seconds, side=side)
        fsms[side].start(0.0)
        emus[side] = VisionEmu(rng=np.random.default_rng(seed + 20 + i), **(vision or {}))
    vis_every = max(1, int(round(1.0 / (vision_hz * dt))))
    frames = 0
    push_log = []
    for step in range(int(seconds / dt)):
        t = step * dt
        if step % vis_every == 0:
            for side in sides:
                car = cars[side]
                others = [cars[s] for s in sides if s != side]
                rp = emus[side].observe(world, car)
                gs = game.build_state(rp, pose=car.pose(), memory=mems[side], own_side=side)
                prev = fsms[side].state
                cmd = fsms[side].update(gs, car.pose(), t, car.is_done(),
                                        opponent=(others[0].pose()[:2] if others else None),
                                        memory=mems[side])
                if debug and fsms[side].state == "PUSH" and prev != "PUSH":
                    th = math.radians(car.h)
                    cc, ss = math.cos(th), math.sin(th)
                    tgt = fsms[side].target or {}
                    best = []
                    for m in tgt.get("members", []):
                        dx, dy = m["x"] - car.x, m["y"] - car.y
                        best.append((round(cc * dx + ss * dy), round(-ss * dx + cc * dy),
                                     m["char"]))
                    push_log.append((round(t, 1), round(car.x), round(car.y),
                                     round(car.h),
                                     None if car.arrive_err is None else round(car.arrive_err),
                                     best, round((tgt.get("plan") or {}).get("push_mm", 0)),
                                     car.blade_down))
                car.send(cmd)
        for side in sides:
            cars[side].step(dt)
        world.update([cars[s] for s in sides], t)
        if render and step % max(1, int(0.1 / dt)) == 0:
            render(world, cars, fsms, t)
            frames += 1
    return {
        "score": {s: world.score[s] for s in sides},
        "scored": {s: world.scored_count[s] for s in sides},
        "seconds": seconds,
        "trips": {s: fsms[s].trips for s in fsms},
        "recoveries": {s: fsms[s].recoveries for s in fsms},
        "states": {s: fsms[s].state for s in fsms},
        "history": {s: fsms[s].history for s in fsms},
        "events": world.events,
        "frames": frames,
        "pieces": world.pieces,
        "push_log": push_log,
    }
