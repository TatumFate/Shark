# -*- coding: utf-8 -*-
"""2D 仿真：车 + 棋子 + 推板 + 球门区，用来在没有硬件时验证决策层。

仿真里的分工和真实系统一一对应：
    真实：realtime.py(视觉) → game.build_state(态势) → fsm.py(决策) → 串口 → firmware(执行)
    仿真：VisionEmu(视觉)      → game.build_state         → fsm.py      → SimChassis(执行)

所以现在没有底盘、没有真棋子，也能把"选目标 → 开过去 → 对准 → 推 → 投放 → 撤离"
整条决策链跑通、量出得分和用时；等底盘好了，只需把 SimChassis 换成串口发送。
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
    """模拟下位机：执行 GOTO / TURN / DRIVE / BLADE / STOP。

    GOTO 拆成三段（和固件里的实现保持一致）：转向目标点 → 直走 → 摆正到指定朝向。
    指令是幂等的：重复收到同一条指令不会把任务重新开始（真实串口也要求这样）。
    """

    def __init__(self, x, y, heading_deg, v_max=None, w_max=110.0, rng=None):
        self.x, self.y, self.h = float(x), float(y), float(heading_deg)
        self.v_max = float(v_max or config.ROBOT_SPEED_MM_S)
        self.w_max = float(w_max)
        self.task = None
        self.last_cmd = None
        self.blade_down = False
        self.rng = rng or np.random.default_rng(0)
        self.odo = 0.0                      # 累计里程(mm)，用于卡住检测

    # ---- 接口
    def pose(self):
        return (self.x, self.y, wrap_deg(self.h))

    def is_done(self):
        return self.task is None

    def send(self, cmd):
        s = str(cmd)
        if s == self.last_cmd:              # 幂等：同一条指令不重启任务
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
        elif v == "STOP":
            self.task = None
        elif v == "IDLE":
            self.task = None

    # ---- 一步积分
    def step(self, dt):
        if self.task is None:
            return
        t = self.task
        if t["type"] == "turn":
            err = wrap_deg(t["h"] - self.h)
            w = max(-self.w_max, min(self.w_max, err * 3.0))
            if abs(err) < 3.0:
                self.task = None
                return
            self.h = wrap_deg(self.h + w * dt)
            self.h += float(self.rng.normal(0, 0.03))
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
                # 关键：直行段也要持续修正航向（纯追踪）。
                # 否则一冲过目标点，就会沿着原航向一路开到墙角、再也回不来。
                bearing = math.degrees(math.atan2(dy, dx))
                err = wrap_deg(bearing - self.h)
                w = max(-self.w_max, min(self.w_max, err * 2.5))
                self.h = wrap_deg(self.h + w * dt)
                v = min(self.v_max, max(120.0, dist * 3.5))
                v *= max(0.25, 1.0 - abs(err) / 120.0)      # 航向偏差大就减速
                step = min(v * dt, dist)
                self.x += math.cos(math.radians(self.h)) * step
                self.y += math.sin(math.radians(self.h)) * step
                self.odo += step
            else:                            # 到点后摆正
                err = wrap_deg(t["h"] - self.h)
                if abs(err) < 3.0:
                    self.task = None
                    return
                w = max(-self.w_max, min(self.w_max, err * 3.0))
                self.h = wrap_deg(self.h + w * dt)
        elif t["type"] == "drive":
            left = t["left"]
            if abs(left) < 10.0:
                self.task = None
                return
            v = min(self.v_max, max(200.0, abs(left) * 4.0))
            step = math.copysign(min(v * dt, abs(left)), left)
            self.x += math.cos(math.radians(self.h)) * step
            self.y += math.sin(math.radians(self.h)) * step
            t["left"] -= step
            self.odo += abs(step)
        # 位置噪声 + 场地边界
        if self.task is not None:
            self.x += float(self.rng.normal(0, 0.6))
            self.y += float(self.rng.normal(0, 0.6))
        self.x = max(30.0, min(config.FIELD_L - 30.0, self.x))
        self.y = max(30.0, min(config.FIELD_W - 30.0, self.y))


class SimWorld:
    """棋子 + 推板推动 + 计分。"""

    def __init__(self, pieces, rng=None):
        self.pieces = pieces
        self.rng = rng or np.random.default_rng(1)
        self.scored = []            # 已进区并冻结的棋子
        self.push_count = 0

    @staticmethod
    def make_pieces(n=32, rng=None):
        """在堆放区随机撒 n 枚棋子（不重叠），红黑各一半，分值按 config.VALUE。"""
        rng = rng or random.Random(0)
        chars_red = [("帅", 1), ("仕", 2), ("相", 2), ("车", 2), ("马", 2), ("炮", 2), ("兵", 5)]
        chars_blk = [("将", 1), ("士", 2), ("象", 2), ("车", 2), ("马", 2), ("炮", 2), ("卒", 5)]
        out = []
        for side, spec in (("red", chars_red), ("black", chars_blk)):
            for ch, k in spec:
                for _ in range(k):
                    out.append({"side": side, "char": ch,
                                "value": config.VALUE.get(config.NORMALIZE.get(ch, ch), 0)})
        out = out[:n]
        placed = []
        for p in out:
            for _ in range(4000):
                x = rng.uniform(config.PILE_CX - config.PILE_L / 2 + config.PIECE_R,
                                config.PILE_CX + config.PILE_L / 2 - config.PIECE_R)
                y = rng.uniform(config.PILE_CY - config.PILE_W / 2 + config.PIECE_R,
                                config.PILE_CY + config.PILE_W / 2 - config.PIECE_R)
                if all(math.hypot(x - q["x"], y - q["y"]) > config.PIECE_DIA * 1.1 for q in placed):
                    placed.append({**p, "x": x, "y": y, "r": config.PIECE_R, "scored": False})
                    break
        return placed

    def update(self, car):
        """推板推动 + 边界 + 计分。"""
        f = (math.cos(math.radians(car.h)), math.sin(math.radians(car.h)))
        l = (-f[1], f[0])
        half_w = config.BLADE_WIDTH_MM / 2.0
        for p in self.pieces:
            if p["scored"]:
                continue
            dx, dy = p["x"] - car.x, p["y"] - car.y
            along = dx * f[0] + dy * f[1]
            lat = dx * l[0] + dy * l[1]
            if car.blade_down and 0.0 < along < config.BLADE_AHEAD_MM + p["r"] + 12.0 \
                    and abs(lat) < half_w - p["r"] * 0.2:
                # 被推：保持在推板前方（前后沿齐平），并有一点侧向滑移
                new_along = config.BLADE_AHEAD_MM + p["r"] + 3.0
                slide = 0.0
                if self.rng.random() < 0.12:
                    slide = math.copysign(4.0, lat if lat != 0 else 1.0)
                p["x"] = car.x + f[0] * new_along + l[0] * (lat + slide)
                p["y"] = car.y + f[1] * new_along + l[1] * (lat + slide)
                self.push_count += 1
            # 墙（挡板）
            p["x"] = max(p["r"], min(config.FIELD_L - p["r"], p["x"]))
            p["y"] = max(p["r"], min(config.FIELD_W - p["r"], p["y"]))
            # 计分：整体进入己方球门区
            gx0, gx1, gy0, gy1 = game.own_goal_box()
            if (p["x"] - p["r"] >= gx0 - 1 and p["x"] + p["r"] <= gx1
                    and p["y"] - p["r"] >= gy0 and p["y"] + p["r"] <= gy1):
                p["scored"] = True
                self.scored.append(p)

    @property
    def score(self):
        return sum(p["value"] for p in self.scored)


class VisionEmu:
    """模拟前视相机：只看到车前的一块区域，带位置噪声、漏检、字符误判。"""

    def __init__(self, forward_mm=520.0, half_w=400.0, noise=3.0,
                 dropout=0.05, char_err=0.08, rng=None):
        self.forward_mm = forward_mm
        self.half_w = half_w
        self.noise = noise
        self.dropout = dropout
        self.char_err = char_err
        self.rng = rng or np.random.default_rng(2)

    def observe(self, world, car):
        """返回**车体坐标系**下的棋子列表（x 前，y 左），与真实相机标定的坐标系一致。"""
        th = math.radians(car.h)
        c, s = math.cos(th), math.sin(th)
        out = []
        for p in world.pieces:
            if p["scored"]:
                continue
            dx, dy = p["x"] - car.x, p["y"] - car.y
            rx = c * dx + s * dy                 # 前向
            ry = -s * dx + c * dy                # 侧向
            if not (60.0 < rx < self.forward_mm + 280.0 and abs(ry) < self.half_w):
                continue
            if self.rng.random() < self.dropout:
                continue
            ch = p["char"]
            if self.rng.random() < self.char_err:
                ch = "车" if ch != "车" else "马"      # 模拟识别错一个字
            out.append({"x": rx + float(self.rng.normal(0, self.noise)),
                        "y": ry + float(self.rng.normal(0, self.noise)),
                        "r": p["r"], "side": p["side"],
                        "char": config.NORMALIZE.get(ch, ch),
                        "value": config.VALUE.get(config.NORMALIZE.get(ch, ch), 0),
                        "score": 0.9, "fresh": True})
        return out


def run_episode(seed=0, seconds=None, dt=0.02, vision_hz=5.0, render=None,
                opponent=None, verbose=False, start=None, vision=None):
    """跑一局。render=(画布函数) 时每 0.2s 画一帧。"""
    seconds = float(seconds or config.MATCH_SECONDS)
    rng = np.random.default_rng(seed)
    pieces = SimWorld.make_pieces(32, random.Random(seed))
    world = SimWorld(pieces, rng)
    sx, sy = start or (config.START_L / 2, config.FIELD_W - config.START_W / 2)
    heading = math.degrees(math.atan2(config.PILE_CY - sy, config.PILE_CX - sx))
    car = SimChassis(sx, sy, heading, rng=np.random.default_rng(seed + 10))
    mem = game.FieldMemory()
    vk = dict(vision or {})
    emu = VisionEmu(rng=np.random.default_rng(seed + 20), **vk)
    fsm = DecisionFSM(match_seconds=seconds)
    fsm.start(0.0)

    vis_every = max(1, int(round(1.0 / (vision_hz * dt))))
    frames = 0
    for step in range(int(seconds / dt)):
        t = step * dt
        if step % vis_every == 0:
            robot_pieces = emu.observe(world, car)
            gs = game.build_state(robot_pieces, pose=car.pose(), memory=mem)
            cmd = fsm.update(gs, car.pose(), t, car.is_done(),
                             opponent=opponent(world, car) if opponent else None, memory=mem)
            car.send(cmd)
        car.step(dt)
        world.update(car)
        if render and step % max(1, int(0.2 / dt)) == 0:
            render(world, car, fsm, t)
            frames += 1

    return {
        "score": world.score,
        "scored_n": len(world.scored),
        "seconds": seconds,
        "trips": fsm.trips,
        "recoveries": fsm.recoveries,
        "state": fsm.state,
        "rate": world.score / max(seconds, 1e-6),
        "left": sum(1 for p in world.pieces if not p["scored"]),
        "frames": frames,
        "history": fsm.history,
    }
