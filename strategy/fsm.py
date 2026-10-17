# -*- coding: utf-8 -*-
"""决策层：把"棋局态势"翻译成"下位机动作"。

输入是 vision/game.py 的 build_state() 输出（实时程序里的 state 直接可用），
输出是一条运动指令（strategy/commands.py）。

状态机：
    IDLE → SEARCH ─找到目标→ APPROACH ─到点→ VERIFY ─还在→ ALIGN
                        ↑                              ↓
                        └──────── CLEARUP ← RETREAT ← DEPOSIT ← PUSH ← BLADE_DOWN
    APPROACH/PUSH/DEPOSIT 卡住或超时 → RECOVER（倒车脱困 + 目标冷却）→ SEARCH
    时间不够再跑一趟 → DONE

这一层只做"决定"，不碰电机、不碰图像：
  · 谁来喂：realtime.py 的 state（含 pieces / plans / 记忆）
  · 谁来执行：下位机（firmware/）或仿真器（strategy/sim.py）
所以现在没有底盘也能验证，等底盘好了只换"指令发送"这一处。
"""
import math
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vision"))
import config  # noqa: E402
import game  # noqa: E402
from commands import Command, blade, drive, goto, idle, stop, turn  # noqa: E402


class DecisionFSM:
    def __init__(self, match_seconds=None, speed_mm_s=None):
        self.match_seconds = float(match_seconds or config.MATCH_SECONDS)
        self.speed = float(speed_mm_s or config.ROBOT_SPEED_MM_S)
        self.state = "IDLE"
        self.t_start = None
        self.queue = deque()
        self.target = None
        self.blacklist = []              # [(x, y, until)]
        self.trips = 0
        self.recoveries = 0
        self.expected_value = 0
        self.history = []                # 状态切换记录，调试用
        self._now = 0.0
        self._pose = (0.0, 0.0, 0.0)
        self._state_t0 = 0.0
        self._stall_ref = None           # (x, y, t)
        self._spin_at = -1e9
        self._spins = 0                  # 连续原地扫描次数（扫不到就去堆放区看看）
        self._search_wp = False          # 当前队列是不是"去堆放区找"的搜索航点
        self._cur_cmd = None
        self._cur_cmd_t0 = 0.0

    # ------------------------------------------------------------------ 工具
    def start(self, now):
        self.t_start = now
        self._now = now
        self._set("SEARCH", "开始")

    def _set(self, st, why=""):
        self.state = st
        self._state_t0 = self._now
        self.history.append((round(self._now, 2), st, why))
        if len(self.history) > 200:
            self.history = self.history[-200:]

    def _issue(self, *cmds):
        self.queue = deque(cmds)

    def _reset_stall(self):
        self._stall_ref = (self._pose[0], self._pose[1], self._now)

    def _stalled(self):
        # 只对 DRIVE 判断卡住：GOTO 会先原地转向，位移为 0 是正常的，
        # 用位移判断会把"正在转向"误判成"卡住"（这个坑在仿真里抓到过一次）。
        if not self.queue or self.queue[0].verb != "DRIVE":
            return False
        if self._stall_ref is None:
            self._reset_stall()
            return False
        sx, sy, st = self._stall_ref
        if math.hypot(self._pose[0] - sx, self._pose[1] - sy) > 25:
            self._reset_stall()
            return False
        return (self._now - st) > config.STALL_SEC

    def _blacklisted(self, cx, cy, now):
        self.blacklist = [b for b in self.blacklist if b[2] > now]
        return any(math.hypot(b[0] - cx, b[1] - cy) < 90 for b in self.blacklist)

    def _pick(self, gs, now, opponent):
        """从候选里挑第一趟：跳过冷却中的、离对手太近的。"""
        for p in gs.get("plans") or []:
            if self._blacklisted(p["x"], p["y"], now):
                continue
            if opponent and math.hypot(p["plan"]["aim"][0] - opponent[0],
                                       p["plan"]["aim"][1] - opponent[1]) < config.OPPONENT_SAFE_MM:
                continue
            return p
        return None

    def _find_near(self, gs, cx, cy, tol=90.0):
        for p in gs.get("plans") or []:
            if math.hypot(p["x"] - cx, p["y"] - cy) < tol:
                return p
        return None

    # ------------------------------------------------------------------ 主循环
    def update(self, gs, pose, now, action_done=True, opponent=None, memory=None):
        """每个视觉帧调用一次。返回当前要发下位机的指令。"""
        self._now = float(now)
        self._pose = (float(pose[0]), float(pose[1]), float(pose[2]))
        cur = str(self.queue[0]) if self.queue else None
        if cur != self._cur_cmd:                 # 指令变了才重置计时
            self._cur_cmd = cur
            self._cur_cmd_t0 = float(now)
        if self.state == "IDLE":
            return idle()
        if self.t_start is None:
            self.start(now)

        remain = self.match_seconds - (now - self.t_start)
        if remain <= 0 and self.state not in ("DONE", "RETREAT"):
            self._set("DONE", "时间到")

        # 推棋/投放期间棋子位置一直在变：暂停登记，免得台账里记出一堆重复的"幽灵"
        if memory is not None:
            memory.paused = self.state in ("PUSH", "DEPOSIT", "BLADE_DOWN")

        # 上一条指令完成 → 出队
        if action_done and self.queue:
            self.queue.popleft()

        if self.state == "DONE":
            self.queue.clear()
            return stop()

        if self._stalled() and self.state in ("APPROACH", "PUSH", "DEPOSIT"):
            self._interrupt("RECOVER", "执行中卡住")
        elif self.state == "SEARCH" and self.queue and now - self._cur_cmd_t0 > 20.0:
            self.queue.clear()                       # 搜索航点跑太久（如被墙挡住）→ 放弃
            self._search_wp = False
        elif self.state == "APPROACH" and now - self._state_t0 > config.APPROACH_TIMEOUT_S:
            self._interrupt("RECOVER", "去目标超时")

        handler = getattr(self, "_on_" + self.state.lower(), None)
        if handler:
            handler(gs, now, remain, opponent, memory)
        return self.queue[0] if self.queue else idle()

    def _interrupt(self, st, why):
        """异常处理：放下推板、倒车、转开，然后清空队列重新规划。"""
        self.recoveries += 1
        if self.target:
            self.blacklist.append((self.target["x"], self.target["y"],
                                   self._now + config.BLACKLIST_S))
        self.target = None
        self.queue.clear()
        if st == "RECOVER":
            self._set("RECOVER", why)
            self.queue = deque([blade(False), drive(-config.RECOVER_BACK_MM),
                                turn(self._pose[2] + 40)])
        else:
            self._set(st, why)
        self._reset_stall()

    # ------------------------------------------------------------------ 各状态
    def _on_search(self, gs, now, remain, opponent, memory):
        target = self._pick(gs, now, opponent)
        if target is not None and self._search_wp:
            self.queue.clear()                       # 路上看见目标了，立刻放弃搜索航点
            self._search_wp = False
        if self.queue:
            return                                    # 正在扫描，等它转完
        if target is None:
            # 视野里没有可推的 → 先原地扫；连扫几圈还是看不到，就开去堆放区找
            # （车头相机只看得到 0.8m，开局车在角落、棋子在场中间，不主动过去永远看不见）
            self._spins += 1
            if self._spins >= 3:
                self._spins = 0
                self.queue = deque([goto(config.PILE_CX, config.PILE_CY, self._pose[2])])
                self._search_wp = True
            elif now - self._spin_at > 0.3:
                self._spin_at = now
                self.queue = deque([turn(self._pose[2] + config.SEARCH_TURN_DEG)])
            return
        self._spins = 0
        self._search_wp = False
        eta = float(target.get("eta_s") or 8.0)
        if eta * 1.15 > remain:
            self._set("DONE", f"剩余 {remain:.0f}s 不够跑这趟（需 {eta:.0f}s）")
            return
        self.target = target
        self.expected_value = int(target.get("value", 0))
        self._set("APPROACH", f"目标 {target['value']}分/{target['n']}枚")
        self.queue = deque([goto(target["plan"]["aim"][0], target["plan"]["aim"][1],
                                 target["plan"]["heading_deg"])])
        self._reset_stall()

    def _on_approach(self, gs, now, remain, opponent, memory):
        if self.queue:
            return
        self._set("VERIFY", "到对准点")

    def _on_verify(self, gs, now, remain, opponent, memory):
        """到点后复看一眼：目标还在不在、有没有漂。防"照着旧位置推空气"。"""
        again = None
        if self.target is not None:
            again = self._find_near(gs, self.target["x"], self.target["y"])
        if again is None:
            self._set("SEARCH", "目标不在了，重选")
            return
        self.target = again
        self.expected_value = int(again.get("value", 0))
        self._set("ALIGN", f"对准 {again['value']}分")
        self.queue = deque([turn(again["plan"]["heading_deg"])])

    def _on_align(self, gs, now, remain, opponent, memory):
        if self.queue:
            return
        self._set("BLADE_DOWN", "摆正，放推板")
        self.queue = deque([blade(True)])

    def _on_blade_down(self, gs, now, remain, opponent, memory):
        if self.queue:
            return
        self._set("PUSH", f"推进 {self.target['plan']['push_mm']:.0f}mm")
        self.queue = deque([drive(self.target["plan"]["push_mm"] + config.PUSH_MARGIN_MM)])
        self._reset_stall()

    def _on_push(self, gs, now, remain, opponent, memory):
        if self.queue:
            return
        self._set("DEPOSIT", "投放")
        self.queue = deque([drive(config.DEPOSIT_EXTRA_MM)])
        self._reset_stall()

    def _on_deposit(self, gs, now, remain, opponent, memory):
        if self.queue:
            return
        self.trips += 1
        self._set("RETREAT", "撤离")
        self.queue = deque([blade(False), drive(-config.RETREAT_MM),
                            turn(self._pose[2] + 55)])

    def _on_retreat(self, gs, now, remain, opponent, memory):
        if self.queue:
            return
        # 这一趟扫过的走廊里，棋子的位置全变了 → 从台账里清掉
        if memory is not None and self.target is not None:
            gx, gy = game.own_goal_center()
            memory.forget_segment(self.target["x"], self.target["y"], gx, gy,
                                  config.BLADE_WIDTH_MM + 120)
        self._set("CLEARUP", "清台账")
        self.target = None

    def _on_clearup(self, gs, now, remain, opponent, memory):
        self._set("SEARCH", "找下一趟")

    def _on_recover(self, gs, now, remain, opponent, memory):
        if self.queue:
            return
        self._set("SEARCH", "脱困完成")

    # ------------------------------------------------------------------ 观测
    def status(self, now=None):
        now = self._now if now is None else now
        remain = self.match_seconds - (now - self.t_start) if self.t_start else self.match_seconds
        tgt = None
        if self.target:
            tgt = f"{self.target['value']}分/{self.target['n']}枚"
        return {"state": self.state, "remain": max(0.0, remain), "trips": self.trips,
                "recoveries": self.recoveries, "target": tgt,
                "expected": self.expected_value,
                "cmd": str(self.queue[0]) if self.queue else "$IDLE"}
