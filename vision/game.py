# -*- coding: utf-8 -*-
"""棋局理解层：把"看到棋子"翻译成"这局该怎么打"。

视觉层只回答"哪儿有棋子、是哪一方的、是什么棋"；
这一层回答比赛真正关心的问题：

1. 每枚棋子在哪个区域（堆放区 / 己方球门区 / 对方球门区 / 对抗区）
2. 己方球门区里的棋子值多少分（本次可见范围）
3. **下一步该推哪一堆**——不是"推最近的一枚"，而是找分值合计最大的簇
   （推板一次能带一堆，这才是得分效率的关键）
4. 推这一堆需要车怎么走：对准点、车头朝向、推进距离
5. 打包成给下位机/决策层的数据（含串口文本帧示例）

注意：判定"己方球门区得分"需要场地坐标。车体坐标系下要先用位姿
（里程计/IMU）换算过去，见 to_field_frame()。另外车头相机只能看到一块视窗，
"累计得分"应该由上层状态机在每次成功投放后累加（Scoreboard），
不要指望单帧画面能算出来。
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# ------------------------------------------------------------------ 坐标
def to_field_frame(pieces, pose=None):
    """把车体坐标系的棋子换算到场地坐标系。pose=(x, y, theta_deg) 是车在场地上的位姿。"""
    if not pose:
        return pieces
    x0, y0, th = pose[0], pose[1], math.radians(pose[2])
    c, s = math.cos(th), math.sin(th)
    out = []
    for p in pieces:
        out.append({**p, "x": round(x0 + c * p["x"] - s * p["y"], 1),
                    "y": round(y0 + s * p["x"] + c * p["y"], 1)})
    return out


# ------------------------------------------------------------------ 区域
def own_goal_box():
    """己方球门区（甲队=左墙长条；本队是乙队时自动镜像）。"""
    if config.PLAYER_NAME == "甲队":
        return (0.0, config.GOAL_DEPTH, config.GOAL_Y0, config.GOAL_Y1)
    return (config.FIELD_L - config.GOAL_DEPTH, config.FIELD_L, config.GOAL_Y0, config.GOAL_Y1)


def rival_goal_box():
    x0, x1, y0, y1 = own_goal_box()
    return (config.FIELD_L - x1, config.FIELD_L - x0, y0, y1)


def own_goal_center():
    x0, x1, y0, y1 = own_goal_box()
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def classify_zone(x, y):
    """返回棋子所在区域：own_goal / rival_goal / pile / field。"""
    for name, box in (("own_goal", own_goal_box()), ("rival_goal", rival_goal_box())):
        x0, x1, y0, y1 = box
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    if (abs(x - config.PILE_CX) <= config.PILE_L / 2
            and abs(y - config.PILE_CY) <= config.PILE_W / 2):
        return "pile"
    return "field"


def label_zone(z):
    return {"own_goal": "己方球门区", "rival_goal": "对方球门区",
            "pile": "棋子堆放区", "field": "对抗区"}.get(z, z)


def annotate_zones(pieces):
    out = []
    for p in pieces:
        z = classify_zone(p["x"], p["y"])
        out.append({**p, "zone": z, "zone_cn": label_zone(z)})
    return out


# ------------------------------------------------------------------ 得分
def count_value(pieces, mode=None, own_side=None):
    """己方球门区里棋子的分值合计。

    mode=own_color 只算本队颜色；mode=any_color 推进去就算分。
    本队颜色还没确定（OWN_SIDE=None）时，先按"不限颜色"统计，方便规则出来前跑通流程。
    """
    mode = mode or config.SCORE_MODE
    own_side = config.OWN_SIDE if own_side is None else own_side
    in_goal = [p for p in pieces if p["zone"] == "own_goal"]
    if mode == "any_color" or own_side is None:
        return sum(p.get("value", 0) for p in in_goal), len(in_goal)
    mine = [p for p in in_goal if p["side"] == own_side]
    return sum(p.get("value", 0) for p in mine), len(mine)


class Scoreboard:
    """累计得分：每次把一堆棋子成功推进球门区后调用 add() 累加。"""

    def __init__(self):
        self.total = 0
        self.detail = []

    def add(self, value, count=1, tag=None):
        self.total += value
        self.detail.append((value, count, tag))
        return self.total

    def reset(self):
        self.total, self.detail = 0, []


# ------------------------------------------------------------------ 自动计分
class GoalCounter:
    """视野内自动计分：看到棋子出现在己方球门区就计一次，用位置去重。

    为什么需要它：车头相机只能看到一块视窗，看不到球门区全貌，而且投放完成后
    棋子就离开视野了。有了这一层，只要投放那几帧里球门区在画面内，分就会自动加上。
    它统计的是"视野内观测到的入区棋子"，是真实得分的**下界**；
    正式判分以主办方为准，这里主要用于策略决策和赛后复盘。
    """

    def __init__(self, tol_mm=28.0, mode=None, own_side=None):
        self.tol = float(tol_mm)
        self.mode = mode
        self.own_side = own_side
        self.items = []

    def update(self, pieces):
        """返回这一帧"新进区"的棋子列表。"""
        added = []
        mode = self.mode or config.SCORE_MODE
        own_side = self.own_side if self.own_side is not None else config.OWN_SIDE
        for p in pieces:
            if p.get("zone") != "own_goal" or not p.get("char"):
                continue
            if mode == "own_color" and own_side and p["side"] != own_side:
                continue
            if any(math.hypot(p["x"] - q["x"], p["y"] - q["y"]) <= self.tol for q in self.items):
                continue
            self.items.append({"x": p["x"], "y": p["y"], "value": p.get("value", 0),
                               "char": p["char"], "side": p["side"]})
            added.append(p)
        return added

    @property
    def total(self):
        return sum(q["value"] for q in self.items)

    @property
    def count(self):
        return len(self.items)

    def reset(self):
        self.items = []


# ------------------------------------------------------------------ 聚类：找"一铲能带走的一堆"
def cluster_pieces(pieces, radius=None):
    """把已识别的棋子按距离聚类（并查集，单链式），返回每簇的成员列表。

    只对"已认出字"的棋子聚类：字都没认出来，就不知道值不值得推。
    """
    radius = float(radius or config.CLUSTER_R)
    items = [p for p in pieces if p.get("char")]
    n = len(items)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if math.hypot(items[i]["x"] - items[j]["x"], items[i]["y"] - items[j]["y"]) <= radius:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(items[i])
    return list(groups.values())


def summarize_cluster(members):
    """算一簇的合计分值、中心、推进方向、到球门区的距离。"""
    xs = [p["x"] for p in members]
    ys = [p["y"] for p in members]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    gx, gy = own_goal_center()
    dx, dy = gx - cx, gy - cy
    dist = math.hypot(dx, dy) or 1e-6
    ux, uy = dx / dist, dy / dist          # 推进方向：从簇中心指向己方球门区
    ext = max(abs((p["x"] - cx) * ux + (p["y"] - cy) * uy) for p in members)
    n_red = sum(1 for p in members if p["side"] == "red")
    return {
        "n": len(members),
        "value": sum(p.get("value", 0) for p in members),
        "x": round(cx, 1), "y": round(cy, 1),
        "red_ratio": round(n_red / len(members), 2),
        "span_mm": round(ext, 1),
        "dist_goal": round(dist, 1),
        "dir": [round(ux, 4), round(uy, 4)],
        "chars": [p["char"] for p in members],
        "members": members,
    }


def plan_push(cluster, blade_ahead=None):
    """给出推这一簇的走法（场地坐标系）。

    返回：车体中心该去的对准点 / 车头朝向 / 推进距离 / 预期得分。
    控制层拿到就是三件事：开过去对准、摆正车头、直推。
    """
    blade = float(blade_ahead if blade_ahead is not None else config.BLADE_AHEAD_MM)
    cx, cy = cluster["x"], cluster["y"]
    ux, uy = cluster["dir"]
    stand_off = blade + cluster["span_mm"]
    return {
        "aim": [round(cx - ux * stand_off, 1), round(cy - uy * stand_off, 1)],
        "heading_deg": round(math.degrees(math.atan2(uy, ux)), 1),
        "push_mm": round(cluster["dist_goal"] - config.GOAL_DEPTH / 2.0, 1),
        "expect_value": cluster["value"],
        "expect_count": cluster["n"],
        "chars": cluster["chars"],
    }


def rank_clusters(clusters, keep=3):
    """按性价比排序：分值高、距离近、别太散的优先。"""
    got = []
    for c in clusters:
        cost = c["dist_goal"] / 1000.0 + 0.35 * c["span_mm"] / 1000.0
        got.append({**c, "priority": round(c["value"] / (0.6 + cost), 2)})
    return sorted(got, key=lambda c: -c["priority"])[:keep]


# ------------------------------------------------------------------ 场地记忆（小视野必需）
class FieldMemory:
    """把"见过的棋子"记在场地坐标系里。

    为什么必须要有：车上只有一个前视相机，一帧只能看到一小块地方。
    如果每帧只看眼前那两三枚，就永远规划不出"先把哪一片扫掉最划算"。
    有了记忆，车可以边开边把棋子登记进来，规划时用的是**整场台账**，
    而不是当前这一帧；对已经推进球门区的棋子就把它从台账里划掉。

    局限（要清楚）：棋子被推动后位置会变，而记忆里存的是旧位置。
    所以每条记忆都带"最后一次看到"的时间，超时（MEMORY_TTL_S）就标记为
    "可能已移动"，规划时降权，并且优先相信刚刚看到的那一批。
    """

    def __init__(self, tol_mm=None, ttl_s=None):
        self.tol = float(tol_mm or config.MEMORY_TOL_MM)
        self.ttl = float(ttl_s or config.MEMORY_TTL_S)
        self.items = []            # [{x,y,side,char,value,score,t_seen,hits,zone}]

    def update(self, pieces, now=None):
        """把这一帧看到的棋子并进台账（同色 + 距离近 → 认为是同一枚）。"""
        now = time.time() if now is None else now
        added = []
        for p in pieces:
            if not p.get("char"):
                continue
            best, bd = None, 1e9
            for it in self.items:
                if it["side"] != p["side"] or it.get("gone"):
                    continue
                d = math.hypot(it["x"] - p["x"], it["y"] - p["y"])
                if d < bd:
                    bd, best = d, it
            if best is not None and bd <= self.tol:
                best.update({"x": p["x"], "y": p["y"], "char": p["char"],
                             "value": p.get("value", 0), "score": p.get("score", 0),
                             "t_seen": now, "zone": p.get("zone")})
                best["hits"] = best.get("hits", 1) + 1
            else:
                self.items.append({"x": p["x"], "y": p["y"], "side": p["side"],
                                   "char": p["char"], "value": p.get("value", 0),
                                   "score": p.get("score", 0), "t_seen": now,
                                   "zone": p.get("zone"), "hits": 1})
                added.append(p)
        # 进了己方球门区的，从"待推台账"里划掉
        for it in self.items:
            if it.get("zone") == "own_goal":
                it["gone"] = True
        return added

    def snapshot(self, now=None, fresh_only=False):
        now = time.time() if now is None else now
        out = []
        for it in self.items:
            if it.get("gone"):
                continue
            age = now - it["t_seen"]
            if age > self.ttl:
                it["stale"] = True
            if fresh_only and it.get("stale"):
                continue
            out.append({**it, "age_s": round(age, 1)})
        return out

    def clear(self):
        self.items = []

    def __len__(self):
        return sum(1 for it in self.items if not it.get("gone"))


# ------------------------------------------------------------------ 得分效率：分/秒
def estimate_seconds(cluster, pose=None, speed_mm_s=None):
    """跑这一趟大概要几秒：去目标 + 推 + 固定开销。

    pose=(x,y,theta) 是车当前在场地上的位姿；不知道就按"从簇的位置算"。
    """
    speed = float(speed_mm_s or config.ROBOT_SPEED_MM_S)
    if pose:
        d_to = math.hypot(cluster["x"] - pose[0], cluster["y"] - pose[1])
    else:
        d_to = cluster["dist_goal"]          # 保守假设：车就在球门区附近
    travel = (d_to + cluster["dist_goal"]) / max(speed, 1.0)
    return travel + config.ALIGN_SEC + config.TRIP_OVERHEAD_S


def plan_from_memory(memory, pose=None, speed_mm_s=None, blade_ahead=None,
                     cluster_r=None, keep=3):
    """从**整场台账**里挑出"分/秒"最高的一趟，给出规划。

    这直接回答了"先拿分高的还是先拿最近的"：都不是。
    该看的是得分效率 = 这一趟的总分 ÷ 这一趟花的时间，
    其中时间 = 开过去 + 推进球门区 + 对准与开销。
    """
    clusters = [summarize_cluster(m) for m in cluster_pieces(memory.snapshot(), cluster_r)]
    got = []
    for c in clusters:
        # 记忆里越久没看到的越可能已经被撞走，轻微降权
        ages = [m.get("age_s", 0) for m in c["members"]]
        age_pen = 1.0 / (1.0 + (sum(ages) / len(ages)) / max(config.MEMORY_TTL_S, 1.0))
        eta = estimate_seconds(c, pose, speed_mm_s)
        got.append({**c,
                    "eta_s": round(eta, 2),
                    "rate": round(c["value"] * age_pen / eta, 3),
                    "priority": round(c["value"] * age_pen / eta, 3),
                    "plan": plan_push(c, blade_ahead)})
    got.sort(key=lambda c: -c["rate"])
    return got[:keep]


# ------------------------------------------------------------------ 输出给下位机 / 决策层
def serial_frame(plan, t_ms=None):
    """一个紧凑的文本帧示例（每帧一行，逗号分隔，末尾 * 校验位）。

    真实协议你们自己定，这里只示范"该发什么信息"：
    动作、对准点、车头朝向、推进距离、预期得分。
    """
    if not plan:
        return "IDLE"
    body = (f"PUSH,{plan['aim'][0]:.0f},{plan['aim'][1]:.0f},"
            f"{plan['heading_deg']:.0f},{plan['push_mm']:.0f},"
            f"{plan['expect_value']},{plan['expect_count']}")
    if t_ms is not None:
        body = f"{t_ms}," + body
    return f"${body}*{sum(body.encode()) & 0xFF:02X}"


def brief(state):
    """一行中文播报，给调试界面用。"""
    plans = state.get("plans") or []
    if not plans:
        return f"没有可推目标（已识别 {state.get('n_known', 0)} 枚）"
    p = plans[0]
    eta = f"，预计 {p['eta_s']:.1f}s（{p['rate']:.1f} 分/秒）" if p.get("eta_s") else ""
    mem = f"记忆 {state['n_memory']} 枚 / " if state.get("n_memory") is not None else ""
    return (f"{mem}推荐：推 {p['n']} 枚（{''.join(p['chars'])}）合计 {p['value']} 分 "
            f"→ 朝向 {p['plan']['heading_deg']:.0f}°，推进 {p['plan']['push_mm'] / 1000:.2f}m"
            f"{eta}")


def build_state(pieces, pose=None, scoreboard=None, score_mode=None, own_side=None,
                memory=None, blade_ahead=None, cluster_r=None):
    """把视觉结果整理成一份完整的"棋局态势"。

    pieces 每枚应含 x, y, side, char, value（recognize_chars 的输出即满足）。
    """
    pieces = annotate_zones(to_field_frame(pieces, pose))
    if memory is not None:
        memory.update(pieces)
        ranked = plan_from_memory(memory, pose, blade_ahead=blade_ahead, cluster_r=cluster_r)
    else:
        ranked = rank_clusters([summarize_cluster(m)
                                for m in cluster_pieces(pieces, cluster_r)])
        for c in ranked:
            c["plan"] = plan_push(c, blade_ahead)
    in_view_value, in_view_n = count_value(pieces, score_mode, own_side)
    state = {
        "pieces": pieces,
        "n_known": sum(1 for p in pieces if p.get("char")),
        "n_memory": (len(memory) if memory is not None else None),
        "clusters": ranked,
        "plans": ranked,
        "in_view_value": in_view_value,
        "in_view_count": in_view_n,
        "on_field_value": sum(p.get("value", 0) for p in pieces),
        "cum_total": (scoreboard.total if scoreboard else None),
    }
    state["brief"] = brief(state)
    return state


if __name__ == "__main__":
    # 自测：一份假数据，看聚类与推进规划是否合理
    demo = [
        {"x": 1500, "y": 1000, "side": "red", "char": "车", "value": 9},
        {"x": 1560, "y": 1040, "side": "red", "char": "马", "value": 8},
        {"x": 1470, "y": 960, "side": "red", "char": "炮", "value": 8},
        {"x": 1620, "y": 1080, "side": "black", "char": "卒", "value": 1},
        {"x": 2300, "y": 1500, "side": "red", "char": "兵", "value": 1},
    ]
    st = build_state(demo, pose=(0, 0, 0))
    for c in st["clusters"]:
        print(f"簇：{c['n']} 枚 {c['chars']} 合计 {c['value']} 分  "
              f"距球门 {c['dist_goal']:.0f}mm  优先级 {c['priority']}")
        print(f"    规划：对准 {c['plan']['aim']}，车头 {c['plan']['heading_deg']}°，"
              f"推进 {c['plan']['push_mm']:.0f}mm")
        print(f"    串口帧：{serial_frame(c['plan'], t_ms=1234)}")
    print("播报：", st["brief"])
