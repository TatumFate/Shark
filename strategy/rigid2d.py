# -*- coding: utf-8 -*-
"""极简 2D 刚体引擎：圆盘刚体 + 运动学 OBB，顺序冲量 + 库仑摩擦 + 子步进。

没有外部依赖（本机装不了 pymunk/pybullet），但求解过程是真正的刚体动力学：
  - 每个棋子有质量、转动惯量、线速度、角速度；
  - 圆-圆、圆-OBB 接触用法向冲量 + 切向摩擦冲量求解；
  - 车体/夹爪是无限质量的运动学刚体，按真实速度参与接触；
  - 每帧 4 个子步，位置修正防止穿透。
"""
import math

MU_GROUND = 0.45
E_PIECE = 0.05
G_MM = 9800.0


class Body:
    __slots__ = ("x", "y", "vx", "vy", "a", "w", "r", "m", "inv_m", "inv_i",
                 "held", "scored", "z", "side", "char", "value", "id", "owner")

    def __init__(self, x, y, side, char, value, pid, mass=0.020, r=22.0):
        self.x, self.y = float(x), float(y)
        self.vx = self.vy = 0.0
        self.a = self.w = 0.0
        self.r = r
        self.m = mass
        self.inv_m = 1.0 / mass
        self.inv_i = 1.0 / (0.5 * mass * r * r)
        self.held = False
        self.scored = False
        self.z = 0.0
        self.side, self.char, self.value, self.id = side, char, value, pid
        self.owner = None


def obb_contact(px, py, pr, ox, oy, oh, hl, hw):
    c, s = math.cos(oh), math.sin(oh)
    dx, dy = px - ox, py - oy
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    qx = max(-hl, min(hl, lx))
    qy = max(-hw, min(hw, ly))
    nx, ny = lx - qx, ly - qy
    d = math.hypot(nx, ny)
    if d > pr:
        return None
    if d > 1e-6:
        nx, ny = nx / d, ny / d
        pen = pr - d
    else:
        dxf, dyf = hl - abs(lx), hw - abs(ly)
        if dxf < dyf:
            nx, ny = (1.0 if lx > 0 else -1.0), 0.0
            pen = pr + dxf
            qx, qy = math.copysign(hl, lx), ly
        else:
            nx, ny = 0.0, (1.0 if ly > 0 else -1.0)
            pen = pr + dyf
            qx, qy = lx, math.copysign(hw, ly)
    wx = c * nx - s * ny
    wy = s * nx + c * ny
    cx = ox + c * qx - s * qy
    cy = oy + s * qx + c * qy
    return wx, wy, pen, cx, cy


def _disc_disc(a, b):
    dx, dy = b.x - a.x, b.y - a.y
    d2 = dx * dx + dy * dy
    need = a.r + b.r
    if d2 >= need * need or d2 < 1e-12:
        return
    d = math.sqrt(d2)
    nx, ny = dx / d, dy / d
    pen = need - d
    rvx, rvy = b.vx - a.vx, b.vy - a.vy
    vn = rvx * nx + rvy * ny
    if vn < 0:
        j = -(1.0 + E_PIECE) * vn / (a.inv_m + b.inv_m)
        a.vx -= j * nx * a.inv_m
        a.vy -= j * ny * a.inv_m
        b.vx += j * nx * b.inv_m
        b.vy += j * ny * b.inv_m
        tx, ty = -ny, nx
        vt = rvx * tx + rvy * ty
        jt = max(-0.3 * j, min(0.3 * j, -vt / (a.inv_m + b.inv_m)))
        a.vx -= jt * tx * a.inv_m
        a.vy -= jt * ty * a.inv_m
        b.vx += jt * tx * b.inv_m
        b.vy += jt * ty * b.inv_m
    corr = pen * 0.5
    a.x -= nx * corr
    a.y -= ny * corr
    b.x += nx * corr
    b.y += ny * corr


def _disc_obb(p, ox, oy, oh, hl, hw, ovx, ovy, ow):
    hit = obb_contact(p.x, p.y, p.r, ox, oy, oh, hl, hw)
    if not hit:
        return
    nx, ny, pen, cx, cy = hit
    vcx = ovx - ow * (cy - oy)
    vcy = ovy + ow * (cx - ox)
    rvx, rvy = p.vx - vcx, p.vy - vcy
    vn = rvx * nx + rvy * ny
    if vn < 0:
        j = -(1.0 + E_PIECE) * vn * p.m
        p.vx += j * nx * p.inv_m
        p.vy += j * ny * p.inv_m
        tx, ty = -ny, nx
        vt = rvx * tx + rvy * ty
        jt = max(-0.4 * j, min(0.4 * j, -vt * p.m))
        p.vx += jt * tx * p.inv_m
        p.vy += jt * ty * p.inv_m
        torque = (cx - p.x) * (jt * ty) - (cy - p.y) * (jt * tx)
        p.w += torque * p.inv_i
    p.x += nx * pen * 0.85
    p.y += ny * pen * 0.85


class World:
    def __init__(self, bodies):
        self.bodies = bodies

    def step(self, dt, car, jaws, goal_box, score_cb=None):
        sdt = dt / 4.0
        for _ in range(4):
            for p in self.bodies:
                if p.scored or p.held:
                    continue
                if p.z <= 0.01:
                    sp = math.hypot(p.vx, p.vy)
                    if sp > 0:
                        k = max(0.0, sp - MU_GROUND * G_MM * sdt) / sp
                        p.vx *= k
                        p.vy *= k
                    p.w *= 0.95
                p.x += p.vx * sdt
                p.y += p.vy * sdt
                p.a += p.w * sdt
            n = len(self.bodies)
            for _it in range(2):
                for i in range(n):
                    a = self.bodies[i]
                    if a.scored or a.held:
                        continue
                    for j in range(i + 1, n):
                        b = self.bodies[j]
                        if b.scored or b.held:
                            continue
                        _disc_disc(a, b)
            oh = math.radians(car.h)
            for p in self.bodies:
                if p.scored or p.held:
                    continue
                _disc_obb(p, car.x, car.y, oh, 95.0, 90.0, car.vx, car.vy, car.w)
            for jw in jaws:
                ox, oy, oh2, hl, hw = jw
                for p in self.bodies:
                    if p.scored or p.held:
                        continue
                    _disc_obb(p, ox, oy, oh2, hl, hw, car.vx, car.vy, car.w)
            for p in self.bodies:
                if p.scored or p.held:
                    continue
                if p.x < p.r:
                    p.x, p.vx = p.r, abs(p.vx) * 0.2
                if p.x > 3000.0 - p.r:
                    p.x, p.vx = 3000.0 - p.r, -abs(p.vx) * 0.2
                if p.y < p.r:
                    p.y, p.vy = p.r, abs(p.vy) * 0.2
                if p.y > 2000.0 - p.r:
                    p.y, p.vy = 2000.0 - p.r, -abs(p.vy) * 0.2
        for p in self.bodies:
            if p.scored or p.held or p.z > 2.0:
                continue
            x0, x1, y0, y1 = goal_box
            if p.x - p.r >= x0 - 0.5 and p.x + p.r <= x1 and p.y - p.r >= y0 and p.y + p.r <= y1:
                p.scored = True
                if score_cb:
                    score_cb(p)
