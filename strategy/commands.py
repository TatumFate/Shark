# -*- coding: utf-8 -*-
"""给下位机的运动指令：定义 + 串口编码。

协议（文本行，末尾 * 两位十六进制校验；115200 8N1）：
    $GOTO,x,y,heading*CHK     去(x,y)毫米并摆正到 heading 度（0=+x，顺时针为正）
    $TURN,heading*CHK         原地转到 heading 度
    $DRIVE,dist*CHK           直走 dist 毫米（负数=倒车）
    $BLADE,0|1*CHK            推板抬起/放下
    $STOP*CHK                 立即停（安全）
    $IDLE*CHK                 没有任务
下位机回传（建议）：
    $P,<x>,<y>,<heading>,<done 0|1>*CHK     位姿 + 当前指令是否完成
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    verb: str
    args: tuple = ()

    def body(self):
        parts = [self.verb]
        for a in self.args:
            if isinstance(a, float):
                parts.append(f"{a:.1f}".rstrip("0").rstrip("."))
            else:
                parts.append(str(a))
        return ",".join(parts)

    def encode(self):
        body = self.body()
        return f"${body}*{sum(body.encode()) & 0xFF:02X}"

    def __str__(self):
        return self.encode()

    @property
    def is_motion(self):
        return self.verb in ("GOTO", "TURN", "DRIVE")


def goto(x, y, heading_deg):
    return Command("GOTO", (round(float(x)), round(float(y)), round(float(heading_deg))))


def turn(heading_deg):
    return Command("TURN", (round(float(heading_deg)),))


def drive(dist_mm):
    return Command("DRIVE", (round(float(dist_mm)),))


def blade(down):
    return Command("BLADE", (1 if down else 0,))


def stop():
    return Command("STOP")


def idle():
    return Command("IDLE")
