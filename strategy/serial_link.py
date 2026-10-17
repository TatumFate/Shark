# -*- coding: utf-8 -*-
"""串口链路：上位机（Python）↔ 下位机（STM32）。

只做三件事：
    1. 发指令：把 Command 编码成一行 `$GOTO,1200,800,180*B1\n` 写出去
    2. 收位姿：后台线程读 `$P,x,y,h,done*XX`，校验和不对就丢掉
    3. 报告状态：pose() / is_done() / age_s()（多久没收到心跳）

为什么要看"多久没收到心跳"：串口线松了、单片机死机了，程序不能傻等——
超时就该停车（由调用方根据 age_s() 判断）。

依赖：pyserial（python -m pip install pyserial）。没装时 open() 会给出明确提示。
"""
import threading
import time


def checksum(body):
    return sum(body.encode("ascii", "ignore")) & 0xFF


def parse_pose_line(line):
    """解析 `$P,x,y,h,done*XX`，返回 dict 或 None（格式/校验不对就 None）。"""
    line = line.strip()
    if not line.startswith("$") or "*" not in line:
        return None
    body, _, chk = line[1:].partition("*")
    try:
        if int(chk, 16) != checksum(body):
            return None
    except ValueError:
        return None
    parts = body.split(",")
    if len(parts) < 5 or parts[0] != "P":
        return None
    try:
        return {"x": float(parts[1]), "y": float(parts[2]), "h": float(parts[3]),
                "done": parts[4].strip() == "1"}
    except ValueError:
        return None


class SerialLink:
    def __init__(self, port, baud=115200, timeout=0.05):
        self.port = port
        self.baud = int(baud)
        self.timeout = timeout
        self.ser = None
        self._pose = (0.0, 0.0, 0.0)
        self._done = True
        self._t_last = 0.0
        self._last_cmd = None
        self._stop = False
        self.rx_lines = 0
        self.rx_bad = 0
        self.tx_frames = 0

    # ---------------------------------------------------------------- 打开
    def open(self):
        try:
            import serial  # pyserial
        except ImportError:
            raise SystemExit("没装 pyserial。在终端里跑：python -m pip install pyserial")
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        except Exception as e:
            ports = []
            try:
                from serial.tools import list_ports
                ports = [p.device for p in list_ports.comports()]
            except Exception:
                pass
            raise SystemExit(f"打不开串口 {self.port}：{e}\n"
                             f"当前电脑上可用的串口：{ports or '一个都没有（USB转TTL没插/没装驱动？）'}")
        self._t_last = time.time()
        threading.Thread(target=self._reader, daemon=True).start()
        return self

    # ---------------------------------------------------------------- 发
    def send(self, cmd):
        """发一条指令。和上一条完全一样就不重复发（幂等，避免把任务重置）。"""
        if self.ser is None:
            return
        s = str(cmd)
        if s == self._last_cmd:
            return
        self.ser.write((s + "\n").encode("ascii"))
        self._last_cmd = s
        self.tx_frames += 1

    def stop(self):
        self._last_cmd = None
        self.send_raw("$STOP*" + "%02X" % checksum("STOP"))

    def send_raw(self, line):
        if self.ser is not None:
            self.ser.write((line + "\n").encode("ascii"))

    # ---------------------------------------------------------------- 收
    def _reader(self):
        buf = b""
        while not self._stop and self.ser is not None:
            try:
                chunk = self.ser.read(64)
            except Exception:
                time.sleep(0.05)
                continue
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                p = parse_pose_line(raw.decode("ascii", "ignore"))
                if p is None:
                    self.rx_bad += 1
                    continue
                self._pose = (p["x"], p["y"], p["h"])
                self._done = p["done"]
                self._t_last = time.time()
                self.rx_lines += 1

    # ---------------------------------------------------------------- 查询
    def pose(self):
        return self._pose

    def is_done(self):
        return self._done

    def age_s(self):
        """距上次收到心跳多少秒。>0.5s 说明链路可能断了。"""
        return time.time() - self._t_last if self._t_last else 999.0

    def close(self):
        self._stop = True
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
