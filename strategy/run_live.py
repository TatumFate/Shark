# -*- coding: utf-8 -*-
"""实车入口：视觉 → 决策 → 串口 → 下位机，并把全过程写成日志。

三种用法（从"什么都不接"到"真车"）：
    1) 只跑视觉+决策，不发指令（先验证链路，最安全）
       python strategy/run_live.py --camera 0 --no-serial

    2) 回放录像/图片，不接车（调参用）
       python strategy/run_live.py --source vision/data/raw/demo.mp4 --no-serial --pose 1500 1000 180

    3) 真车（接上 STM32 之后）
       python strategy/run_live.py --camera 0 --port COM7

按键：空格=开始/暂停 · s=存一张标注图 · q=退出
日志：logs/live_YYYYmmdd_HHMMSS.jsonl（每帧一行，直接发给我就能分析）
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vision"))
import common  # noqa: E402
import config  # noqa: E402
import game  # noqa: E402
import realtime as rt  # noqa: E402
import recognize_chars as rc  # noqa: E402
from commands import idle, stop  # noqa: E402
from fsm import DecisionFSM  # noqa: E402

LINK_TIMEOUT_S = 0.5


def main():
    ap = argparse.ArgumentParser(description="实车：视觉+决策+串口")
    ap.add_argument("--source", default="0", help="相机号 / 视频 / 图片")
    ap.add_argument("--camera", type=int, default=None, help="等价于 --source <相机号>")
    ap.add_argument("--mode", choices=["calib", "gt"], default="calib")
    ap.add_argument("--frame-name", default=config.FRAME)
    ap.add_argument("--mm-per-px", type=float, default=1.5)
    ap.add_argument("--roi", type=float, default=800.0)
    ap.add_argument("--center", type=float, nargs=2, default=(500.0, 0.0))
    ap.add_argument("--port", default=None, help="串口，例如 COM7")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--no-serial", action="store_true", help="不发指令（只跑视觉+决策）")
    ap.add_argument("--pose", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=("X", "Y", "H"), help="没有串口时用的固定位姿")
    ap.add_argument("--frames", type=int, default=0, help="跑 N 帧就退出（0=一直跑）")
    ap.add_argument("--log-dir", default=str(HERE.parent / "logs"))
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--auto-start", action="store_true",
                    help="启动后立刻开始（录像回放/无窗口时用）")
    args = ap.parse_args()

    # ---- 视觉 ----
    if args.mode == "gt":
        H, _ = common.load_homography(path=config.DIR_SYN / "gt_homography.json")
    else:
        H, _ = common.load_homography(args.frame_name)
    if H is None:
        raise SystemExit("没有标定文件，先跑 vision/calib_field.py")
    src = rt.open_source(args.camera if args.camera is not None else args.source, 1280, 720)
    engine = rt.RealtimeVision(H, rc.load_templates(), args.mm_per_px, args.roi,
                               tuple(args.center))

    # ---- 串口 ----
    link = None
    if not args.no_serial:
        if not args.port:
            raise SystemExit("真车模式要给 --port COMx（不知道哪个口：设备管理器→端口）；"
                             "只想先跑通就用 --no-serial")
        from serial_link import SerialLink
        link = SerialLink(args.port, args.baud).open()
        print(f"[串口] {args.port} @ {args.baud}，等待下位机心跳…")
    else:
        print("[串口] 未启用（只跑视觉+决策，不发指令）")

    # ---- 决策 ----
    fsm = DecisionFSM()
    started = bool(args.auto_start)
    t0 = time.time() if started else 0.0
    if started:
        fsm.start(0.0)

    # ---- 日志 ----
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"live_{stamp}.jsonl"
    log_f = open(log_path, "w", encoding="utf-8")
    meta = {"type": "meta", "time": stamp, "mode": args.mode, "frame": args.frame_name,
            "mm_per_px": args.mm_per_px, "roi": args.roi, "center": list(args.center),
            "source": str(args.camera if args.camera is not None else args.source),
            "serial": args.port if link else None, "baud": args.baud,
            "piece_dia": config.PIECE_DIA, "blade_w": config.BLADE_WIDTH_MM,
            "match_seconds": config.MATCH_SECONDS, "python": sys.version.split()[0]}
    log_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    log_f.flush()
    print(f"[日志] {log_path}")

    # ---- 窗口 ----
    win = "run_live (space=start/pause s=save q=quit)"
    headless = args.no_window
    if not headless:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except cv2.error:
            headless = True
    writer = None
    frames = 0
    t_last_warn = 0.0
    try:
        while True:
            frame = src.read()
            if frame is None:
                break
            pieces, rect, origin, fresh, state = engine.process(frame)
            now = time.time() - t0 if started else 0.0
            pose = link.pose() if link else tuple(args.pose)
            done = link.is_done() if link else True
            link_age = link.age_s() if link else None

            cmd = idle()
            link_lost = (link is not None and link_age > LINK_TIMEOUT_S)
            if started and not link_lost:
                cmd = fsm.update(state, pose, now, done, memory=engine.memory)
                if link is not None:
                    link.send(cmd)
            elif started and link_lost:
                # 链路断了：冻结决策（不能让它"以为指令都完成了"继续往下跑），只发停车
                cmd = stop()
                link.stop()
                if time.time() - t_last_warn > 1.0:
                    t_last_warn = time.time()
                    print(f"[警告] {LINK_TIMEOUT_S}s 没收到下位机心跳，已发 STOP，决策冻结")

            st = (fsm.status(now) if started else {"state": "IDLE", "remain": config.MATCH_SECONDS})
            if link_lost:
                st = dict(st, state="LINK_LOST")
            row = {
                "t": round(now, 3), "frame": frames,
                "state": st["state"], "cmd": str(cmd),
                "pose": [round(v, 1) for v in pose], "done": bool(done),
                "link_age": None if link_age is None else round(link_age, 3),
                "n_pieces": len(pieces), "n_known": state["n_known"],
                "n_memory": state.get("n_memory"),
                "cum_total": state.get("cum_total"), "in_view_value": state["in_view_value"],
                "brief": state["brief"],
                "plan": (state["plans"][0]["plan"] if state["plans"] else None),
                "ms": {k: round(v, 1) for k, v in engine.stats.items()},
            }
            log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            frames += 1
            if frames % 30 == 0:
                log_f.flush()

            if headless:
                time.sleep(0.001)
            else:
                fps = 1000.0 / max(engine.stats["total"], 1e-6)
                view = rt.draw_raw_overlay(frame.copy(), state, engine.to_src,
                                           fps, engine.stats)
                items = [
                    (f"决策 {st['state']}   剩余 {st.get('remain', 0):.0f}s   指令 {cmd}",
                     (8, view.shape[0] - 56), 18, (0, 255, 255)),
                    (("未开始：按空格开始" if not started else
                      (f"链路 {link_age:.2f}s" if link_age is not None else "无串口（只记录）")),
                     (8, view.shape[0] - 30), 18, (0, 255, 0)),
                ]
                view = common.put_texts_cn(view, items)
                cv2.imshow(win, view)
                if args.save_video:
                    if writer is None:
                        vpath = config.DIR_RAW / f"live_{stamp}.mp4"
                        writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 20.0, (view.shape[1], view.shape[0]))
                    writer.write(view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    if not started:
                        started = True
                        t0 = time.time()
                        fsm.start(0.0)
                        print("[开始] 自主运行")
                    else:
                        started = not started
                        if not started and link is not None:
                            link.stop()
                        print("[暂停]" if not started else "[继续]")
                if key == ord("s"):
                    common.imwrite_cn(config.DIR_OUTPUT / f"live_{stamp}_{frames:05d}.png", view)

            if args.frames and frames >= args.frames:
                break
    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        if link is not None:
            try:
                link.stop()
            except Exception:
                pass
            link.close()
        log_f.close()
        if writer is not None:
            writer.release()
        src.release()
        if not headless:
            cv2.destroyAllWindows()
        print(f"[结束] 共 {frames} 帧，日志：{log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
