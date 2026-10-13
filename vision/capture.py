# -*- coding: utf-8 -*-
"""相机取图：单张抓拍 / 连续预览 / 录视频。

用法：
    python vision/capture.py --save              # 拍一张存到 data/raw/ 然后退出
    python vision/capture.py                     # 实时预览，按键：s 存图  r 录/停  q 退出
    python vision/capture.py -c 1 --width 1280 --height 720
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import config  # noqa: E402


def open_camera(index, width, height, fps=30):
    """打开相机。Windows 上优先用 DSHOW，成功率最高。"""
    for backend, _name in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "默认")):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, frame
            cap.release()
    raise RuntimeError(f"打不开 {index} 号相机，先用 check_env.py 扫一下")


def stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def main():
    ap = argparse.ArgumentParser(description="相机取图")
    ap.add_argument("-c", "--camera", type=int, default=0, help="相机编号")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--save", action="store_true", help="只拍一张然后退出")
    ap.add_argument("--out", default=str(config.DIR_RAW), help="保存目录")
    ap.add_argument("--name", default=None, help="文件名（默认按时间戳）")
    args = ap.parse_args()

    cap, _ = open_camera(args.camera, args.width, args.height)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("按 s 存图，按 r 开始/停止录像，按 q 退出")

    writer = None
    frames = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("读帧失败")
            break
        frames += 1
        fps_now = frames / max(time.time() - t0, 1e-6)

        if args.save:
            name = args.name or f"shot_{stamp()}.png"
            common.imwrite_cn(out_dir / name, frame)
            break

        view = frame.copy()
        view = common.put_text_cn(
            view,
            f"{frame.shape[1]}x{frame.shape[0]}  {fps_now:4.1f} FPS   REC={'ON' if writer else 'OFF'}",
            (10, 8), 22, (0, 255, 255),
        )
        cv2.imshow("capture (s=save r=record q=quit)", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            name = args.name or f"shot_{stamp()}.png"
            common.imwrite_cn(out_dir / name, frame)
        if key == ord("r"):
            if writer is None:
                path = out_dir / f"rec_{stamp()}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(path), fourcc, 25.0,
                                         (frame.shape[1], frame.shape[0]))
                print("[录像开始]", path)
            else:
                writer.release()
                writer = None
                print("[录像结束]")
        if writer is not None:
            writer.write(frame)

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
