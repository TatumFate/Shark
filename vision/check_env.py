# -*- coding: utf-8 -*-
"""环境自检：确认解释器、OpenCV、相机都能用。

用法（VS Code 终端里）：
    python vision/check_env.py
"""
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def main():
    print("=" * 60)
    print("Python      :", sys.version.split()[0], "|", platform.machine())
    print("解释器路径  :", sys.executable)
    try:
        import cv2
        import numpy as np
        import PIL
        print("OpenCV      :", cv2.__version__)
        print("NumPy       :", np.__version__)
        print("Pillow      :", PIL.__version__)
    except Exception as e:                     # noqa: BLE001
        print("!! 依赖缺失：", type(e).__name__, e)
        print("   修复： python -m pip install -r requirements.txt")
        return 1

    if "a" in sys.version.split()[0] or "b" in sys.version.split()[0]:
        print("!! 警告：当前是预发布版 Python，科学计算库容易加载失败，建议用 3.12")

    # 字体
    fonts = [config.FONT_CN] + list(config.FONT_QI)
    for f in fonts:
        print(("字体 OK   : " if Path(f).exists() else "字体缺失  : ") + f)

    # 相机
    print("-" * 60)
    print("扫描相机（0-5 号）…")
    # 没有相机时 OpenCV 会在底层狂刷 WARN，先把日志级别调静音，扫完再恢复
    try:
        cv2.setLogLevel(0)
    except Exception:                          # noqa: BLE001
        pass
    found = []
    for idx in range(6):
        cap = None
        for backend, name in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "默认")):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                frames = []
                for _ in range(3):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        frames.append(frame)
                if frames:
                    # 有些虚拟设备会"打开成功但全是黑屏"，用灰度标准差筛一下
                    import numpy as np
                    import cv2 as _cv2
                    stds = [float(_cv2.cvtColor(f, _cv2.COLOR_BGR2GRAY).std()) for f in frames]
                    best = max(stds)
                    if best < 3:
                        print(f"  设备 {idx}（{name}）能打开但画面全黑，可能是虚拟摄像头，跳过")
                    else:
                        print(f"  相机 {idx} 可用（{name}）："
                              f"{frames[-1].shape[1]}x{frames[-1].shape[0]}  画面变化度 {best:.0f}")
                        found.append(idx)
                        cap.release()
                        cap = None
                        break
            cap.release()
        if cap is None:
            continue
    if not found:
        print("  没找到相机。检查：USB 是否插好 / 是否被别的程序（如相机App）占用。")
    try:
        cv2.setLogLevel(3)
    except Exception:                          # noqa: BLE001
        pass

    print("=" * 60)
    print("自检结束。下一步：python vision/capture.py --save")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
