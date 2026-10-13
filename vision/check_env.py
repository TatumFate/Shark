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
import common  # noqa: E402


def looks_blocked(frame):
    """识别 Windows 的"摄像头被屏蔽"占位图。

    实测那张占位图是深蓝底 + 带斜杠相机图标：B/G/R 均值 ≈ 81/43/37，
    蓝色主导像素占 61%，整体很暗。真实场景偶尔也会偏蓝偏暗，
    所以这里只当作"提醒"，同时会把预览图存下来让人眼确认。
    """
    import numpy as np
    b = frame[..., 0].astype(np.int16)
    g = frame[..., 1].astype(np.int16)
    r = frame[..., 2].astype(np.int16)
    blue = (b > r + 25) & (b > g + 10)
    return float(blue.mean()) > 0.55 and float(frame.mean()) < 90


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
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:                          # noqa: BLE001
        pass
    import numpy as np
    found = []
    blocked_reported = set()
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
                    # 有些设备会"打开成功但没有真实画面"，这里做两级判断
                    stds = [float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).std()) for f in frames]
                    best = max(stds)
                    last = frames[-1]
                    prev = config.DIR_OUTPUT / f"camera_{idx}_preview.png"
                    common.imwrite_cn(prev, last)
                    blocked = looks_blocked(last)
                    if blocked:
                        if idx not in blocked_reported:
                            blocked_reported.add(idx)
                            print(f"  设备 {idx}（{name}）返回的是**系统屏蔽占位图**：")
                            print(f"      {frame.shape[1]}x{frame.shape[0]}，预览已存到 {prev}")
                            print("      这不是代码问题，而是 Windows 没给这个账号摄像头权限。")
                            print("      解决：设置 → 隐私和安全性 → 相机 → 分别打开"
                                  "「相机访问」和「让桌面应用访问你的相机」；")
                            print("      再检查：笔记本的物理遮挡/快捷键、以及是否有别的程序"
                                  "（相机App/OBS/会议软件）正占用摄像头。")
                        found.append(("blocked", idx))
                    elif best < 3:
                        print(f"  设备 {idx}（{name}）能打开但画面全黑，可能是虚拟摄像头，跳过")
                    else:
                        print(f"  相机 {idx} 可用（{name}）："
                              f"{last.shape[1]}x{last.shape[0]}  画面变化度 {best:.0f}"
                              f"  预览已存到 {prev}")
                        found.append(("ok", idx))
                        cap.release()
                        cap = None
                        break
            cap.release()
        if cap is None:
            continue
    if not any(k == "ok" for k, _ in found):
        if any(k == "blocked" for k, _ in found):
            print("  结论：能找到摄像头设备，但拿到的是屏蔽占位图 → 先按上面的隐私设置处理。")
        else:
            print("  没找到相机。检查：USB 是否插好 / 是否被别的程序（如相机App）占用。")
    else:
        print("  结论：至少一路相机能出真实画面，可以跑 capture.py 了。")
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_WARNING)
    except Exception:                          # noqa: BLE001
        pass

    print("=" * 60)
    print("自检结束。下一步：python vision/capture.py --save")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
