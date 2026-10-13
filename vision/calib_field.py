# -*- coding: utf-8 -*-
"""地面单应标定：把"图像像素"换算成"毫米坐标"。

三种用法：
1) 自动找场地四角（对着整块场地的照片用，最快）
     python vision/calib_field.py --image vision/data/raw/shot.png --mode auto
2) 手动点选（真实车上最常用：在车前铺一张已知尺寸的标定板，点它的四个角）
     python vision/calib_field.py --image vision/data/raw/board.png --mode pick --frame robot --board 400 300
3) 用合成图自带的真值做对照（验证流程是否走通）
     python vision/calib_field.py --image vision/data/synthetic/field_oblique.png --mode gt

点选操作：鼠标左键依次点 4 个角，按 r 重来，按回车/空格确认，Esc 取消。
算完后加 --preview 可以看校正效果（叠加 0.5m 网格）。
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import config  # noqa: E402


def order_quad(pts):
    """把 4 个角点排成统一顺序：左上→右上→右下→左下（图像坐标，y 向下）。"""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    center = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(ang)                     # y 向下时，角度递增 = 视觉顺时针
    pts = pts[order]
    start = np.argmin(pts[:, 0] + pts[:, 1])     # 起点取最靠左上的角
    return np.roll(pts, -start, axis=0)


def detect_field_quad(img):
    """在"能看到整块场地"的照片里自动找场地内边界（挡板内侧）的 4 个角。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 棋子/标线会让地面出现空洞，先闭运算补齐，再取最大轮廓
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k, iterations=3)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("没找到亮色地面区域，改用 --mode pick 手动点角点")
    big = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(big)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.08, 15):
        quad = cv2.approxPolyDP(hull, eps * peri, True)
        if len(quad) == 4:
            return order_quad(quad.reshape(-1, 2))
    raise RuntimeError("轮廓不是四边形（可能场地没拍全 / 背景太乱），改用 --mode pick")


def pick_points(img, n=4):
    """鼠标点选 n 个点。"""
    pts = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x, y))
            print(f"  第 {len(pts)} 点：({x}, {y})")

    win = "calib: 左键点角点  r=重来  Enter=确认  Esc=取消"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = img.copy()
        for i, p in enumerate(pts):
            cv2.circle(view, p, 6, (0, 255, 255), -1)
            view = common.put_text_cn(view, str(i + 1), (p[0] + 8, p[1] - 26), 24, (0, 255, 255))
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            pts.clear()
            print("  已清空")
        elif key in (13, 32) and len(pts) >= 4:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("已取消")
    cv2.destroyAllWindows()
    return np.asarray(pts, np.float64)


def target_points(kind, board, rotate=0):
    """4 个角点对应的真实毫米坐标（顺序：左上→右上→右下→左下）。"""
    if kind == "field":
        w, h = config.FIELD_L, config.FIELD_W
    else:
        w, h = board
    pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    if rotate:
        pts = np.roll(pts, -int(rotate / 90), axis=0)
    return pts


def main():
    ap = argparse.ArgumentParser(description="地面单应标定")
    ap.add_argument("--image", required=True, help="标定用图片")
    ap.add_argument("--mode", choices=["auto", "pick", "gt"], default="pick")
    ap.add_argument("--frame", choices=["field", "robot"], default=config.FRAME,
                    help="field=整场地俯视坐标；robot=以车为中心的坐标")
    ap.add_argument("--board", type=float, nargs=2, default=(400.0, 300.0),
                    metavar=("W", "H"), help="robot 模式下标定板的宽高(mm)")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="映射不对（上下颠倒）时用它纠")
    ap.add_argument("--mm-per-px", type=float, default=config.MM_PER_PX)
    ap.add_argument("--preview", action="store_true", help="保存校正预览图")
    args = ap.parse_args()

    img = common.imread_cn(args.image)
    if img is None:
        raise SystemExit(f"读不到图片：{args.image}")
    print(f"图片 {img.shape[1]}x{img.shape[0]}  {args.image}")

    if args.mode == "gt":
        src_pts = None
        H, meta = common.load_homography(path=config.DIR_SYN / "gt_homography.json")
        meta = dict(meta)
        meta["source"] = "synthetic ground truth"
        print("使用合成图真值单应矩阵")
    else:
        if args.mode == "auto":
            src_pts = detect_field_quad(img)
            print("自动找到场地四角：", src_pts.astype(int).tolist())
        else:
            src_pts = pick_points(img, 4)
        dst_pts = target_points(args.frame, args.board, args.rotate)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            raise SystemExit("单应矩阵求解失败")
        err = np.abs(cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1, 2) - dst_pts)
        print(f"角点反算最大误差：{err.max():.1f} mm")
        meta = {"frame": args.frame, "board": list(args.board) if args.frame == "robot" else None,
                "image_size": [img.shape[1], img.shape[0]],
                "src_points": src_pts.tolist(), "source": args.image}

    size_mm = (config.FIELD_L, config.FIELD_W) if args.frame == "field" else tuple(args.board)
    rect, mm_per_px = common.rectify(img, H, args.mm_per_px, size_mm)
    common.save_homography(H, meta, frame=args.frame)
    print(f"标定已保存 → {common.homography_path(args.frame)}")
    print(f"整图 {size_mm[0]:.0f}x{size_mm[1]:.0f} mm，校正后 {mm_per_px} mm/像素")

    if args.preview:
        view = common.draw_field_overlay(rect, mm_per_px, size_mm)
        for i, p in enumerate(target_points(args.frame, args.board, args.rotate)):
            cv2.circle(view, tuple(np.int32(np.array(p) / mm_per_px)), 12, (0, 0, 255), 3)
        common.imwrite_cn(config.DIR_OUTPUT / "calib_preview.png", view)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
