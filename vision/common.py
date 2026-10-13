# -*- coding: utf-8 -*-
"""公共工具：读写 JSON/单应矩阵、俯视校正、画中文、坐标换算。"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 让 `python vision/xxx.py` 和「在 VS Code 里直接 F5」两种情况都能 import config
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# ------------------------------------------------------------------ JSON
def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def homography_path(frame=None):
    frame = frame or config.FRAME
    return config.DIR_CALIB / f"homography_{frame}.json"


def load_homography(frame=None, path=None):
    """读取标定结果，返回 (H_image2frame, meta)；没有标定文件时返回 (None, {})。"""
    path = Path(path) if path else homography_path(frame)
    if not path.exists():
        return None, {}
    meta = load_json(path)
    H = np.array(meta["H"], dtype=np.float64)
    return H, meta


def save_homography(H, meta, frame=None, path=None):
    path = Path(path) if path else homography_path(frame)
    payload = dict(meta or {})
    payload["H"] = np.asarray(H, dtype=np.float64).tolist()
    return save_json(payload, path)


# ------------------------------------------------------------------ 俯视校正
def rectify(img, H_image2frame, mm_per_px=None, size_mm=None):
    """把相机图按单应矩阵校正成俯视图（像素 = frame 毫米 / mm_per_px）。

    H_image2frame：图像像素 → 坐标系毫米（场地坐标或车体坐标）
    返回 (校正后的图, 实际使用的 mm_per_px)
    """
    mm_per_px = float(mm_per_px or config.MM_PER_PX)
    w_mm, h_mm = size_mm or (config.FIELD_L, config.FIELD_W)
    out_w, out_h = int(round(w_mm / mm_per_px)), int(round(h_mm / mm_per_px))
    S = np.array([[1.0 / mm_per_px, 0, 0], [0, 1.0 / mm_per_px, 0], [0, 0, 1]], np.float64)
    H_px2out = S @ np.asarray(H_image2frame, np.float64)
    return cv2.warpPerspective(img, H_px2out, (out_w, out_h)), mm_per_px


def px_to_frame(pts_px, H_image2frame):
    """图像像素点 → 坐标系毫米点。pts_px 形状 (N,2)。"""
    pts = np.asarray(pts_px, np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, np.asarray(H_image2frame, np.float64))
    return out.reshape(-1, 2)


def output_px_to_mm(pts_out_px, mm_per_px):
    """校正图上像素 → 毫米（校正图原点就是坐标系原点）。"""
    return np.asarray(pts_out_px, np.float64) * float(mm_per_px)


# ------------------------------------------------------------------ 中文绘制
_FONT_CACHE = {}


def _font(size, path=None):
    path = path or config.FONT_CN
    key = (path, size)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(path, size)
        except OSError:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def put_text_cn(img, text, org, size=22, color=(0, 0, 255), font_path=None):
    """在 OpenCV 图上写中文（OpenCV 自带 putText 画不了汉字，借 PIL 画）。

    img: BGR 图（原地修改并返回）；org: 文本左上角 (x, y)；color: BGR
    """
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text((int(org[0]), int(org[1])), text, font=_font(size, font_path),
              fill=(int(color[2]), int(color[1]), int(color[0])))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def imread_cn(path, flags=cv2.IMREAD_COLOR):
    """读图（支持中文路径）。

    坑：cv2.imread 在 Windows 上遇到中文路径会直接返回 None，
    所以先按二进制读进来再用 imdecode 解码。
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_cn(path, img):
    """写图（支持中文路径）+ 打印路径。

    坑：cv2.imwrite 在中文路径下同样会静默失败，所以先编码再用 Python 写文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, buf = cv2.imencode(suffix, img)
    if not ok:
        raise RuntimeError(f"编码失败：{path}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    print(f"[保存] {path}")
    return path


# ------------------------------------------------------------------ 其它
def show_or_save(window, img, out_path=None):
    """有界面就弹窗显示，没界面（或按 q）就返回；同时保存一份。"""
    if out_path:
        imwrite_cn(out_path, img)
    if not sys.stdin.isatty():
        return
    try:
        cv2.imshow(window, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def draw_field_overlay(img, mm_per_px, size_mm=None, color=(0, 200, 0)):
    """在俯视图上画出场地分区线（每 0.5m 一格 + 球门区 + 堆放区）。"""
    w_mm, h_mm = size_mm or (config.FIELD_L, config.FIELD_W)
    out = img.copy()
    step_px = int(round(500 / mm_per_px))
    for x in range(0, int(round(w_mm / mm_per_px)) + 1, step_px):
        cv2.line(out, (x, 0), (x, out.shape[0]), color, 1)
    for y in range(0, int(round(h_mm / mm_per_px)) + 1, step_px):
        cv2.line(out, (0, y), (out.shape[1], y), color, 1)
    return out


# ------------------------------------------------------------------ 字形归一化
def crop_to_ink(binary, thresh=40):
    """裁到笔画外接框，去掉四周空白，让不同大小的字可比。"""
    ys, xs = np.nonzero(binary > thresh)
    if len(xs) == 0:
        return binary
    return binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def normalize_patch(patch, size, pad_ratio=0.12):
    """把任意大小的笔画图缩放并居中成 size×size（保持长宽比不变形）。"""
    patch = crop_to_ink(patch)
    h, w = patch.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), np.uint8)
    inner = max(1, int(size * (1 - 2 * pad_ratio)))
    scale = inner / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    res = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = res
    return out


def ncc_scores(query, templates):
    """归一化互相关：query (n, N) 与 templates (m, N) 逐对打分，返回 (n, m)。

    等价于 cv2.matchTemplate 的 TM_CCOEFF_NORMED，但一次矩阵乘法就能算完，
    对"多角度 × 多模板"这种成百上千次比较快得多。
    """
    q = np.asarray(query, np.float32).reshape(len(query), -1)
    t = np.asarray(templates, np.float32).reshape(len(templates), -1)
    q = q - q.mean(axis=1, keepdims=True)
    t = t - t.mean(axis=1, keepdims=True)
    q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    return q @ t.T
