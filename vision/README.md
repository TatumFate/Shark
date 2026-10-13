# vision · 赛题一视觉工程

目标：把"场上哪儿有棋子、是哪一方的、是什么棋、值多少分"稳定地算出来，
后面再叠加定位、路径、推棋策略。

## 一、环境

| 项目 | 值 |
| --- | --- |
| 解释器 | `D:\新建文件夹\Python312\python.exe`（Python 3.12.10，免安装版） |
| 依赖 | numpy 2.5 / opencv-python 5.0 / pillow 12.3，已装好 |
| 重装依赖 | 终端里跑 `python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` |

> 为什么不用 `D:\新建文件夹\python.exe`？那是 Python 3.14.0a1 的 **alpha 版**，
numpy 的 cp314 轮子是按正式版 ABI 编译的，加载时会报
`DLL load failed while importing _multiarray_umath`。已实测确认，换 3.12 解决。
> 仓库里如果还有一个 `.venv` 目录，那是我第一次尝试留下的空壳（基于 3.14 创建的），
已经在 `.gitignore` 里，直接手动删掉即可。

## 二、文件一览

| 文件 | 作用 | 典型命令 |
| --- | --- | --- |
| `config.py` | **所有可调参数**：场地尺寸、棋子直径、分值表、阈值、字体 | 改完直接重跑 |
| `common.py` | 公共工具：单应矩阵读写、俯视校正、画中文、字形归一化、NCC 打分 | 被其它脚本 import |
| `check_env.py` | 环境自检：版本、字体、扫描相机 | `python vision/check_env.py` |
| `capture.py` | 相机取图：抓拍/预览/录像 | `python vision/capture.py --save` |
| `make_synthetic.py` | **合成测试图**（不用相机就能开发） | `python vision/make_synthetic.py` |
| `calib_field.py` | 地面单应标定（像素 ↔ 毫米） | `python vision/calib_field.py --image <图> --mode auto --preview` |
| `char_templates.py` | 生成汉字模板（含繁体写法、两种字体） | `python vision/char_templates.py` |
| `detect_pieces.py` | 棋子检测 + 红黑判断 | `python vision/detect_pieces.py --image <图> --mode gt --compare-gt` |
| `recognize_chars.py` | 棋子汉字识别（模板匹配 + 角度搜索） | `python vision/recognize_chars.py --image <图> --mode gt --compare-gt` |
| `pipeline.py` | **端到端主入口**：图 → 校正 → 检测 → 识别 → 结果图/JSON/统计 | `python vision/pipeline.py --image <图> --mode gt --compare-gt` |

数据目录（都在 `.gitignore` 里，不进仓库；`calib/` 例外，标定结果要入库）：

```
vision/data/
├─ raw/         相机原图、录像
├─ synthetic/   合成测试图 + 真值（gt_pieces.json / gt_homography.json）
├─ calib/       单应矩阵标定结果（homography_field.json / homography_robot.json）
├─ output/      结果图、结果 JSON、校正预览
└─ templates/   字符模板 templates.npz + preview.png
```

## 三、最小上手路径

```powershell
# 0) 环境自检（确认解释器和 OpenCV 正常）
python vision/check_env.py

# 1) 生成合成测试图（不需要相机，直接就能开发）
python vision/make_synthetic.py

# 2) 生成汉字模板
python vision/char_templates.py

# 3) 端到端跑一遍，并和真值比对精度
python vision/pipeline.py --image vision/data/synthetic/field_oblique.png --mode gt --compare-gt
```

跑完看这几张图（VS Code 里点开就能看）：

- `vision/data/synthetic/field_oblique.png` —— 模拟相机拍到的场地
- `vision/data/output/pipeline_rectified.png` —— 校正后的俯视图
- `vision/data/output/pipeline_annotated.png` —— 检测 + 识别的标注结果

## 四、当前基线（在合成图上实测）

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| 检出率 / 查全率 | 100%（32/32） | Hough 多组参数合并 + NMS |
| 查准率 | 100% | 用"盘面里有没有笔画"过滤误检 |
| 定位误差 | 平均 1.6 mm | 相对真值 |
| 红/黑判断 | 100% | 逐棋子自适应阈值 |
| 汉字识别 | 78.1%（25/32） | 模板匹配基线，**错误全部集中在细笔画的低对比度黑棋** |
| 单帧耗时 | 约 0.9 s（检测 0.55 s + 识别 0.28 s） | 6M 像素校正图；`--mm-per-px 2` 可提速数倍 |

汉字识别的 7 个错误全是"字细 + 分辨率低 + 对比度差"的黑棋被误判成 `象`，
分数普遍只有 0.44~0.55，和次优只差 0.02~0.07。
这是模板匹配的天然短板，也是下一步升级的动机（见第六节）。

## 五、数据流

```
相机图
  │  calib_field.py 得到的单应矩阵 H
  ▼
俯视校正 (rectify)  ←── 1 mm/像素，整场地 3000×2000
  │
  ├─ Hough / 轮廓 找出所有圆形 → 圆心+半径
  │      └─ 逐棋子自适应阈值 → 红/黑
  ▼
每个棋子的 (x_mm, y_mm, r_mm, side)
  │  取盘面小图 → 24 个角度旋转 → 二值化 → 归一化 48×48
  │  与"同色候选字"模板做归一化互相关（矩阵乘法一次算完）
  ▼
(char, value, score, margin)
  ▼
结果 JSON + 标注图 + 统计（红黑分值、优先目标列表）
```

## 六、下一步（按优先级）

1. **用真棋子 + 真相机跑一遍**：`__pycache__` 之外最重要的一步。
   把实物棋子直径量准，改 `config.PIECE_DIA`；再拍几十张不同光照的照片，看阈值要不要调。
2. **标定真实相机**：`calib_field.py --mode pick`，在车前方铺一张已知尺寸的板子点四个角，
   得到 `homography_robot.json`（车体坐标系）。这样检测结果就是"车前方多少毫米"。
3. **攒数据、上小 CNN**：模板法已到天花板。写个小工具把每枚棋子的盘面小图按识别结果存下来，
   人工核对后就是训练集（几百张就能训），换 MobileNetV3-Small / 自建 5 层 CNN，
   目标识别率 ≥98%，并导出 ONNX 部署。
4. **光电加分项**：低角度掠射照明（强化凹刻字的阴影对比）、偏振片压反光、
   可选线激光三角测量。这部分既提识别率，又是评审眼里的"光电含量"。
5. **接车**：把 `pipeline.py` 里的"读图"换成"读相机帧"，
   输出通过串口/CAN 发给 STM32；再加状态机与推棋策略。

## 七、坑（都已经踩过并处理）

1. **`cv2.imread` / `cv2.imwrite` 遇到中文路径会静默失败**（返回 None / 不报错）。
   本项目统一用 `common.imread_cn / imwrite_cn`（先编解码再用 Python 读写文件）。
2. **预发布版 Python 装不上科学计算库**，见第一节。
3. **固定灰度阈值会漏检**：相机分辨率一低、笔画一细，笔画灰度会从 30 涨到 120。
   所以这里用"相对该棋子盘面亮度"的自适应阈值（`config.INK_DELTA`）。
4. **二值化会连棋子外圈、装饰环一起算成笔画**，导致所有棋子长得一样。
   解决：用连通域分析剔除贴边的笔画（`recognize_chars.ink_patch`）。
5. **Hough 单组参数漏检严重**，所以跑 3 组参数合并后 NMS 去重。
