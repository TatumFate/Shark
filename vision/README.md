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
| `realtime.py` | **实时识别**：相机/视频 → 车前视窗校正 → 检测 → 带预算识别 → 棋局理解 → 输出 | `python vision/realtime.py --source 0` |
| `game.py` | **棋局理解层**：区域判定、得分统计、推扫目标聚类、推进规划、串口帧 | `python vision/game.py`（自测） |

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

## 七、实时化（realtime.py）

比赛是实时的，但**算子完全相同**：`pipeline.py` 是"一帧"的版本，`realtime.py` 就是把它放进循环。
区别在三个工程改造，实测把单帧从 780 ms 压到 30 ms 左右：

| 改造 | 做法 | 效果 |
| --- | --- | --- |
| 只校正车前视窗 | `rectify_window()` 只把车前 800mm×800mm 校正出来，不校正整块 3m×2m 场地 | 校正 3ms → 0.4ms，检测 513ms → 29ms |
| 识别带预算 | 每帧最多新认 `--max-rec-per-frame` 枚，其余的沿用缓存 | 识别 264ms → 0.3ms（缓存填满后） |
| 结果走输出接口 | 每帧返回 `(pieces, ...)`，`top_targets()` 给出候选目标，`--emit-hz` 打 JSON | 接串口时只改一处 |

实测（视窗 800mm、1.5 mm/像素、视窗内塞满 32 枚棋子，属于最坏情况）：

```
mm/px=1.5，每帧新认 3 枚：平均 38 ms → 26 FPS（缓存填满后稳定 29 ms / 34 FPS）
mm/px=2.0，每帧新认 1 枚：平均 24 ms → 41 FPS
回放抖动视频（±6px 位移 + 噪声）：平均 34 ms → 29 FPS
```

三个必须知道的点：

1. **缓存匹配半径要比棋子直径小**（默认 `0.8 × PIECE_DIA`）。半径取大一点会把两枚相邻的
   棋子合并成一个缓存条目，症状是"缓存永远只有 26 条、有几枚棋子一直显示 ?"。
2. **同一帧内一个缓存条目只能被一枚棋子占用**，否则相邻棋子会互相抢身份。
3. **车体坐标系下缓存只在接近静止时有效**：车一动，同一枚棋子在车体系里的坐标就变了。
   正式上车时要用里程计/IMU 的位姿把检测结果换算到**场地坐标系**，再缓存；
   这一步等底盘能跑之后再接（`RealtimeVision.process` 里加一次坐标变换即可）。

## 八、棋局理解层（game.py）—— 视觉和"比赛"之间的那一层

视觉层只回答"哪儿有棋子、是哪一方的、是什么棋"。比赛真正要的是下面这些问题，
都在 `game.py` 里：

| 功能 | 函数 | 说明 |
| --- | --- | --- |
| 区域判定 | `classify_zone()` | 每枚棋子在己方球门区 / 对方球门区 / 棋子堆放区 / 对抗区 |
| 得分统计 | `count_value()` / `GoalCounter` | 见下面的"两种得分"说明 |
| **推扫目标聚类** | `cluster_pieces()` / `rank_clusters()` | 找"一铲能带走的一堆"：分值合计最大的簇优先，而不是最近的一枚 |
| 推进规划 | `plan_push()` | 给出对准点、车头朝向、推进距离——控制层直接照着走 |
| 给下位机的数据 | `serial_frame()` | `$PUSH,对准X,对准Y,朝向,推进距离,预期分值,枚数*校验` |
| 坐标换算 | `to_field_frame()` | 车体坐标 → 场地坐标（需要里程计位姿，`--pose x y theta`） |

### 两种"得分"别搞混

- `in_view_value`：**当前画面里**落在己方球门区内的棋子分值（只在投放那几帧有意义）。
- `GoalCounter.total`：**视野累计**——只要看到棋子出现在己方球门区就加一次（按位置去重）。
  这是真实得分的**下界**，用于策略和复盘；正式判分以主办方为准。
- 车头相机看不到整个球门区，所以"累计得分"一定要靠事件累加，不能靠单帧画面。

### 规则未定项（会影响这里的行为）

赛题原文只说"推动棋子整体进入己方球门区，按棋子对应分值累计得分"，没说是否区分颜色。
所以在 `config.py` 里留了两个开关，拿到细则后一行就能改：

```python
OWN_SIDE = None            # "red" / "black" / None（未定，先按不区分颜色统计）
SCORE_MODE = "own_color"   # own_color=只算本队颜色；any_color=推进去就算
```

`CLUSTER_R`（默认 220mm）定义"一铲"的概念，取值约等于推板有效宽度的一半；
单链聚类会把连片的棋子串成一大簇，所以如果你希望上层一次只处理一小片，
把它调到 120~150mm。

## 九、坑（都已经踩过并处理）

1. **`cv2.imread` / `cv2.imwrite` 遇到中文路径会静默失败**（返回 None / 不报错）。
   本项目统一用 `common.imread_cn / imwrite_cn`（先编解码再用 Python 读写文件）。
2. **预发布版 Python 装不上科学计算库**，见第一节。
3. **固定灰度阈值会漏检**：相机分辨率一低、笔画一细，笔画灰度会从 30 涨到 120。
   所以这里用"相对该棋子盘面亮度"的自适应阈值（`config.INK_DELTA`）。
4. **二值化会连棋子外圈、装饰环一起算成笔画**，导致所有棋子长得一样。
   解决：用连通域分析剔除贴边的笔画（`recognize_chars.ink_patch`）。
5. **Hough 单组参数漏检严重**，所以跑 3 组参数合并后 NMS 去重。

## 十、把你的实物象棋录进系统（详细步骤）

系统自带的字符模板是**用系统字体渲染**的楷体/黑体。你手里那副棋的字形、粗细、
油墨、反光都可能不同，所以识别率会打折扣。用你自己的棋子照片生成模板，
实测能把合成图上的 78% 顶到 96% 以上。整个流程 20 分钟能做完。

### 第 0 步：量直径（1 分钟，用卡尺或直尺）

量一枚棋子的直径，填进 `config.py`：

```python
PIECE_DIA = 44.0     # 改成你量到的值，常见 35 / 40 / 44 / 50
```

为什么必须做：检测器的搜索半径窗口是 `0.8~1.25 × PIECE_DIA/2`，
填错 20% 以上就会大量漏检或误检。

### 第 1 步：拍照片

要求（照做就能省掉后面所有麻烦）：

- 棋子**全部字朝上**，平铺在桌上，**彼此不重叠**，最好按颜色分成两堆；
- 俯拍（手机正对桌面拍），**镜头别太斜**，整盘棋子都在画面里；
- 光线均匀，别开闪光灯（会有强反光把字盖掉）；
- 画面里**放一把直尺**（后面可以点两点换算像素→毫米，不放也行）。

存到 `vision/data/raw/pieces.png`（VS Code 里直接拍照拖进来，或用 `capture.py`）。

### 第 2 步：一键切样本 + 量比例

```powershell
# 画面里有尺子：弹窗点尺子上 100mm 的两端（--ref-mm 100）
python vision/collect_samples.py --image vision/data/raw/pieces.png --scale-pick --ref-mm 100

# 已知比例（例如整场俯拍是 1mm/像素）：直接给
python vision/collect_samples.py --image vision/data/raw/pieces.png --mm-per-px 1.0

# 什么都不给也行，只是量不出毫米尺寸
python vision/collect_samples.py --image vision/data/raw/pieces.png
```

它会做四件事：自动找出每枚棋子 → 按 1.5×直径裁出小图 → 存到
`vision/data/pieces/samples/`（文件名形如 `unknown_red_000.png`，颜色已自动判好）
→ 拼一张索引图 `_index.png`。

> 尺子点选和相机抓帧都需要图形界面；如果是在没有桌面的环境里跑，去掉 `--scale-pick` 即可。

### 第 3 步：打标签（唯一需要动手的地方，约 10 分钟）

打开 `vision/data/pieces/samples/_index.png`（索引图里每张样本下面有编号，
红圈=红棋、白圈=黑棋），然后**把文件名开头的 `unknown` 改成棋子上的字**：

```
unknown_red_000.png   →   车_red_000.png
unknown_black_007.png →   卒_black_007.png
```

规则只有两条：**第 1 段是字，第 2 段是颜色**（脚本已经填好颜色，别改）；
不清楚的、拍糊的直接删掉。

在 VS Code 的资源管理器里点文件名按 `F2` 就能改，改完顺手看一眼索引图确认。

### 第 4 步：生成模板

```powershell
python vision/char_templates.py
```

看到 `[模板] real（你的实物棋子）：N 张样本` 就成功了。
它会把"你的实物字形模板"和"系统字体模板"一起写进 `templates.npz`。

### 第 5 步：看识别率

```powershell
python vision/recognize_chars.py --eval-samples vision/data/pieces/samples
```

输出形如 `识别正确率 96.9%（31/32）`，还会列出**偏弱的字**和**错例清单**，
照着补拍几张那几种字再重跑第 2~5 步即可。

### 两个重要提醒

1. **这个数字偏乐观**：模板和测试用的是同一批照片。想要诚实的数字，
   就拍两次：一批（比如 30 张）用来建模板，另一批（另外 10~20 张，
   换个角度/距离/光线）只用来评估。评估样本放进另一个目录，
   用 `--eval-samples <那个目录>` 就行（但不要把它并入 `--samples`）。
2. **样本要入库，模板不用**：`vision/data/pieces/samples/` 不会被 `.gitignore`
   忽略（队友拉到同一副棋的照片就能复现出一样的模板），
   而 `data/templates/` 是生成产物、不入库——换个队友 `char_templates.py` 跑一次就行。

### 后面可以再升级

样本攒到几百张（不同距离、光照、角度）之后，就可以拿它们训一个小 CNN
（`char_templates.py` 的同一批文件名就是现成标签），
再用 ONNX 部署到 RDK X5，把识别率从"90 多"推到"99 以上"。
现在这一套模板匹配的意义是：**先把流程和基线跑通，不阻塞其他模块**。

## 十一、小视野下怎么做决策（记忆 + 得分效率）

比赛时车上只有一个**前视相机**，一帧只看得到车前 0.5~1m 的一小块。
这带来三个后果，代码里都按这个前提设计：

**后果 1：不能用单帧做全局决策。**
所以有了 `game.FieldMemory`（场地记忆）：把"见过的棋子"按**场地坐标**记进台账，
规划时用的是整场台账，而不是当前这一帧。看到棋子进入己方球门区，就从台账划掉。

**后果 2：目标选择要跨帧、要能容忍"看不见"。**
`plan_from_memory()` 从台账里挑目标；同一枚棋子在视野里出现多次不会重复记账
（同色 + 距离 < `MEMORY_TOL_MM` 认为是同一枚）。

**后果 3：决策指标必须是"分/秒"，不是"分最高"或"最近"。**
`estimate_seconds()` 把一趟的耗时算出来：

```
一趟耗时 ≈ (开到目标 + 从目标推进球门区) / 巡航速度 + 对准耗时 + 每趟固定开销
得分效率 = 这一趟的总分 ÷ 一趟耗时        ← 按这个排序
```

实测示例（假数据，两个候选：3 枚共 17 分近处 vs 1 枚 9 分远处）：

```
3枚['炮','马','卒'] 17分 预计7.5s → 2.27 分/秒   ← 选它
2枚['车','兵']      10分 预计6.2s → 1.63 分/秒
1枚['车']            9分 预计9.1s → 0.99 分/秒
```

所以"先拿分高的还是先拿最近的"这个问题，正确答案是**先拿分/秒最高的那一趟**：
单枚 9 分跑一个来回，往往不如顺手一铲带走三枚 8 分的。

**需要按实测标定的三个参数**（跑一趟掐表，把实际耗时填回去）：

```python
ROBOT_SPEED_MM_S = 1200.0    # 实测巡航速度
ALIGN_SEC = 1.5              # 对准一次要多久
TRIP_OVERHEAD_S = 2.0        # 展开推板/抬铲/退出球门区的固定开销
```

这三个数填得准，`分/秒` 排序才靠谱——**这比再调视觉参数重要得多**。

**还缺的三块（按优先级）**：

1. **路径规划**：现在给的是"对准点 + 车头朝向 + 推进距离"三个点，
   还没有避障和绕行。3m×2m 的小场地里直线走就够用，
   需要补的是"对手车挡在路上怎么办"和"被撞歪之后怎么重定位"。
2. **对手意图**：检测对方车的位置与朝向，判断它会不会先抢走我正在瞄准的那一堆，
   以及它是否正冲向我的球门区。
3. **状态机**：搜索 → 对准 → 推扫 → 投放 → 撤离。`game.py` 输出的是"目标与规划"，
   把这个规划变成电机指令、并处理失败重试，是控制层的事。
