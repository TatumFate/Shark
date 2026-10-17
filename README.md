# Shark

第十五届全国大学生光电设计竞赛 · 赛题一「智能车光电象棋对抗」参赛项目仓库。

## 目录结构

```
Shark/
├─ README.md            本文件
├─ ROADMAP.md           **任务清单与分支规范（先看这个）**
├─ docs/                硬件参数、测试记录（两个待填模板）
├─ requirements.txt     视觉工程依赖
├─ .vscode/             VS Code 配置：解释器、调试、推荐扩展、任务
├─ vision/              视觉工程（识别 + 实时 + 棋局理解，已跑通）
├─ strategy/            决策层（状态机 + 2D 仿真，已在仿真里跑到 182 分）
├─ firmware/            STM32 执行层骨架（协议 + 差速 + 里程计，待上板验证）
└─ vision/README.md     视觉工程说明
```

各模块的脚本清单和用法见各自的 `README.md`；`mech/`（结构图纸）、
`tools/`（辅助脚本）按需创建。

## 从这里开始

1. 先看 [`ROADMAP.md`](ROADMAP.md)：当前该做什么、分支怎么用、目录怎么分。
2. 填 [`docs/硬件参数.md`](docs/硬件参数.md)：所有代码都依赖这里的数（棋子直径、轮径、相机位姿）。
3. 把自己的象棋录进系统：见 [`vision/README.md`](vision/README.md) 第十节（20 分钟出真实识别率）。
4. 没有硬件也想看决策效果：`python strategy/run_sim.py --video`（会录一段仿真比赛视频）。

## 环境（重要）

- **解释器**：`D:\新建文件夹\Python312\python.exe`（Python 3.12.10）
  —— 不要用 `D:\新建文件夹\python.exe`，那是 3.14.0a1 alpha 版，numpy/OpenCV 加载会失败。
- VS Code 打开本仓库后，装 `ms-python.python` 扩展，解释器选上面那个路径（已写进 `.vscode/settings.json`）。
- 依赖：`numpy`、`opencv-python`、`pillow`，见 `requirements.txt`。

## 快速开始

```powershell
python vision/check_env.py                                                    # 自检
python vision/make_synthetic.py                                               # 生成合成测试图
python vision/char_templates.py                                                # 生成汉字模板
python vision/pipeline.py --image vision/data/synthetic/field_oblique.png --mode gt --compare-gt
```

也可以直接在 VS Code 左侧「运行和调试」里选 `7 · 全流程（推荐从这里开始）` 按 F5。

当前基线（合成图）：检出 32/32、定位误差 1.6 mm、红黑 100%、汉字识别 78.1%。
详细说明见 [vision/README.md](vision/README.md)。
