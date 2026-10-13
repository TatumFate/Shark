# Shark

第十五届全国大学生光电设计竞赛 · 赛题一「智能车光电象棋对抗」参赛项目仓库。

## 目录结构

```
Shark/
├─ README.md            本文件
├─ requirements.txt     视觉工程依赖
├─ .vscode/             VS Code 配置：解释器、调试、推荐扩展、任务
└─ vision/              视觉工程（当前进度：场地/棋子识别全流程已跑通）
   ├─ README.md         工程说明、脚本用法、当前指标、下一步
   ├─ config.py         所有可调参数
   ├─ common.py         单应校正、中文绘制、字形归一化等公共工具
   ├─ check_env.py      环境自检
   ├─ capture.py        相机取图
   ├─ make_synthetic.py 合成测试图（不用相机也能开发）
   ├─ calib_field.py    地面单应标定
   ├─ char_templates.py 汉字模板生成
   ├─ detect_pieces.py  棋子检测 + 红黑判断
   ├─ recognize_chars.py 汉字识别
   └─ pipeline.py       端到端主入口
```

后续还会加入 `firmware/`（STM32 底盘固件）、`docs/`（方案与参赛材料）、
`mech/`（结构图纸），按需创建。

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
