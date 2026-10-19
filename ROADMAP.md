# ROADMAP · 赛题一备赛任务清单

更新：2026-10-17 ｜ 详细说明见《备赛总手册》（Codex 输出目录）
用法：每周一认领任务，周六实测日更新勾选和数字。

---

## 0. 一页看板：现在到哪了

| 模块 | 状态 | 关键文件 | 怎么验证 |
| --- | --- | --- | --- |
| 棋子检测 + 认字 | ✅ 可用 | `vision/detect_pieces.py`、`recognize_chars.py` | `python vision/test_vision.py` |
| 实时识别（车前视窗） | ✅ 40 FPS | `vision/realtime.py` | `realtime.py --bench 20` |
| 棋局理解（区域/聚类/分秒排序） | ✅ 可用 | `vision/game.py` | 实时窗口第二三行 |
| 决策状态机 | ✅ 仿真满分 | `strategy/fsm.py` | `strategy/run_sim.py` |
| 串口链路（上位机↔下位机） | ✅ 代码就绪，待上板 | `strategy/serial_link.py` | 见 `docs/串口联调说明.md` |
| 实车入口（视觉→决策→串口） | ✅ 三种模式 | `strategy/run_live.py` | `--no-serial` 可先跑 |
| 日志分析 | ✅ 一键报告 | `strategy/report_log.py` | `report_log.py logs/xxx.jsonl` |
| STM32 固件 | ⚠️ 骨架，未编译未上板 | `firmware/` | 见 `firmware/README.md` 五步 |
| 真实棋子样本 | ❌ 等棋子 | `vision/data/pieces/samples/` | `recognize_chars.py --eval-samples` |
| 真实相机标定 | ❌ 等相机装车 | `vision/data/calib/homography_robot.json` | `calib_field.py --preview` |
| 底盘 / 推板 / 整车 | ❌ 等硬件 | `firmware/`、`mech/` | — |
| 对手检测 / 路径绕行 | ❌ 未开始 | — | — |

**一句话**：代码侧（视觉 + 决策 + 串口 + 日志）已经打通，仿真里 60 秒拿满 182 分；
现在卡在"没有真棋子、没有装好的车"。

---

## 1. 立即可做（不需要任何硬件）

- [x] 视觉回归脚本 `vision/test_vision.py`（查全/定位/识别/帧率四项）
- [x] 串口链路 + 实车入口 + 日志分析（`serial_link.py`、`run_live.py`、`report_log.py`）
- [x] 检测提速：Hough 提前退出 + 圆心精修（识别率 78.1% 不变，帧率 24 → 40 FPS）
- [ ] **装 pyserial**：`python -m pip install pyserial`（这台机器上还没装成功）
- [ ] 半自动标注：识别结果直接当文件名，人只改错的（32 枚 10 分钟 → 1~2 分钟）
- [ ] 多帧投票：同一枚棋子连续 3~5 帧投票，消除闪烁、再提准确率
- [ ] 干跑联调：`run_live.py --no-serial` + 录像，把"视觉→决策→指令"整条链跑通
- [ ] 对手车检测：先用经典 CV（场地干净，最大运动块就是对手）
- [ ] 路径绕行：`GOTO` 前插一个中途航点，避开挡路的对手

---

## 2. 硬件到位后（按依赖顺序）

### A. 棋子到了（20 分钟）

- [ ] 量直径 → 改 `vision/config.py` 的 `PIECE_DIA`，填 `docs/硬件参数.md`
- [ ] 拍照 → 切样本 → 打标签 → `python vision/char_templates.py`
- [ ] `python vision/recognize_chars.py --eval-samples vision/data/pieces/samples`
- [ ] 验收：识别率 ≥ 90%（实拍模板通常 95%+）

### B. 相机装车（第 2 周）

- [ ] 刚性相机支架，俯角 50°~65°，车前 150~500mm 入画且能看清字
- [ ] 用**车上那颗相机**标定：`calib_field.py --mode pick --frame robot --board 400 300`
- [ ] 记录高度/俯角/标定日期/反算误差到 `docs/硬件参数.md`
- [ ] 验收：反算误差 < 5mm；`run_live.py --no-serial` 画面对得上

### C. 底盘与固件（第 2~4 周，可与 B 并行）

- [ ] 把 `firmware/` 并进 CubeMX 工程，填 `config.h`（轮径/轮距/线数/减速比）
- [ ] 五步上电调试：轮子方向 → 手推 1m 对里程计 → 转 90° → 串口心跳 → `$GOTO` 到位
- [ ] 验收：走直线 1m 偏差 < 30mm；转 90° 误差 < 3°；心跳稳定

### D. 接起来（第 4~6 周）

- [ ] `python strategy/run_live.py --camera 0 --port COMx`（板子上是 `/dev/ttyUSB0`）
- [ ] 验收：日志里 `link_age` < 0.1s，状态能走完 `APPROACH→PUSH→RETREAT`
- [ ] 单枚闭环实车：对准一枚、推进球门区、退出
- [ ] 一铲多枚：展开推板，一次 ≥5 枚
- [ ] 验收：60 秒单机得分 ≥ 30 分，投放回流 0 次

### E. 效率与对抗（第 9~12 周）

- [ ] 掐表标定 `ROBOT_SPEED_MM_S` / `ALIGN_SEC` / `TRIP_OVERHEAD_S`
- [ ] 对手检测接入 `fsm.update(..., opponent=...)`（300mm 内减速改道，绝不主动撞）
- [ ] 时间管理：最后 6~8 秒不出长差（FSM 已有，按真实规则时长改）
- [ ] 对抗演练 20 局，胜率 ≥ 70%

### F. 赛前 4 周

- [ ] 连续 50 局零故障；故障记进 `docs/测试记录.md` 并修到不复发
- [ ] 第二台整车 + 备件箱
- [ ] 方案书 / 海报 / 3 分钟视频 / 成本清单 / 安全说明

---

## 3. 规则与报名（越早越好）

- [ ] 发邮件问 6 条规则：颜色是否计分 / 能否抢对方球门区棋子 / 单场时长 / 棋子规格与投放方式 / 激光是"必须"还是"允许"、有无激光玩法 / **推对方棋子进己方门是扣分还是不计分？扣多少？**
- [x] 已按"只能推自己颜色、推错扣分"实现规则模型（`PENALTY_FACTOR` 可调，细则出来改一个数即可）
- [ ] 问清报名时间、组队资格（人数、研究生比例、队长要求）、细则下发时间
- [ ] 拿到分值表 → 改 `vision/config.py` 的 `VALUE`
- [ ] 拿到规则 → 改 `SCORE_MODE` / `OWN_SIDE` / `PLAYER_NAME`

---

## 4. 目录规划与分支规范

```
Shark/
├─ vision/      视觉：检测、识别、实时、棋局理解、回归测试（已建）
├─ strategy/    决策：状态机、仿真、串口链路、实车入口、日志分析（已建）
├─ firmware/    STM32：协议、差速、里程计（骨架已建，待上板）
├─ docs/        硬件参数、测试记录、串口联调说明（已建）
├─ mech/        结构图纸、3D 打印件（待建）
└─ logs/        运行时日志（不入库）
```

```bash
git checkout main && git pull                  # main 永远保持能跑
git checkout -b feat/vision-vote               # 每人一条短分支，名字带模块
git add -A && git commit -m "feat(vision): 多帧投票"
git checkout main && git merge feat/vision-vote && git push
```

1. `main` 永远能跑，比赛当天版本从 main 出；
2. 除 main 外任何分支活不过一周，每天收工前合回；
3. **硬件参数只改 `docs/硬件参数.md`**，再同步到 `vision/config.py`；
4. 自动提交（`chore: auto-commit`）赛前一个月关掉；
5. 运行日志 `logs/`、照片 `vision/data/raw/`、结果图 `vision/data/output/` 都不入库。
