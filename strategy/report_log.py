# -*- coding: utf-8 -*-
"""把 run_live.py 生成的日志读成一份"人话报告"。

用法：
    python strategy/report_log.py logs/live_20261017_174526.jsonl

它会打印：跑了多久、帧率、状态分布、几趟、脱困几次、耗时瓶颈、串口断过没有、
得分变化，并列出"可疑点"。同目录会生成一份 .summary.txt，可以直接发给我。
"""
import json
import sys
from collections import Counter
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法：python strategy/report_log.py logs/live_xxx.jsonl")
    path = Path(sys.argv[1])
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        raise SystemExit("日志是空的")
    meta = rows[0] if rows[0].get("type") == "meta" else {}
    data = [r for r in rows if r.get("type") != "meta"]
    if not data:
        raise SystemExit("日志里没有帧数据")

    dur = data[-1]["t"] - data[0]["t"]
    fps = len(data) / dur if dur > 0 else 0
    states = Counter(r["state"] for r in data)
    cmds = Counter(r["cmd"].split(",")[0].lstrip("$") for r in data)
    det = [r["ms"].get("detect", 0) for r in data if r.get("ms")]
    tot = [r["ms"].get("total", 0) for r in data if r.get("ms")]
    link = [r["link_age"] for r in data if r.get("link_age") is not None]
    score = [r.get("cum_total") or 0 for r in data]

    trips = sum(1 for i in range(1, len(data))
                if data[i]["state"] == "RETREAT" and data[i - 1]["state"] != "RETREAT")
    rec = sum(1 for i in range(1, len(data))
              if data[i]["state"] == "RECOVER" and data[i - 1]["state"] != "RECOVER")

    lines = []
    lines.append(f"日志：{path.name}    配置：{meta.get('mode')} / 视窗 {meta.get('roi')}mm / "
                 f"棋子直径 {meta.get('piece_dia')}mm / 串口 {meta.get('serial')}")
    lines.append(f"时长 {dur:.1f}s，{len(data)} 帧，平均 {fps:.1f} FPS")
    lines.append(f"趟数 {trips}，脱困 {rec}，最终视野累计得分 {score[-1]}")
    lines.append("状态分布：" + "  ".join(f"{k} {v}" for k, v in states.most_common()))
    lines.append("指令分布：" + "  ".join(f"{k} {v}" for k, v in cmds.most_common()))
    if det:
        lines.append(f"检测耗时：中位 {sorted(det)[len(det) // 2]:.1f}ms，最慢 {max(det):.1f}ms")
    if tot:
        lines.append(f"单帧总耗时：中位 {sorted(tot)[len(tot) // 2]:.1f}ms，最慢 {max(tot):.1f}ms")
    if link:
        lines.append(f"串口心跳：最大间隔 {max(link):.2f}s，超过 0.5s 的次数 "
                     f"{sum(1 for x in link if x > 0.5)}")
    else:
        lines.append("串口：未启用（只跑视觉+决策）")

    # 可疑点
    warn = []
    if link and max(link) > 0.5:
        warn.append("串口断过（>0.5s 没心跳）→ 查线/波特率/下位机是否复位")
    if rec > max(2, trips):
        warn.append(f"脱困 {rec} 次偏多 → 目标可能太远、或卡在墙/推板下")
    if trips == 0 and dur > 20:
        warn.append("一帧都没推（趟数 0）→ 看 brief 字段：是没识别到，还是没规划出目标")
    if det and tot and max(tot) > 3 * (sorted(tot)[len(tot) // 2] + 1e-6):
        warn.append("有帧耗时异常高 → 可能是偶尔的整图重算/系统卡顿")
    if score and score[-1] == 0 and dur > 30:
        warn.append("视野累计得分一直是 0 → 没拍到球门区，或棋子没进区")

    if warn:
        lines.append("可疑点：")
        lines.extend("  · " + w for w in warn)
    else:
        lines.append("可疑点：未发现明显异常")

    text = "\n".join(lines)
    print(text)
    out = path.with_suffix(".summary.txt")
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\n[已保存] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
