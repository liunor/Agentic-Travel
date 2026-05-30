"""日志归档：按天拆分 + 按级别分类。

每行 JSONL 解析 timestamp → 归入对应日期目录 → 按 level 分为 all / error。
输出结构:
    logs/archive/
    ├── 2026-05-28/
    │   ├── all.jsonl
    │   └── error.jsonl
    └── 2026-05-29/
        ├── all.jsonl
        └── error.jsonl

用法:
    python scripts/archive_logs.py              # 归档，保留原文件
    python scripts/archive_logs.py --clear      # 归档后清空原文件
"""
import os
import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "shiliu_agent.log")
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "logs", "archive")


def _parse_date(line: str) -> str | None:
    """从 JSONL 行中提取日期（YYYY-MM-DD）。"""
    try:
        obj = json.loads(line)
        ts = obj.get("timestamp", "")
        return ts[:10]  # ISO 8601: "2026-05-28T19:05:58.123Z" → "2026-05-28"
    except (json.JSONDecodeError, KeyError):
        return None


def archive(clear: bool = False):
    if not os.path.exists(LOG_FILE):
        print(f"[跳过] {LOG_FILE} 不存在")
        return

    # 按 (日期, 级别) 分组
    buckets: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    total = 0
    unknowns = 0

    # 旧日志可能是 GBK（旧版 RotatingFileHandler 没指定 encoding）
    raw = None
    for enc in ("utf-8", "gbk"):
        try:
            with open(LOG_FILE, "r", encoding=enc) as f:
                raw = f.read()
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        print("[错误] 无法读取日志文件（utf-8/gbk 均失败）")
        return

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        date = _parse_date(line)
        if not date:
            unknowns += 1
            continue
        level = "info"
        try:
            obj = json.loads(line)
            level = obj.get("level", "info").lower()
        except json.JSONDecodeError:
            unknowns += 1
            continue
        buckets[date]["all"].append(line)
        if level == "error":
            buckets[date]["error"].append(line)

    if not buckets:
        print("[跳过] 日志为空或无法解析")
        return

    # 写入归档
    written = 0
    for date in sorted(buckets):
        day_dir = os.path.join(ARCHIVE_DIR, date)
        os.makedirs(day_dir, exist_ok=True)
        for category in ("all", "error"):
            lines = buckets[date].get(category, [])
            if not lines:
                continue
            path = os.path.join(day_dir, f"{category}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for l in lines:
                    f.write(l + "\n")
            written += len(lines)

    # 打印统计
    print(f"总行数: {total}  |  无法解析: {unknowns}  |  归档: {written}")
    for date in sorted(buckets):
        all_count = len(buckets[date]["all"])
        err_count = len(buckets[date]["error"])
        print(f"  {date}/  all={all_count}  error={err_count}")

    # 清空原文件
    if clear:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            pass
        print(f"\n[已清空] {LOG_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="日志归档")
    parser.add_argument("--clear", action="store_true", help="归档后清空原日志文件")
    args = parser.parse_args()
    archive(clear=args.clear)
