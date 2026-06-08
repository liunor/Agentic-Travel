"""清空所有记忆、会话状态和对话历史，恢复到完全干净的初始状态。

用法:
    python scripts/clear_memory.py           # 交互确认后清空
    python scripts/clear_memory.py --force   # 跳过确认，直接清空
    python scripts/clear_memory.py --dry-run # 仅预览将要删除的内容

清空范围:
    - .data/memory/         长期记忆系统（画像、偏好yy、进度、反馈）
    - .data/sessions/       LangGraph 状态快照 + Worker 对话记录
    - .data/chat_history/   JSONL 会话副本

保留:
    - .data/chroma_db/      知识库向量索引（勿删，入库耗时长）
    - .data/docstore/       知识库文档原文
"""
import os
import shutil
import sys
import argparse


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGETS = [
    (".data", "memory", "长期记忆系统（.data/memory/）"),
    (".data", "sessions", "LangGraph 状态 + Worker 对话（.data/sessions/）"),
    (".data", "chat_history", "JSONL 会话副本（.data/chat_history/）"),
]

# 确保不会误删项目根目录
MAX_DEPTH = 3


def count_items(path: str) -> tuple[int, int]:
    """返回 (文件数, 目录数)。"""
    files = dirs = 0
    if not os.path.exists(path):
        return 0, 0
    if os.path.isfile(path):
        return 1, 0
    for root, _dirs, _files in os.walk(path):
        dirs += len(_dirs)
        files += len(_files)
    return files, dirs


def format_size(path: str) -> str:
    """返回路径占用磁盘空间的人类可读格式。"""
    if not os.path.exists(path):
        return "0 B"
    total = 0
    if os.path.isfile(path):
        total = os.path.getsize(path)
    else:
        for root, _dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def dry_run() -> None:
    """预览将要删除的内容。"""
    total_files = 0
    total_dirs = 0
    total_size = 0

    for base, subdir, label in TARGETS:
        full = os.path.join(PROJECT_ROOT, base, subdir)
        f, d = count_items(full)
        if f == 0 and d == 0:
            print(f"  不存在: {label}")
        else:
            size_str = format_size(full)
            print(f"  将删除: {label} ({f} 文件, {d} 目录, {size_str})")
            total_files += f
            total_dirs += d

    print(f"\n总计: {total_files} 文件, {total_dirs} 目录 将被删除")


def clear_all() -> int:
    """执行清空，返回删除的项目数。"""
    cleared = 0
    for base, subdir, label in TARGETS:
        full = os.path.join(PROJECT_ROOT, base, subdir)
        if not os.path.exists(full):
            print(f"  跳过（不存在）: {label}")
            continue

        f, d = count_items(full)
        size_str = format_size(full)

        try:
            if os.path.isfile(full):
                os.remove(full)
            else:
                shutil.rmtree(full)
            print(f"  已清空: {label} ({f} 文件, {d} 目录, {size_str})")
            cleared += 1
        except Exception as e:
            print(f"  失败: {label} — {e}")

    return cleared


def main() -> None:
    parser = argparse.ArgumentParser(description="清空所有记忆数据")
    parser.add_argument("--force", "-f", action="store_true", help="跳过确认直接执行")
    parser.add_argument("--dry-run", "-n", action="store_true", help="仅预览，不实际删除")
    args = parser.parse_args()

    print("=" * 50)
    print("  记忆清空脚本")
    print("=" * 50)
    print()

    if args.dry_run:
        dry_run()
        print("\n未执行任何删除操作（--dry-run）。")
        return

    dry_run()

    if not args.force:
        print()
        try:
            confirm = input("确认清空以上所有记忆数据？此操作不可撤销 [y/N]: ")
        except (EOFError, KeyboardInterrupt):
            print("\n已取消。")
            return
        if confirm.strip().lower() != "y":
            print("已取消。")
            return

    print()
    cleared = clear_all()
    print(f"\n完成: {cleared}/{len(TARGETS)} 项已清空。")
    print("下次启动 Agent 将从头开始，所有记忆为空。")


if __name__ == "__main__":
    main()
