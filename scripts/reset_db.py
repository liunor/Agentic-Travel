"""清空 LangGraph 状态数据库和所有 Worker 对话历史，恢复初始状态。"""
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(PROJECT_ROOT, ".data", "sessions")
SQLITE_FILES = [
    "coordinator_memory.sqlite3",
    "coordinator_memory.sqlite3-shm",
    "coordinator_memory.sqlite3-wal",
]


def reset():
    if not os.path.exists(SESSIONS_DIR):
        print(f"[跳过] {SESSIONS_DIR} 不存在，无需清空。")
        return

    # 1. 删除 SQLite 数据库文件
    for fname in SQLITE_FILES:
        fpath = os.path.join(SESSIONS_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"[删除] {fname}")

    # 2. 删除所有 session 子目录（Worker 档案）
    for item in os.listdir(SESSIONS_DIR):
        item_path = os.path.join(SESSIONS_DIR, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"[删除] {item}/ (含 {count_files(item_path)} 个文件)")

    print("\n数据库已清空。下次启动会自动创建空库。")


def count_files(path: str) -> int:
    n = 0
    for root, dirs, files in os.walk(path):
        n += len(files)
    return n


if __name__ == "__main__":
    confirm = input("确定要清空所有对话历史和 Worker 记录吗？[y/N] ")
    if confirm.lower() != "y":
        print("取消操作。")
        sys.exit(0)
    reset()
