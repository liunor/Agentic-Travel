"""docstore 可视化检查工具 — 概览全局，按需钻取。

用法:
    python scripts/inspect_docstore.py               # 概览 → 交互查询
    python scripts/inspect_docstore.py 峨眉           # 直接展示匹配文件的树
    python scripts/inspect_docstore.py 峨眉 --detail  # 展示匹配文件的完整内容
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

DOCSTORE_PATH = Path(__file__).resolve().parent.parent / ".data" / "docstore" / "docstore.json"

# ── 终端颜色 ───────────────────────────────────────────
BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[32m"; BLUE = "\033[34m"
CYAN = "\033[36m"; YELLOW = "\033[33m"
RED = "\033[31m"; RESET = "\033[0m"


def load_docstore():
    if not DOCSTORE_PATH.exists():
        print(f"docstore 不存在: {DOCSTORE_PATH}")
        sys.exit(1)
    with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_nodes(data: dict) -> dict:
    """解析 docstore JSON，返回 {file_name: [node_dict, ...]}。"""
    nodes_data = data.get("docstore/data", {})
    nodes_by_file = defaultdict(list)
    for node_id, node_val in nodes_data.items():
        inner = node_val["__data__"]
        meta = inner["metadata"]
        nodes_by_file[meta["file_name"]].append({
            "id": node_id,
            "text": inner["text"],
            "file_name": meta["file_name"],
            "header_path": meta.get("header_path", meta["file_name"]),
            "doc_id": meta.get("doc_id", ""),
        })
    return dict(nodes_by_file)


def truncate(text: str, max_len: int = 80) -> str:
    t = text.replace("\n", " ")
    return t if len(t) <= max_len else t[:max_len] + "..."


def print_header(title: str):
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")


# ═══════════════════════════════════════════════════════
#  概览（每次必显）
# ═══════════════════════════════════════════════════════
def show_overview(nodes_by_file: dict):
    print_header("概览")

    if not nodes_by_file:
        print("  (empty)")
        return

    all_nodes = []
    for nodes in nodes_by_file.values():
        all_nodes.extend(nodes)

    total_chars = sum(len(n["text"]) for n in all_nodes)
    avg_chars = total_chars // len(all_nodes) if all_nodes else 0

    print(f"  文件数:   {len(nodes_by_file)}")
    print(f"  节点总数: {len(all_nodes)}")
    print(f"  总字符数: {total_chars:,}")
    print(f"  平均字符: {avg_chars}")
    print()

    for fname, nodes in sorted(nodes_by_file.items()):
        chars = sum(len(n["text"]) for n in nodes)
        doc_id = nodes[0]["doc_id"] if nodes else "?"
        print(f"  {GREEN}{fname}{RESET}")
        print(f"    节点: {len(nodes)}  |  字符: {chars:,}  |  MD5: {doc_id[:12]}...")


# ═══════════════════════════════════════════════════════
#  单文件树
# ═══════════════════════════════════════════════════════
def show_tree_for_file(fname: str, nodes: list):
    print(f"\n  {GREEN}{BOLD}{fname}{RESET}")

    nodes_sorted = sorted(nodes, key=lambda n: (n["header_path"].count(">"), n["header_path"]))

    for node in nodes_sorted:
        parts = node["header_path"].split(" > ")
        depth = len(parts) - 1

        indent = "    " + "  " * (depth - 1) if depth > 0 else "    "

        if depth == 0:
            prefix = f"  {BLUE}●{RESET}"
        elif depth == 1:
            prefix = f"  {CYAN}├─{RESET}"
        else:
            prefix = f"  {DIM}├{'─' * depth}{RESET}"

        label = parts[-1] if depth > 0 else parts[0]
        print(f"{indent}{prefix} {YELLOW}{label}{RESET}  {DIM}({len(node['text'])} 字){RESET}")
        print(f"{indent}     {DIM}{truncate(node['text'], 60)}{RESET}")


# ═══════════════════════════════════════════════════════
#  单文件完整文本
# ═══════════════════════════════════════════════════════
def show_detail_for_file(fname: str, nodes: list):
    for node in sorted(nodes, key=lambda n: n["header_path"]):
        print(f"\n  {BOLD}{'─'*58}{RESET}")
        print(f"  {GREEN}路径:{RESET} {node['header_path']}")
        print(f"  {GREEN}字数:{RESET} {len(node['text'])}  |  {GREEN}Node ID:{RESET} {node['id'][:8]}...")
        print(f"  {BOLD}{'─'*58}{RESET}")
        print(node["text"])


# ═══════════════════════════════════════════════════════
#  模糊匹配
# ═══════════════════════════════════════════════════════
def match_files(nodes_by_file: dict, keyword: str) -> dict:
    """用关键词模糊匹配文件名，返回匹配子集。"""
    kw = keyword.lower().strip()
    return {k: v for k, v in nodes_by_file.items() if kw in k.lower()}


# ═══════════════════════════════════════════════════════
#  交互式循环
# ═══════════════════════════════════════════════════════
def interactive_loop(nodes_by_file: dict):
    """交互查询：输入文件名关键词 → 展示树；/detail → 展开全文；/all → 全部；/exit 退出。"""
    print_header("交互查询")
    print(f"  输入文件名关键词 → 查看层级树")
    print(f"  输入 /关键字 --detail → 查看完整内容")
    print(f"  {DIM}/all{RESET}    查看全部文件树    {DIM}/exit{RESET}    退出")

    while True:
        try:
            raw = input(f"\n{BOLD}查询{RESET} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not raw:
            continue
        if raw in ("/exit", "/q"):
            return

        # 解析参数
        parts = raw.split()
        keyword_parts = []
        show_detail = False
        for p in parts:
            if p in ("--detail", "-d"):
                show_detail = True
            else:
                keyword_parts.append(p)
        keyword = " ".join(keyword_parts)

        if keyword in ("/all",):
            for fname, nodes in sorted(nodes_by_file.items()):
                show_tree_for_file(fname, nodes)
            continue

        # 匹配目标文件
        if not keyword_parts:
            print(f"  {RED}请输入文件名关键词，或 /all 查看全部{RESET}")
            continue

        matched = match_files(nodes_by_file, keyword)
        if not matched:
            print(f"  {RED}未找到匹配 '{keyword}' 的文件{RESET}")
            continue

        if show_detail:
            for fname, nodes in sorted(matched.items()):
                show_detail_for_file(fname, nodes)
        else:
            for fname, nodes in sorted(matched.items()):
                show_tree_for_file(fname, nodes)


# ═══════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════
def main():
    data = load_docstore()
    nodes_by_file = build_nodes(data)

    if not nodes_by_file:
        print("docstore 中没有数据，请先运行 python main.py ingest")
        return

    show_overview(nodes_by_file)

    # 命令行快捷模式
    args = [a for a in sys.argv[1:] if not a.startswith("--detail") and a != "-d"]
    want_detail = "--detail" in sys.argv or "-d" in sys.argv

    if args:
        keyword = " ".join(args)
        matched = match_files(nodes_by_file, keyword)
        if not matched:
            print(f"\n未找到匹配 '{keyword}' 的文件，进入交互模式")
            interactive_loop(nodes_by_file)
            return

        if want_detail:
            for fname, nodes in sorted(matched.items()):
                show_detail_for_file(fname, nodes)
        else:
            for fname, nodes in sorted(matched.items()):
                show_tree_for_file(fname, nodes)
    else:
        interactive_loop(nodes_by_file)

    print()


if __name__ == "__main__":
    main()
