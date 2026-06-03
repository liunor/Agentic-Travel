import os
import re
import time
from typing import Dict, Any, List, Tuple

from server.memory.types import TRAVEL_MEMORY_TYPES, is_valid_memory_type, get_type_display_name

MEMORY_DIR = os.path.join(".data", "memory")
INDEX_FILENAME = "MEMORY.md"

# ── 索引上限（双重检查：行数 + 字节数）──
MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25_000  # 25KB

# ── 头部扫描行数上限（仅读取 frontmatter + 简短预览，不加载完整正文）──
HEADER_SCAN_LINES = 30

# ── 陈旧度告警阈值（超过该天数未更新的记忆会被标记）──
STALE_WARN_DAYS = 7


class MemoryManager:
    def __init__(self, memory_dir: str = MEMORY_DIR):
        self.memory_dir = memory_dir
        self.ensure_memory_dir()

    # ── 目录保障 ──
    def ensure_memory_dir(self) -> None:
        """确保记忆目录存在。"""
        os.makedirs(self.memory_dir, exist_ok=True)

    # ── 索引读取 ──
    def load_memory_index(self) -> str:
        """加载 MEMORY.md 索引文件内容（用于注入 System Prompt）。"""
        index_path = os.path.join(self.memory_dir, INDEX_FILENAME)
        if not os.path.exists(index_path):
            return (
                "# 峨眉山旅程记忆索引 (Travel Memory Index)\n\n"
                "当前旅程记忆索引为空。当您与智能体互动时，智能体会自动记录您的画像、偏好、即时进度与反馈并在此展现。\n"
            )
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"加载记忆索引失败: {str(e)}"

    # ── 单条读写删 ──
    def read_memory_topic(self, filename: str) -> str:
        """读取指定记忆主题文件的完整内容（含 frontmatter 与正文）。"""
        clean_filename = os.path.basename(filename)
        file_path = os.path.join(self.memory_dir, clean_filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到记忆主题文件: {clean_filename}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def save_memory_topic(self, filename: str, content: str, memory_type: str, description: str) -> str:
        """保存或更新记忆主题文件，自动重建索引。"""
        if not is_valid_memory_type(memory_type):
            raise ValueError(
                f"不合法的记忆类型: '{memory_type}'。可选类型: {', '.join(TRAVEL_MEMORY_TYPES)}"
            )

        clean_filename = os.path.basename(filename)
        if not clean_filename.endswith(".md"):
            clean_filename += ".md"
        if clean_filename == INDEX_FILENAME:
            raise ValueError(f"不能直接覆盖索引文件 {INDEX_FILENAME}")

        name = os.path.splitext(clean_filename)[0]
        frontmatter = f"---\nname: {name}\ndescription: {description}\ntype: {memory_type}\n---\n\n"
        full_content = frontmatter + content.strip()

        file_path = os.path.join(self.memory_dir, clean_filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        self._regenerate_index()
        return f"成功保存记忆主题 '{clean_filename}' 并已更新记忆索引。"

    def delete_memory_topic(self, filename: str) -> str:
        """删除指定的记忆主题文件，自动重建索引。"""
        clean_filename = os.path.basename(filename)
        file_path = os.path.join(self.memory_dir, clean_filename)
        if os.path.exists(file_path) and clean_filename != INDEX_FILENAME:
            os.remove(file_path)
            self._regenerate_index()
            return f"成功删除记忆主题 '{clean_filename}'。"
        raise FileNotFoundError(f"未找到记忆主题文件: {clean_filename}")

    # ── 【轻量头部扫描】：仅读取前 HEADER_SCAN_LINES 行，不加载完整正文 ──
    def scan_memory_headers(self) -> List[Dict[str, Any]]:
        """
        扫描全部记忆文件，仅读取前 HEADER_SCAN_LINES 行（frontmatter + 简短预览）。
        不加载完整正文，是 LLM 记忆选择器的数据来源。
        返回字段：filename, name, type, description, preview, mtime
        """
        memories = []
        for file in os.listdir(self.memory_dir):
            if not file.endswith(".md") or file == INDEX_FILENAME:
                continue
            file_path = os.path.join(self.memory_dir, file)
            try:
                head_lines = []
                with open(file_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= HEADER_SCAN_LINES:
                            break
                        head_lines.append(line)
                head_text = "".join(head_lines)
                frontmatter, preview = self._parse_frontmatter(head_text)
                memories.append({
                    "filename": file,
                    "name": frontmatter.get("name", os.path.splitext(file)[0]),
                    "type": frontmatter.get("type", "unknown"),
                    "description": frontmatter.get("description", "无描述"),
                    "preview": preview[:200] if preview else "",  # 仅前 200 字预览
                    "mtime": os.path.getmtime(file_path),
                })
            except Exception:
                continue
        memories.sort(key=lambda x: x["mtime"], reverse=True)
        return memories

    # ── 【陈旧度检测】──
    def check_stale_memories(self, days: int = STALE_WARN_DAYS) -> List[Dict[str, Any]]:
        """
        检测超过指定天数未更新的陈旧记忆。
        返回包含 age_days 字段的记忆头部信息列表。
        """
        now = time.time()
        threshold = days * 86400
        stale = []
        for mem in self.scan_memory_headers():
            age_seconds = now - mem["mtime"]
            if age_seconds > threshold:
                stale.append({**mem, "age_days": int(age_seconds / 86400)})
        return stale

    # ── 内部：frontmatter 解析 ──
    def _parse_frontmatter(self, file_content: str) -> Tuple[Dict[str, str], str]:
        """解析 Markdown 文件的 YAML frontmatter 与正文。"""
        normalized = file_content.replace('\r\n', '\n')
        frontmatter = {}
        body = file_content
        match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', normalized, re.DOTALL)
        if match:
            fm_text, body = match.groups()
            for line in fm_text.split('\n'):
                if ':' in line:
                    key, val = line.split(':', 1)
                    frontmatter[key.strip()] = val.strip()
        return frontmatter, body.strip()

    # ── 内部：索引重建（双重上限截断 + 陈旧度标记）──
    def _regenerate_index(self) -> None:
        """
        扫描所有记忆文件的头部信息，重建 MEMORY.md 索引。
        同时执行双重上限检查（MAX_INDEX_LINES 行 / MAX_INDEX_BYTES 字节），
        超限时截断并附加告警注释。
        """
        headers = self.scan_memory_headers()
        now = time.time()

        lines = [
            "# 峨眉山旅程记忆索引 (Travel Memory Index)\n",
            "这里记录了您在峨眉山旅程中的基础画像、游玩偏好、即时进度与实时反馈。智能体将基于这些记忆动态为您优化游览路线、景点推荐和避坑指南。\n",
            "## 旅程记忆列表 (Travel Memories)\n",
        ]

        if not headers:
            lines.append("当前旅程记忆列表为空。")
        else:
            for mem in headers:
                type_display = get_type_display_name(mem["type"])
                age_days = int((now - mem["mtime"]) / 86400)
                stale_mark = " [陈旧]" if age_days >= STALE_WARN_DAYS else ""
                # 去除描述中已有的类型标签前缀，避免双重标签（如"**[基础画像]** [基础画像] ..."）
                desc = mem["description"]
                for disp in (get_type_display_name(t) for t in TRAVEL_MEMORY_TYPES):
                    if desc.startswith(f"[{disp}]"):
                        desc = desc[len(f"[{disp}]"):].strip()
                        break
                lines.append(
                    f"- [{mem['filename']}]({mem['filename']}) — "
                    f"**[{type_display}]**{stale_mark} {desc}"
                )

        raw = "\n".join(lines) + "\n"

        # ── 双重上限检查：行数 + 字节数 ──
        raw_lines = raw.splitlines()
        raw_bytes = len(raw.encode("utf-8"))
        was_line_truncated = len(raw_lines) > MAX_INDEX_LINES
        was_byte_truncated = raw_bytes > MAX_INDEX_BYTES

        if was_line_truncated or was_byte_truncated:
            truncated_lines = raw_lines[:MAX_INDEX_LINES]
            raw = "\n".join(truncated_lines)
            raw += (
                "\n\n> **WARNING**: MEMORY.md 索引已超过上限（200行 / 25KB），"
                "部分记忆条目未在此处展示。如需查询完整记忆列表，请直接调用 `read_memory_topic` 工具加载具体文件。"
            )

        index_path = os.path.join(self.memory_dir, INDEX_FILENAME)
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(raw)
