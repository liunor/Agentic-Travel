"""
峨眉山文旅智能体 记忆系统。
该系统包含基础画像、游玩偏好、即时上下文和实时反馈四大旅程记忆，专为峨眉山文旅场景量身定制。

记忆检索流程（3 步）：
  1. scan_memory_headers()  - 轻量扫描：仅读取每个文件前 30 行（frontmatter + 预览）
  2. LLM 选择器（injection.py）- 模型根据用户提问选取相关文件
  3. 加载选中记忆的完整内容，注入上下文

记忆的写入与检索全部由后台 sub-agent 自动完成，Coordinator 无任何记忆操作能力。
"""

from server.memory.types import TRAVEL_MEMORY_TYPES, is_valid_memory_type, get_type_display_name
from server.memory.manager import MemoryManager
from server.memory.extractor import extract_travel_memories
from server.memory.injection import get_memory_context_message
from server.memory.prompt import get_memory_system_message

__all__ = [
    'TRAVEL_MEMORY_TYPES',
    'is_valid_memory_type',
    'get_type_display_name',
    'MemoryManager',
    'extract_travel_memories',
    'get_memory_context_message',
    'get_memory_system_message',
]
