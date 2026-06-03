"""
峨眉山文旅智能体 记忆系统 - 动态记忆注入模块。

实现 Claude Code 的 3 步记忆检索流程：
  1. 轻量头部扫描：仅读取每个记忆文件前 30 行（frontmatter + 预览），不加载完整正文
  2. LLM 主动选择：将头部清单 + 当前用户提问发给检索模型，由模型决定加载哪几条（最多5个）
  3. 按需完整加载：仅获取选中文件的完整内容，注入为上下文 SystemMessage
"""

import json
import time
from typing import List, Optional, Any

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from server.memory.manager import MemoryManager, STALE_WARN_DAYS
from server.agent.llm_factory import get_tool_llm
from utils.logger import get_logger

logger = get_logger("shiliu.memory.injection")

# ── 模块级缓存：选择器 SystemMessage 只构建一次，保证 DeepSeek prefix cache 命中 ──
_CACHED_SELECTOR_SYSTEM_MESSAGE = None


def _get_selector_system_message() -> SystemMessage:
    global _CACHED_SELECTOR_SYSTEM_MESSAGE
    if _CACHED_SELECTOR_SYSTEM_MESSAGE is not None:
        return _CACHED_SELECTOR_SYSTEM_MESSAGE
    content = (
        "你是峨眉山文旅智能体的记忆检索助手。\n"
        "根据游客当前的提问，从提供的记忆清单中选出最相关的记忆文件（最多5个）。\n"
        "只选择与当前提问真正相关的文件，不相关的一概不选。如果没有任何相关记忆，返回空列表。\n"
        "直接返回文件名列表，不需要解释原因。"

    )
    _CACHED_SELECTOR_SYSTEM_MESSAGE = SystemMessage(content=content)
    logger.info("记忆选择器 System Prompt 已缓存", chars=len(content))
    return _CACHED_SELECTOR_SYSTEM_MESSAGE


class MemorySelectionResult(BaseModel):
    selected_files: List[str] = Field(
        default=[],
        description="需要加载完整内容的记忆文件名列表（最多5个，必须从提供清单中的文件名中选取）"
    )


def _build_header_listing(headers: List[dict]) -> str:
    """
    将头部扫描结果拼成供 LLM 阅读的清单文本。
    格式：- [type] filename.md (X天前) [陈旧]: description
    """
    now = time.time()
    lines = []
    for h in headers:
        age_seconds = now - h["mtime"]
        age_days = int(age_seconds / 86400)
        age_str = "今天" if age_days == 0 else f"{age_days}天前"
        stale_mark = " [陈旧]" if age_days >= STALE_WARN_DAYS else ""
        lines.append(
            f"- [{h['type']}] {h['filename']} ({age_str}){stale_mark}: {h['description']}"
        )
    return "\n".join(lines)


async def get_memory_context_message(messages: List[Any]) -> Optional[SystemMessage]:
    """
    3 步异步记忆检索：轻量头部扫描 → LLM 主动选择 → 按需完整加载。

    Args:
        messages: 历史对话消息列表

    Returns:
        包含检索记忆的 SystemMessage，或 None（无相关记忆 / 出现异常）
    """
    if not messages:
        return None

    # ── Step 0：提取游客最近一次提问 ──
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                user_query = content.strip()
                break

    if not user_query:
        return None

    manager = MemoryManager()

    # ── Step 1：轻量头部扫描（只读前 30 行，不加载正文）──
    headers = manager.scan_memory_headers()
    if not headers:
        logger.debug("记忆目录为空，跳过记忆注入。")
        return None

    listing = _build_header_listing(headers)
    logger.debug("完成头部扫描", memory_count=len(headers))

    # ── Step 2：LLM 主动选择相关记忆文件 ──
    selector_prompt = (
        f"游客当前提问：「{user_query}」\n\n"
        f"当前记忆清单：\n{listing}\n\n"
        f"请从上述清单中选出与该提问最相关的记忆文件（最多5个）。"
    )

    try:
        llm = get_tool_llm()
        structured_llm = llm.with_structured_output(MemorySelectionResult)

        result = await structured_llm.ainvoke([
            _get_selector_system_message(),
            HumanMessage(content=selector_prompt),
        ])
        selected_files = result.selected_files if result else []
    except Exception as e:
        logger.warning("LLM 记忆选择器调用失败，跳过记忆注入", error=str(e))
        return None

    if not selected_files:
        logger.debug("LLM 判断当前提问无相关记忆，跳过注入。")
        return None

    logger.info("LLM 选择了需要加载的记忆文件", selected=selected_files)

    # ── Step 3：按需加载选中文件的完整内容 ──
    # 构建 filename -> header 的快速查找表
    header_map = {h["filename"]: h for h in headers}
    recalled = []
    for filename in selected_files[:5]:
        try:
            full_content = manager.read_memory_topic(filename)
            h = header_map.get(filename, {})
            recalled.append({
                "filename": filename,
                "type": h.get("type", "unknown"),
                "description": h.get("description", ""),
                "content": full_content,
            })
            logger.debug("已加载记忆文件完整内容", filename=filename)
        except Exception as e:
            logger.warning("加载记忆文件完整内容失败，跳过该条", filename=filename, error=str(e))
            continue

    if not recalled:
        return None

    # ── 附加：陈旧度告警 ──
    stale_memories = manager.check_stale_memories()
    stale_warning = ""
    if stale_memories:
        stale_info = ", ".join(
            f"{s['filename']}（{s['age_days']}天前）" for s in stale_memories
        )
        stale_warning = (
            f"\n\n以下记忆已超过 {STALE_WARN_DAYS} 天未更新，数据可能陈旧，"
            f"请结合实际情况判断是否仍然适用：{stale_info}"
        )

    # ── 组装最终注入消息（纯 JSON，无 XML）──
    memory_data = {
        "recalled_memories": recalled,
        "instruction": (
            "以上是根据游客当前问题由模型智能选取的长期旅程记忆。"
            "请在后续规划与回答中，充分结合这些已知事实（画像、体能限制、偏好、当前位置等），"
            "为游客提供最具关怀感、针对性的峨眉山量身定制路线及建议。"
            + stale_warning
        ),
    }

    memory_json = json.dumps(memory_data, ensure_ascii=False, indent=2)
    logger.info("记忆上下文注入完成", recalled_count=len(recalled))
    return SystemMessage(content=f"Traveler Memory Context:\n```json\n{memory_json}\n```")
