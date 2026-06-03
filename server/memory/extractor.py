"""
峨眉山文旅智能体 记忆系统 - 后台记忆提取模块。

extract_travel_memories() 为 fire-and-forget 异步后台任务，在每轮对话结束时触发。
它负责从对话历史中提取具有长期价值的游客信息，并写入记忆目录（.data/memory/）。
"""
import os
from utils.logger import get_logger
from pydantic import BaseModel, Field
from typing import List, Optional
from server.memory.manager import MemoryManager
from server.agent.llm_factory import get_tool_llm
from server.memory.types import TRAVEL_MEMORY_TYPES
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

logger = get_logger("shiliu.memory.extractor")

# ── 模块级缓存：静态 System Prompt 只构建一次，保证 DeepSeek prefix cache 命中 ──
_CACHED_EXTRACTOR_SYSTEM_MESSAGE = None


def _get_extractor_system_message() -> SystemMessage:
    global _CACHED_EXTRACTOR_SYSTEM_MESSAGE
    if _CACHED_EXTRACTOR_SYSTEM_MESSAGE is not None:
        return _CACHED_EXTRACTOR_SYSTEM_MESSAGE

    system_prompt = """你是一个专门针对四川峨眉山风景区的文旅智能体记忆提取副手。你的任务是分析最新一轮游客与智能体之间的对话，提取具有长期留存价值的画像、偏好、即时进度与生理反馈，将其持久化写入记忆文件夹。

### 峨眉山文旅记忆分类定义:
1. **persona（基础画像 - 解决能不能）**: 结构化、不易变的硬性约束。如：随行人员有老人/小孩、身体恐高、有腿伤膝盖疼、体力极差、预算区间等。
2. **preference（游玩偏好 - 解决想不想）**: 软性主观偏好。如：对佛教/寺庙文化感兴趣或反感、极偏好大自然风光/猴区、想要缆车/坐观光车避开徒步、饮食禁忌（吃素/忌辣）等。
3. **realtime_ctx（即时上下文 - 解决当前进度）**: 行程中的多变状态。如：当前所在位置（如五显岗、雷洞坪、金顶）、实时天气装备状况、行李携带状态、已打卡游览点、今天剩余时间等。
4. **feedback（实时反馈 - 解决承受度及体验）**: 游客在肉体或心理上的即时感受。如："爬坡太累了，腿快断了"、"山上风太凉了，冻感冒了"、"猴子太凶了，害怕"。

### 核心排除标准:
- 绝对不要记录游客和智能体之间礼貌性的日常客套寒暄（如"你好"、"谢谢"、"辛苦了"、"没问题"）。
- 绝对不要记录临时性的工具调试信息或过于琐碎、很快就会失效的话题。
- 如果这轮对话中没有发现任何符合上述四个大类别的长期核心事实，operations 列表请返回空。

### 提取及更新指示：
- 对比下面提供给你的现有记忆清单。如果游客的新陈述属于清单中已有文件的主题（例如：关于体力或者同行人员），请指定该 filename 进行覆盖或更新。
- 如果是新主题，请设定一个拼音或英文小写的 markdown 文件名（如 'traveler_persona.md'）。
- 记忆内容 content 请言简意赅，用第一或第三人称陈述游客的关键事实，不需要包含寒暄或 AI 自己的推理。"""

    _CACHED_EXTRACTOR_SYSTEM_MESSAGE = SystemMessage(content=system_prompt)
    logger.info("记忆提取 System Prompt 已缓存", chars=len(system_prompt))
    return _CACHED_EXTRACTOR_SYSTEM_MESSAGE


class MemoryOperation(BaseModel):
    filename: str = Field(
        description="记忆文件名，必须以 '.md' 结尾（如 'traveler_persona.md'）。如果是已有记忆的更新，必须与已有文件名精确一致。"
    )
    content: str = Field(
        description="记忆的具体主体内容。简明扼要地记录游客的事实、画像、偏好或身体反馈。"
    )
    memory_type: str = Field(
        description="记忆分类，必须是 'persona', 'preference', 'realtime_ctx', 'feedback' 之一。"
    )
    description: str = Field(
        description="一句话简短描述该记忆主题，用于显示在 MEMORY.md 索引中。"
    )


class MemoryExtractionResult(BaseModel):
    operations: List[MemoryOperation] = Field(
        default=[],
        description="从最近对话中提取出的需要写入或更新的记忆操作列表。如果没有新的事实，返回空列表。"
    )


def _extract_text_content(content) -> str:
    """
    安全提取消息的文本内容。
    兼容 str、list（多模态）、None 三种格式。
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # 多模态内容格式：[{"type": "text", "text": "..."}]
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", "").strip())
            elif isinstance(block, str):
                parts.append(block.strip())
        return " ".join(p for p in parts if p)
    return ""

def _build_dialog_text(messages: List[BaseMessage]) -> str:
    """
    从消息列表中提取对话文本，兼容所有消息类型。
    - HumanMessage → 游客
    - AIMessage（有文本内容的）→ 智能体
    - ToolMessage / 其他 → 跳过（工具结果不属于对话文本）
    """
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = _extract_text_content(msg.content)
            if text:
                lines.append(f"游客: {text}")
        elif isinstance(msg, AIMessage):
            text = _extract_text_content(msg.content)
            # 只记录有实际文本内容的 AI 回复（排除纯 tool_calls 的空 content）
            if text:
                lines.append(f"智能体: {text}")
        # ToolMessage / SystemMessage 等直接跳过
    return "\n".join(lines)


async def extract_travel_memories(app, session_id: str, last_processed_len: int = 0) -> None:
    """
    后台异步提取并更新峨眉山游客旅行记忆。
    该方法被设计为 'fire-and-forget' 的后台任务，运行于每轮对话结束时，不阻塞前台用户响应。
    """
    logger.info("开始进行后台旅程记忆提取...", session_id=session_id)

    # 获取当前 LangGraph 线程的最新状态
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = await app.aget_state(config)
    except Exception as e:
        logger.error("获取 LangGraph 线程状态失败，跳过记忆提取", error=str(e))
        return

    messages = state.values.get("messages", [])
    if not messages:
        logger.info("会话消息历史为空，跳过记忆提取。")
        return

    logger.debug("获取到消息历史", total=len(messages), last_processed=last_processed_len)

    # 提取最近一轮对话文本（增量：自上次处理位置之后的消息）
    if last_processed_len < len(messages):
        recent_messages = messages[last_processed_len:]
    else:
        # 兜底：取最近 6 条
        recent_messages = messages[-6:]

    dialog_text = _build_dialog_text(recent_messages)

    if not dialog_text.strip():
        logger.info(
            "最近对话中无有效的游客/智能体文本内容（可能全为工具调用），跳过记忆提取。",
            recent_count=len(recent_messages)
        )
        return

    logger.debug("对话文本提取完成", chars=len(dialog_text))

    # 加载现有记忆清单与缓存的 System Prompt
    manager = MemoryManager()
    index_content = manager.load_memory_index()

    existing_block = ""
    if index_content.strip():
        existing_block = (
            "## 现有已保存记忆清单\n\n"
            "请仔细比对以下清单。如果属于同一主题请覆盖或合并已有文件，切勿新建重复文件：\n\n"
            f"```markdown\n{index_content}\n```\n"
        )

    system_msg = _get_extractor_system_message()

    user_prompt = (
        f"{existing_block}"
        f"## 待分析对话\n\n"
        f"请分析以下最新一轮对话，提取并更新需要保存的记忆：\n\n"
        f"[最新对话]\n{dialog_text}"
    )

    logger.debug("记忆提取 prompt 组装完毕，正在调用大模型进行结构化分析...")

    # 调用 LLM 并使用 with_structured_output 进行强类型提取
    try:
        llm = get_tool_llm()
        structured_llm = llm.with_structured_output(MemoryExtractionResult)

        result = await structured_llm.ainvoke([
            system_msg,
            HumanMessage(content=user_prompt),
        ])
    except Exception as e:
        logger.error("调用大模型提取旅行记忆失败", error=str(e))
        return

    # 执行提取出的记忆操作
    if not result or not result.operations:
        logger.info("经大模型分析，本轮对话未产生需要持久化的核心旅行记忆。")
        return

    logger.info(f"大模型成功提取到 {len(result.operations)} 条记忆变更操作，开始写入磁盘...")
    for op in result.operations:
        try:
            filename = op.filename
            if not filename.endswith(".md"):
                filename += ".md"

            manager.save_memory_topic(
                filename=filename,
                content=op.content,
                memory_type=op.memory_type,
                description=op.description
            )
            logger.info(
                f"成功自动持久化记忆主题文件: {filename}",
                type=op.memory_type,
                description=op.description
            )
        except Exception as ex:
            logger.error(f"写入记忆主题文件 {op.filename} 失败", error=str(ex))

    logger.info("后台记忆自动提取与持久化工作全部完成。")
