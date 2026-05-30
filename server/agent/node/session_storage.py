"""
会话持久化存储模块。

该模块实现了 Worker 对话历史的磁盘持久化与完整恢复，是跨轮次
"继续对话（send_message）"功能的数据基础。
主要功能包括：
- 提供基于文件系统的 Worker 隔离存储方案：每个 Worker 拥有独立目录，
  存放 metadata.json（元数据）和 transcript.jsonl（增量对话记录）。
- 提供 `record_sidechain_transcript` 以 JSONL 追加模式写入消息，
  实现事件溯源——每次只 append 新消息，绝不覆写历史。
- 提供 `get_transcript` 从 JSONL 完整重建 Worker 对话上下文，
  并内置 `_filter_unresolved_tool_uses` 清洗逻辑，过滤因 TaskStop
  强杀导致的残缺 tool_calls，防止后续 LLM 调用报 400 BadRequest。

Functions:
    get_worker_dir(session_id, worker_id) -> str
        生成 Worker 专属的磁盘目录路径，若不存在则自动创建。
        路径格式：`DATA_DIR/<session_id>/subagents/<worker_id>/`。

    write_agent_metadata(session_id, worker_id, metadata) -> None
        将 Worker 元数据（agentType、directive 等）写入 metadata.json，
        覆盖式写入。

    read_agent_metadata(session_id, worker_id) -> Dict[str, Any]
        读取 metadata.json 并返回字典。文件不存在时抛出 FileNotFoundError。

    record_sidechain_transcript(session_id, worker_id, messages) -> None
        将 LangChain Message 列表转换为字典后追加写入 transcript.jsonl。
        每行一个 JSON 对象，空消息列表直接跳过。

    _filter_unresolved_tool_uses(messages) -> List[BaseMessage]
        内部清洗函数。扫描消息列表，丢弃 tool_calls 中缺少对应 ToolMessage
        的残缺 AIMessage，避免因强杀导致 API 400 错误。

    get_transcript(session_id, worker_id) -> List[BaseMessage]
        读取 transcript.jsonl 全部行，反序列化为 LangChain Message 列表，
        并自动调用 `_filter_unresolved_tool_uses` 清洗后返回。

Constants:
    DATA_DIR
        数据根目录，默认值为 `<cwd>/.data/sessions`。Worker 的所有持久化
        文件均存放在此目录下。

Dependencies:
    - `langchain_core.messages.BaseMessage`:
        LangChain 所有消息类型的基类，本模块中用作函数签名的类型标注。
    - `langchain_core.messages.AIMessage` / `ToolMessage`:
        分别表示 LLM 回复消息（含 tool_calls）和工具执行结果消息，
        仅在 `_filter_unresolved_tool_uses` 的 isinstance 判断中使用。
    - `langchain_core.messages.messages_to_dict` / `messages_from_dict`:
        将 LangChain Message 对象序列化为 JSON 字典（写入 JSONL）、
        以及从字典反序列化回 Message 对象（从 JSONL 重建对话上下文）。

Side effects:
    - `get_worker_dir` 会自动创建不存在的目录树。
    - `write_agent_metadata` 为覆盖写，不会保留旧文件的任何内容。
    - `record_sidechain_transcript` 为追加写，同一文件会随对话持续增长。
    - 本模块不管理文件清理，长期运行需外部机制回收过期 Worker 目录。
"""
import os
import json
from typing import List, Dict, Any
from utils.logger import get_logger
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages import BaseMessage, messages_to_dict, messages_from_dict

logger = get_logger("shiliu.agent.storage")

DATA_DIR = os.path.join(os.getcwd(), ".data", "sessions")


def get_worker_dir(session_id: str, worker_id: str) -> str:
    """ 获取特工专属的独立文件夹路径

    Args:
        session_id: 会话 ID，标识一个完整的用户交互流程
        worker_id: 特工 ID，标识一个独立的 Worker 实例

    Returns:
        worker_dir: 该 Worker 的专属目录路径，格式为 `DATA_DIR/<session_id>/subagents/<worker_id>/`
    """
    path = os.path.join(DATA_DIR, session_id, "subagents", worker_id)
    os.makedirs(path, exist_ok=True)
    return path


def write_agent_metadata(session_id: str, worker_id: str, metadata: Dict[str, Any]):
    """ 将 Worker 的元数据写入 JSON 文件，覆盖式写入

    Args:
        session_id: 会话 ID，标识一个完整的用户交互流程
        worker_id: 特工 ID，标识一个独立的 Worker 实例
        metadata: Worker 的元数据字典，至少包含 agentType 和 directive 字段，其他字段可选

    Returns:
        None
    """
    path = get_worker_dir(session_id, worker_id)
    file_path = os.path.join(path, "metadata.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def read_agent_metadata(session_id: str, worker_id: str) -> Dict[str, Any]:
    """
    从 JSON 文件读取 Worker 的元数据并返回字典
    Args:
        session_id: 会话 ID，标识一个完整的用户交互流程
        worker_id: 特工 ID，标识一个独立的 Worker 实例

    Returns:
        metadata: Worker 的元数据字典，至少包含 agentType 和 directive 字段，其他字段可选
    """
    file_path = os.path.join(get_worker_dir(session_id, worker_id), "metadata.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到特工 {worker_id} 的元数据文件")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def record_sidechain_transcript(session_id: str, worker_id: str, messages: List[BaseMessage]):
    """ 将 Worker 的对话消息追加写入 JSONL 文件，每行一个独立的 JSON 对象

    Args:
        session_id: 会话 ID，标识一个完整的用户交互流程
        worker_id: 特工 ID，标识一个独立的 Worker 实例
        messages: LangChain 的 BaseMessage 列表，包含 AIMessage、ToolMessage 等类型的消息对象

    Returns:
        None
    """
    if not messages:
        return

    path = get_worker_dir(session_id, worker_id)
    file_path = os.path.join(path, "transcript.jsonl")

    # 将 LangChain 的 Message 对象转化为标准 JSON 字典
    msg_dicts = messages_to_dict(messages)

    with open(file_path, "a", encoding="utf-8") as f:
        for msg_dict in msg_dicts:
            f.write(json.dumps(msg_dict, ensure_ascii=False) + "\n")


def _filter_unresolved_tool_uses(messages: List[BaseMessage]) -> List[BaseMessage]:
    """ 内部清洗函数。扫描消息列表，丢弃 tool_calls 中缺少对应 ToolMessage 的残缺 AIMessage，避免因强杀导致 API 400 错误。

    Args:
        messages: BaseMessage 列表，包含 AIMessage、ToolMessage 等类型的消息对象

    Returns:
        cleaned_messages: 经过清洗后的 BaseMessage 列表，已丢弃残缺的 AIMessage
    """
    if not messages:
        return []

    # 收集所有成功收到结果的 tool_call_id
    resolved_tool_ids = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            resolved_tool_ids.add(msg.tool_call_id)

    cleaned_messages = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # 检查这个 AI 消息里的工具调用是否都有结果
            has_unresolved = any(call['id'] not in resolved_tool_ids for call in msg.tool_calls)
            if has_unresolved:
                logger.warning("发现并清理断头(残缺)的工具调用记录")
                continue  # 丢弃这条残缺消息
        cleaned_messages.append(msg)

    return cleaned_messages


def get_transcript(session_id: str, worker_id: str) -> List[BaseMessage]:
    """
    从 JSONL 文件读取 Worker 的完整对话记录，反序列化为 LangChain Message 列表，并自动调用 `_filter_unresolved_tool_uses` 清洗后返回
    Args:
        session_id: 会话 ID，标识一个完整的用户交互流程
        worker_id: 特工 ID，标识一个独立的 Worker 实例

    Returns:
        messages: 经过清洗后的 BaseMessage 列表，包含 AIMessage、ToolMessage 等类型的消息对象
    """
    file_path = os.path.join(get_worker_dir(session_id, worker_id), "transcript.jsonl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到特工 {worker_id} 的对话记录")

    msg_dicts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                msg_dicts.append(json.loads(line))

    raw_messages = messages_from_dict(msg_dicts)

    return _filter_unresolved_tool_uses(raw_messages)