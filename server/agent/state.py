"""
Agent 状态定义模块。

该模块定义了 LangGraph 状态图中流转的所有状态类型与 Reducer 函数，
是图中各节点之间数据传递的类型契约。
主要功能包括：
- 提供 `AgentState`（Coordinator 全局状态），作为 LangGraph 主图的
  唯一状态类型，承载消息历史与会话元信息。
- 提供 `RagInternalState`（RAG 子图私有状态），供 RAG 检索环路内部
  节点共享检索结果、判官裁决与循环控制变量，对外层 Coordinator 透明。
- 提供 `merge_memory` 与 `merge_unique_list` 两个 Reducer 函数，
  分别用于消息列表的追加合并与去重合并。

Functions:
    merge_memory(left, right) -> List[Any]
        Reducer 函数，委托 `langgraph.graph.message.add_messages` 实现
        LangChain 标准消息合并语义（同 ID 覆盖、ToolMessage 关联 AIMessage）。
        用于 `AgentState.messages` 字段的增量更新。

    merge_unique_list(left, right) -> List[str]
        Reducer 函数，合并两个字符串列表并去重，保持原有顺序。
        用于 `RagInternalState.visited_nodes` 字段，避免同一节点被
        重复检索。

Classes:
    AgentState(TypedDict)
        Coordinator 全局状态，LangGraph 主图的唯一 State 类型。字段：
        - messages: 消息历史列表，Reducer 为 `merge_memory`。
        - session_id: 当前会话唯一标识。
        - user_query: 用户原始输入文本。
        - enable_web_search: 是否启用联网搜索兜底。

    RagInternalState(TypedDict)
        RAG 子图内部状态，仅在 RAG 环路节点间共享。字段按功能分组：
        - 任务信息：task_id、query、query_type、enable_web_search。
        - 循环控制：rag_loop_count、has_deep_read。
        - 检索数据池：visited_nodes（去重）、rag_context（RagContextSpec 列表）、
          rag_trajectory（工具调用流水）。
        - 判官裁决：rag_judgement_passed（是否通过）、rag_judgement_reason（打回理由）、
          rag_missing_entity（缺失实体，触发图谱跃迁）。
        - 最终输出：rag_final_result（答案或降级总结）。

Dependencies:
    - `langgraph.graph.message.add_messages`: LangGraph 内置消息 Reducer。
    - `schemas.models.RagContextSpec`: RAG 检索片段的结构化模型。
"""
import operator
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, List, Dict, Any

from schemas.models import RagContextSpec


def merge_memory(left: List[Any], right: List[Any]) -> List[Any]:
    return add_messages(left, right)


def merge_unique_list(left: List[str], right: List[str]) -> List[str]:
    seen = set(left)
    result = list(left)
    for item in right:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

class AgentState(TypedDict):
    """
    极简的 Coordinator 状态机：不再有任何的 tasks 字典！
    所有的任务分发和结果都通过 messages 自然流转！
    """
    messages: Annotated[List[Any], merge_memory]
    session_id: str
    user_query: str
    enable_web_search: bool



class RagInternalState(TypedDict):
    """[RAG 私有状态]
    只有 RAG 子图内部的节点（检索、判官等）能看到和修改这些变量。
    """
    # ===== 继承自外层的任务信息 =====
    task_id: int  # 当前正在处理的任务 ID (关联回全局的 tasks)
    query: str  # Planner 规划出的具体搜索词或描述
    query_type: str  # Planner 判定的检索分辨率 (macro/micro/mixed)
    enable_web_search: bool

    has_deep_read: bool

    # ==== RAG 循环控制参数 ====
    rag_loop_count: int

    # ==== RAG 检索数据池 ====
    visited_nodes: Annotated[List[str], merge_unique_list]  # Context Tracker

    # 用 RagContextSpec 对象替换了普通的 dict
    rag_context: Annotated[List[RagContextSpec], operator.add]

    rag_trajectory: Annotated[List[dict], operator.add]  # 工具调用流水
    rag_judgement_passed: bool  # 判官是否通过 (控制路由)
    rag_judgement_reason: str  # 判官打回理由 (用于反思)
    rag_missing_entity: str  # 用于存放判官提取的缺失实体，触发系统级图谱跃迁

    # ==== LLM 最终裁决 ====
    is_knowledge_sufficient: bool  # LLM 认为知识是否足够（未使用）

    # ==== 子图的最终回传给主图 ====
    rag_final_result: str  # 可以是完美答案，也可以是 force_prompt 的降级总结