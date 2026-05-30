"""
Agent 状态图构建模块。

该模块负责编译 LangGraph 状态图，将 Coordinator 节点与工具执行节点
编排为可持久化的推理流水线，是整个 Agentic 系统的拓扑骨架。
主要功能包括：
- 提供 `build_agent()` 工厂函数，构建并返回已编译的 StateGraph 实例。
- 定义图的结构：Coordinator（决策节点）↔ ToolNode（工具执行节点），
  通过 `tools_condition` 条件边实现有工具调用时循环、无工具调用时终止。
- 挂载 SQLite 持久化检查点（SqliteSaver），使跨 Worker 通知的多次推理
  能共享同一 session 的历史上下文。

Functions:
    build_agent()
        构建、配置并编译 Agentic 状态图。主要步骤：
        - 以 `AgentState` 为状态类型创建 `StateGraph`。
        - 注册 `coordinator` 节点（Coordinator 推理逻辑）。
        - 注册 `tools` 节点（ToolNode，内置 spawn_worker、send_message、task_stop_tool）。
        - 添加边：START → coordinator；coordinator 经 tools_condition 分叉到 tools 或 END；
          tools 执行完毕后无条件回到 coordinator。
        - 创建 SQLite 检查点并挂载到 compile 中，用于跨推理轮次的状态持久化。

Dependencies:
    - `server.agent.state.AgentState`: 图的状态类型定义。
    - `server.agent.node.coordinator.coordinator_node`: Coordinator 节点的异步推理函数。
    - `server.tools.worker_tool`: 提供 spawn_worker 与 send_message 工具。
    - `server.tools.task_stop_tool`: 提供 TaskStop 强制终止工具。

Side effects:
    - 首次调用时会在 `DATA_DIR` 下创建 SQLite 数据库文件（`coordinator_memory.sqlite3`）。
    - 返回的编译图实例可直接用于 `ainvoke` / `astream` 等 LangGraph 运行时 API。
"""
import os
import aiosqlite
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

from server.agent.state import AgentState
from server.tools.task_stop_tool import task_stop_tool
from server.agent.node.session_storage import DATA_DIR
from server.agent.node.coordinator import coordinator_node
from server.tools.worker_tool import spawn_worker, send_message


async def build_agent():
    """
    构建并编译 Agentic 状态图。
    Returns:
        编译后的 StateGraph 实例，包含 Coordinator 和工具节点，已挂载 SQLite 检查点。

    """
    workflow = StateGraph(AgentState)
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("tools", ToolNode([spawn_worker, send_message, task_stop_tool]))

    workflow.add_edge(START, "coordinator")
    workflow.add_conditional_edges(
        "coordinator",
        tools_condition,
        {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "coordinator")

    # 挂载本地 SQLite 作为 Coordinator 的主脑记忆
    os.makedirs(DATA_DIR, exist_ok=True)
    db_path = os.path.join(DATA_DIR, "coordinator_memory.sqlite3")

    conn = await aiosqlite.connect(db_path)
    checkpointer = AsyncSqliteSaver(conn)

    return workflow.compile(checkpointer=checkpointer)