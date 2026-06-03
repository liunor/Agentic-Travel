"""
工具注册与权限过滤模块。

该模块实现了一个集中式的工具兵器库，负责管理所有 LangChain `@tool` 对象的
生命周期、权限过滤与稳定排序，是 Coordinator-Worker 架构中工具分发的安全边界。
主要功能包括：
- 提供 `PhysicalToolManager` 类，统一管理本地内置工具与远端 MCP 工具的双通道注册。
- 提供 `get_worker_tools` 沙盒 API：按白名单精确提取 Worker 可用工具，实现最小权限原则。
- 提供 `get_coordinator_tools` 过滤 API：基于 `COORDINATOR_ALLOWED_TOOLS` 白名单自动
  筛选 Coordinator 可触碰的编排工具，防止 Coordinator 越权调用业务工具。
- 所有工具输出均按名称字母排序，确保相同工具集每次产生的 Prompt Token 序列完全一致，
  最大化 LLM Prompt Cache 命中率。

Classes:
    PhysicalToolManager
        实体兵器库管理器。内部维护两个独立字典（`_builtin_tools` 与 `_mcp_tools`），
        初始化时自动从 `ALL_AVAILABLE_TOOLS` 加载本地工具。主要方法：
        - _load_builtin_tools(): 遍历 ALL_AVAILABLE_TOOLS 将本地工具填充到 _builtin_tools。
        - register_mcp_tool(tool): 动态注册远端 MCP 工具到 _mcp_tools 字典。
        - _get_sorted_stable_tools(tools_dict): 按名称字母排序，保证 Prompt Cache 稳定性。
        - _get_all_merged_tools(): 合并双通道工具，内置优先、MCP 在后，各自排序。
        - get_worker_tools(allowed_names): 沙盒 API，按白名单精确提取并排序。
        - get_coordinator_tools(orchestration_tools): 过滤 API，合并编排工具与白名单工具。
        - has_tool(name): 检查指定名称的工具是否已注册。

Constants:
    COORDINATOR_ALLOWED_TOOLS
        Coordinator 白名单集合，包含 `spawn_worker`、`send_message` 及预留的
        `subscribe_pr_activity`。不在白名单中的工具 Coordinator 无法直接调用。

Side effects:
    - 模块导入时即实例化 `physical_tool_manager` 单例，自动完成内置工具的加载。
    - 初始化阶段若 `ALL_AVAILABLE_TOOLS` 中任何工具加载失败会通过 logger 记录，
      不会中断进程。
"""
from typing import Dict, List
from utils.logger import get_logger
from langchain_core.tools import BaseTool
from server.tools.tool_manager import ALL_AVAILABLE_TOOLS

logger = get_logger("shiliu.core.tools")

COORDINATOR_ALLOWED_TOOLS = {
    "spawn_worker",
    "send_message",
    "subscribe_pr_activity"  # 预留给未来的 MCP 订阅工具
}

class PhysicalToolManager:
    def __init__(self):
        self._builtin_tools: Dict[str, BaseTool] = {}
        self._mcp_tools: Dict[str, BaseTool] = {}

        self._load_builtin_tools()

    def _load_builtin_tools(self):
        """ 加载本地内置工具。

        Returns:
            None. 从 ALL_AVAILABLE_TOOLS 中加载工具到 _builtin_tools 字典。
        """
        for t in ALL_AVAILABLE_TOOLS:
            self._builtin_tools[t.name] = t
            logger.debug("本地内置实体工具已挂载", tool_name=t.name)

    def register_mcp_tool(self, tool: BaseTool):
        """ 注册远端 MCP 工具。

        Args:
            tool: BaseTool 实例，来自 MCP 的工具定义。

        Returns:
            None. 将工具注册到 _mcp_tools 字典，并记录日志。
        """
        self._mcp_tools[tool.name] = tool
        logger.info("远端 MCP 工具已挂载", tool_name=tool.name)

    def _get_sorted_stable_tools(self, tools_dict: Dict[str, BaseTool]) -> List[BaseTool]:
        """ 按名称字母排序工具列表，保证 Prompt Cache 稳定性。

        Args:
            tools_dict: 包含工具名称到 BaseTool 实例映射的字典。

        Returns:
            List[BaseTool]. 按 tool.name 字母表顺序排序的工具列表。
        """
        return sorted(list(tools_dict.values()), key=lambda t: t.name)

    def _get_all_merged_tools(self) -> List[BaseTool]:
        """ 获取合并后的完整工具列表，内置工具优先、MCP 工具在后，各自按名称排序。

        Returns:
            List[BaseTool]. 合并后的工具列表，内置工具在前，MCP 工具在后，均按名称排序。
        """
        sorted_builtins = self._get_sorted_stable_tools(self._builtin_tools)
        sorted_mcps = self._get_sorted_stable_tools(self._mcp_tools)
        return sorted_builtins + sorted_mcps


    def get_worker_tools(self, allowed_names: List[str]) -> List[BaseTool]:
        """ Worker 获取工具池。
            1. 接收 Coordinator发来的工具白名单 (allowed_names)。
            2. 从内置工具和 MCP 工具中筛选出名称在白名单中的工具。
            3. 按名称排序并返回，保证同一种 Worker 每次跑出来的 Prompt Token 完全一致。

        Args:
            allowed_names: Worker 允许使用的工具名称列表，通常由 Coordinator 在 tool_calls 中指定。

        Returns:
            List[BaseTool]. 过滤后的工具列表，仅包含名称在 allowed_names 中的工具，按名称排序。
        """
        all_tools = {**self._builtin_tools, **self._mcp_tools}
        executable_tools = []

        for name in allowed_names:
            if name in all_tools:
                executable_tools.append(all_tools[name])
            else:
                logger.error("Worker 试图挂载不存在的底层工具", missing_tool=name)

        return sorted(executable_tools, key=lambda t: t.name)


    def get_coordinator_tools(self, orchestration_tools: List[BaseTool]) -> List[BaseTool]:
        """ Coordinator 获取工具池。
            1. 接收从外部传进来的编排工具 (如 spawn_worker)。
            2. 从所有已注册工具中，扫描是否还有别的允许 Coordinator 使用的工具 (如 MCP 提供的订阅工具)。
            3. 合并、排序并返回。
        Args:
            orchestration_tools: Coordinator 直接需要的编排工具列表，通常包含 spawn_worker、send_message 等。

        Returns:
            List[BaseTool]. 过滤后的工具列表，包含编排工具和白名单工具，按名称排序。

        """
        coordinator_tools = list(orchestration_tools)
        all_tools = self._get_all_merged_tools()

        for t in all_tools:
            # 只有名称在白名单里，或者是特定的 MCP 订阅工具，才允许 Coordinator 触碰
            if t.name in COORDINATOR_ALLOWED_TOOLS or t.name.endswith("subscribe_pr_activity"):
                # 防重处理
                if t.name not in [ct.name for ct in coordinator_tools]:
                    coordinator_tools.append(t)

        return sorted(coordinator_tools, key=lambda t: t.name)

    def has_tool(self, name: str) -> bool:
        return name in self._builtin_tools or name in self._mcp_tools

physical_tool_manager = PhysicalToolManager()