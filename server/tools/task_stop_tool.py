"""
TaskStop 强制终止工具模块。

该模块实现了 Coordinator 对后台 Worker 的物理强杀能力，是 LangGraph
ToolNode 中唯一具备任务终止权限的工具。Coordinator 在 Worker 跑偏、
超时或用户主动取消时，通过调用此工具向 `TaskManager` 发出取消指令。
主要功能包括：
- 提供 `task_stop_tool` LangChain 工具，接收 task_id 并调用底层
  `TaskManager.stop_task` 执行 `future.cancel()` 物理强杀。
- 通过 `TaskStopInputSpec` 做入参校验（task_id 与 shell_id 至少提供一个），
  通过 `TaskStopOutputSpec` 统一所有成功/失败路径的输出格式。

Functions:
    task_stop_tool(task_id: str = None, shell_id: str = None) -> str
        LangChain `@tool("TaskStop")`。执行路径：
        1. 解析 target_id = task_id or shell_id。
        2. 调用 `global_task_manager.get_task(target_id)` 查验任务是否存在。
        3. 若任务不存在 → 返回 success=False, error_code=1。
        4. 若任务非 running 状态 → 返回 success=False, error_code=3。
        5. 调用 `global_task_manager.stop_task(target_id)` 执行 future.cancel()，
           标记 status="killed"，返回 success=True 及任务摘要。
        6. 内部异常兜底 → 返回 success=False, error_code=500。

Constants:
    DESCRIPTION
        ToolNode 向 LLM 展示的工具功能说明（英文），运行时由函数体动态
        注入到 `task_stop_tool.description`。

Dependencies:
    - `schemas.task_stop.TaskStopInputSpec`: Pydantic 入参模型，校验 task_id/shell_id。
    - `schemas.task_stop.TaskStopOutputSpec`: Pydantic 出参模型，统一输出格式。
    - `core.task_manager.global_task_manager`: 提供 get_task 查询与 stop_task 强杀。

Side effects:
    - 成功调用会触发 `future.cancel()`，向对应 Worker 的沙箱协程注入
      CancelledError，导致其终止。该操作不可逆。
"""
import json
from langchain_core.tools import tool
from core.task_manager import global_task_manager
from schemas.task_stop import TaskStopInputSpec, TaskStopOutputSpec

from utils.logger import get_logger

logger = get_logger("shiliu.tools.task_stop")

DESCRIPTION = """
- 通过任务 ID 停止一个正在运行的后台任务。
- 接收一个 task_id 参数，用于识别需要停止的具体任务。
- 返回成功或失败的状态。
- 当你需要终止一个长时间运行的任务时，请使用此工具。
"""


@tool("TaskStop", args_schema=TaskStopInputSpec, description=DESCRIPTION)
async def task_stop_tool(task_id: str = None, shell_id: str = None) -> str:
    target_id = task_id or shell_id
    # 查验底层系统是否有这个任务
    task = global_task_manager.get_task(target_id)

    # 失败路径 1：任务不存在
    if not task:
        logger.warning("尝试停止不存在的任务", task_id=target_id)
        return TaskStopOutputSpec(
            success=False,
            message=f"No task found with ID: {target_id}",
            error_code=1
        ).model_dump_json(exclude_none=True)

    # 失败路径 2：任务没在运行
    if task.status != "running":
        logger.warning("尝试停止非运行中的任务", task_id=target_id, status=task.status)
        return TaskStopOutputSpec(
            success=False,
            message=f"Task {target_id} is not running (status: {task.status})",
            error_code=3
        ).model_dump_json(exclude_none=True)

    # 成功路径
    try:
        result = global_task_manager.stop_task(target_id)

        output = TaskStopOutputSpec(
            success=True,
            message=f"Successfully stopped task: {result['taskId']}",
            task_id=result['taskId'],
            task_type=result['taskType'],
            command=result['command']
        )
        logger.info("任务已被强制终止", task_id=target_id)
        return output.model_dump_json(exclude_none=True)

    # 失败路径 3：内部崩溃
    except Exception as e:
        logger.exception("停止任务时发生异常")
        return TaskStopOutputSpec(
            success=False,
            message=str(e),
            error_code=500
        ).model_dump_json(exclude_none=True)