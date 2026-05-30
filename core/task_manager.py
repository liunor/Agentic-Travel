"""
后台任务生命周期管理模块。

该模块实现了 Worker 异步任务的注册、追踪、插队通信与强制终止，
是 Coordinator 对后台 Worker 进行运行时管控的操作手柄。
不涉及任何大模型逻辑，纯粹为 Python asyncio 层面的任务调度提供簿记能力。
主要功能包括：
- 提供 `ActiveTaskInfo` 数据类，封装单个任务的元信息（ID、类型、指令、
  asyncio.Task future、状态、挂起消息队列）。
- 提供 `TaskManager` 类，作为全局任务注册中心，支持任务注册、查询、
  物理强杀（future.cancel）以及运行中插队消息投递。
- 通过模块级单例 `global_task_manager` 供 `worker_tool`（生产者）
  和 `task_stop_tool` / `send_message_tool`（消费者）跨模块共享。

Classes:
    ActiveTaskInfo
        单个后台任务的数据容器。字段：
        - task_id: 任务唯一标识符。
        - task_type: 任务类型（如 skill 名称或 base_tool 名称）。
        - command: 任务启动时的指令字符串。
        - future: asyncio.Task 对象，持有沙箱循环协程的引用，可用于 cancel 或 await。
        - status: 任务状态（"running" 或 "killed"）。
        - pending_messages: asyncio.Queue，接收 Coordinator 运行中追加的指令。

    TaskManager
        全局任务注册中心。内部维护 `active_tasks` 字典（task_id → ActiveTaskInfo）。
        主要方法：
        - register_task(task_id, task_type, command, future): 注册一个新任务。
        - get_task(task_id): 按 ID 查询任务信息，不存在返回 None。
        - stop_task(task_id): 取消 future 并将状态标记为 "killed"，返回任务摘要。
        - queue_message(task_id, message): 向运行中任务的 pending_messages 插队投递消息。

Side effects:
    - 模块导入时即实例化 `global_task_manager` 单例，全局唯一。
    - `stop_task` 直接调用 `future.cancel()`，会向对应协程注入 CancelledError，
      被取消的沙箱循环应立即清理资源并退出。
    - `pending_messages` 队列在每次 LLM 推理前被 `_execute_sandbox_loop` 清空，
      不在本模块中消费。
"""
import asyncio
from typing import Dict, Optional
from utils.logger import get_logger

logger = get_logger("shiliu.core.task_manager")


class ActiveTaskInfo:
    def __init__(self, task_id: str, task_type: str, command: str, future: asyncio.Task):
        self.task_id = task_id
        self.task_type = task_type
        self.command = command
        self.future = future
        self.status = "running"
        # 挂起队列，供 send_message 插队使用
        self.pending_messages: asyncio.Queue = asyncio.Queue()


class TaskManager:
    def __init__(self):
        self.active_tasks: Dict[str, ActiveTaskInfo] = {}

    def register_task(self, task_id: str, task_type: str, command: str, future: asyncio.Task):
        self.active_tasks[task_id] = ActiveTaskInfo(task_id, task_type, command, future)
        logger.debug("后台任务已注册", task_id=task_id)

    def get_task(self, task_id: str) -> Optional[ActiveTaskInfo]:
        return self.active_tasks.get(task_id)

    def stop_task(self, task_id: str) -> dict:
        task_info = self.active_tasks.get(task_id)
        if not task_info:
            raise ValueError(f"No task found with ID: {task_id}")

        # 物理强杀
        task_info.future.cancel()
        task_info.status = "killed"
        logger.warning("后台任务已被强制终止", task_id=task_id)
        return {
            "taskId": task_info.task_id,
            "taskType": task_info.task_type,
            "command": task_info.command
        }

    def queue_message(self, task_id: str, message: str):
        task_info = self.active_tasks.get(task_id)
        if task_info and task_info.status == "running":
            task_info.pending_messages.put_nowait(message)
            logger.info("已将追加指令放入任务队列", task_id=task_id)
            return True
        return False

global_task_manager = TaskManager()