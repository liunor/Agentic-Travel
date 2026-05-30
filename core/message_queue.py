"""
异步消息队列模块。

该模块实现了 Worker → Coordinator 的单向异步通知通道，是后台任务与
主控循环之间的解耦桥梁。
主要功能包括：
- 提供 `MessageQueue` 类，封装 `asyncio.Queue` 作为底层管道，
  支持入队（enqueue）和阻塞出队（dequeue）操作。
- 通过模块级单例 `global_message_queue` 供 `worker_tool`（生产者）
  和 `runner`（消费者）跨模块共享，无需显式传递引用。

Classes:
    MessageQueue
        基于 asyncio.Queue 的异步消息队列封装。主要方法：
        - enqueue(session_id, message, mode): 将一条 Worker 通知推入队列。
        - dequeue(): 协程，阻塞等待直到有消息到达，返回消息字典。

Side effects:
    - 模块导入时即实例化 `global_message_queue` 单例，全局共享同一队列实例。
    - 队列无界（未设置 maxsize），生产者速率若持续大于消费者可能导致内存积压；
      生产环境应视负载情况加上限流或 maxsize 限制。
"""
import asyncio
from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger("shiliu.core.queue")

class MessageQueue:
    def __init__(self):
        # asyncio.Queue 原生支持异步挂起和唤醒，比写回调函数更优雅、更防死锁
        self.queue = asyncio.Queue()

    async def enqueue(self, session_id: str, message: str, mode: str = 'task-notification'):
        """
        Args:
            session_id: 标识当前任务所属的会话 ID，便于 Coordinator 定位对应的推理上下文。
            message: Worker 任务完成后的通知内容。
            mode: 消息类型标识，默认为 'task-notification'，可用于区分不同类型的消息（如日志、错误报告等），

        Returns:
            None. 该方法将消息封装成字典并推入队列，供消费者异步获取。
        """
        item = {
            "session_id": session_id,
            "value": message,
            "mode": mode
        }
        await self.queue.put(item)
        logger.debug("任务通知已进入主干队列，等待消费", session_id=session_id, mode=mode)

    async def dequeue(self) -> Dict[str, Any]:
        """ 协程方法，阻塞等待直到有消息到达队列，返回消息字典。

        Returns:
            Dict[str, Any]: 包含 session_id、value 和 mode 的消息字典。
        """
        item = await self.queue.get()
        self.queue.task_done()
        return item

global_message_queue = MessageQueue()