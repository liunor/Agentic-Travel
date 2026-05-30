"""
Agent 运行时主循环模块。

该模块负责启动并管理整个 Agentic 系统的运行时生命周期，是连接
LangGraph 状态图与异步消息队列的中枢神经。
主要功能包括：
- 提供 `coordinator_listen_loop` 异步主循环，持续监听全局消息队列中的 Worker 完成通知，
  并在收到通知后自动唤醒 Coordinator 节点继续推理。
- 提供本地调试入口 `main()`，用于在开发环境中构建 Agent 图、注入测试请求并观察完整链路。

Functions:
    coordinator_listen_loop(app)
        异步主循环协程。阻塞等待 `global_message_queue` 中的 Worker 任务通知，
        解析通知中的 session_id 与消息体，构造 HumanMessage 重新送入 LangGraph
        的 `astream` 管道，从而驱动 Coordinator 进行下一轮推理。
        参数 app 为已编译的 LangGraph StateGraph（通常由 `build_agent()` 返回）。

    HumanMessage(content)
        专门用来把用户输入的纯文本（或多模态数据）包装成符合大模型标准接口（Chat Completion API）的结构化对象
        来自 langchain_core 的消息类，用于封装发送给 Coordinator 的文本内容

Side effects:
    - `coordinator_listen_loop` 是一个永不退出的无限循环，设计为伴随整个进程
      生命周期运行，通过 asyncio.CancelledError 或进程终止信号来结束。
    - `coordinator_listen_loop` 中 `if node_name == "coordinator"` 分支内的
      `print(final_reply)` 仅为终端调试输出，生产环境应替换为 WebSocket 推送、
      SSE 事件流或其他实际的客户端交付通道。
    - `main()` 中的 `asyncio.sleep(60)` 仅为调试占位，生产部署时应替换为
      实际的 Web 服务器事件循环（如 uvicorn.run 或 asyncio.Event.wait）。
"""
import asyncio
from utils.logger import get_logger
from server.agent.grapy import build_agent
from langchain_core.messages import HumanMessage
from core.message_queue import global_message_queue

logger = get_logger("shiliu.runner")

async def coordinator_listen_loop(app):
    """ 协程函数，持续监听全局消息队列中的 Worker 完成通知，并唤醒 Coordinator 继续推理。
    Args:
        app:  编译后的 LangGraph StateGraph 实例，包含 Coordinator 节点和相关逻辑。

    Returns:
        None. 该函数设计为一个永不退出的异步循环，持续监听消息队列并触发 Coordinator 推理。

    """
    logger.info("Coordinator 消息队列监听引擎已启动...")

    while True:
        item = await global_message_queue.dequeue()
        session_id = item["session_id"]
        notification_json_str = item["value"]

        logger.info("收到后台特工捷报，准备唤醒 Coordinator", session_id=session_id)

        wrapped_content = (
            f"[WORKER_NOTIFICATION]\n"
            f"```json\n{notification_json_str}\n```\n"
            f"[/WORKER_NOTIFICATION]\n\n"
            f"请结合以上数据，继续推进工作。"
        )
        msg = HumanMessage(content=wrapped_content)

        # 指定 thread_id，让 LangGraph 从 SQLite/Redis 唤醒对应用户的上下文
        config = {"configurable": {"thread_id": session_id}}

        try:
            async for event in app.astream({"messages": [msg]}, config=config):
                for node_name, node_state in event.items():
                    # 这里是 Coordinator 的新一轮发言
                    if node_name == "coordinator":
                        final_reply = node_state["messages"][-1].content
                        print(f"\n[Coordinator 最终答复]:\n{final_reply}\n")

        except Exception as e:
            logger.exception("唤醒 Coordinator 处理通知时发生异常", error=str(e))


# ================== 测试入口 ==================
async def main():
    # 1. 编译构建你的图
    app = await build_agent()

    # 2. 在后台挂起【消息队列监听引擎】
    asyncio.create_task(coordinator_listen_loop(app))

    # 3. 模拟一个用户请求
    test_session = "user_session_001"
    user_input = "帮我查一下乐山市明天的天气，顺便测量一下我现在(成都)到乐山的驾车距离。"

    logger.info(" 用户发起请求", query=user_input)

    config = {"configurable": {"thread_id": test_session}}
    msg = HumanMessage(content=user_input)

    # 触发第一轮对话，Coordinator 会瞬间返回 "我已经派人去查了"
    await app.ainvoke({"messages": [msg]}, config=config)

    # 保持主线程存活，以便观察后台 Worker 的运行和消息队列的回调
    await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())