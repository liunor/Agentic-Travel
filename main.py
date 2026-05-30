"""交互式测试入口：启动 Agent 后进入 REPL 对话循环。

用法:
    python main.py              # 进入交互菜单
    python main.py chat         # 直接进入对话模式
    python main.py ingest       # 直接运行知识库入库
"""
import sys
import time
import asyncio


def run_ingestion():
    """知识库入库：扫描 Data/ 下的 .md 文件 → embedding → ChromaDB。"""
    print("=" * 60)
    print(" 知识库入库")
    print("=" * 60)
    from server.rag.ingestion import run_ingestion as _run
    _run()
    print("\n入库流程结束。")


def show_menu():
    """显示模式选择菜单。"""
    print("=" * 60)
    print(" Agentic RAG — 请选择运行模式")
    print("=" * 60)
    print("  [1] 对话模式 (Chat)")
    print("  [2] 知识库入库 (Ingest)")
    print("  [0] 退出")
    print("-" * 60)

    while True:
        try:
            choice = input("请输入: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if choice in ("1", "chat", "Chat"):
            return "chat"
        if choice in ("2", "ingest", "Ingest"):
            return "ingest"
        if choice in ("0", "exit", "Exit"):
            return None
        print("无效选择，请输入 1/2/0")


def check_config():
    """验证 .env 和 agent_config.yaml 是否正确加载。"""
    from configs.settings import settings

    print("=" * 60)
    print(" 配置检查")
    print("=" * 60)

    keys = [
        ("volcengine_API_KEY", settings.volcengine_API_KEY),
        ("DEEPSEEK_API_KEY", settings.DEEPSEEK_API_KEY),
        ("QWEATHER_API_KEY", settings.QWEATHER_API_KEY),
        ("AMAP_API_KEY", settings.AMAP_API_KEY),
        ("TAVILY_API_KEY", settings.TAVILY_API_KEY),
    ]
    for name, val in keys:
        masked = val[:8] + "..." if val else "【未配置】"
        print(f"  {name:25s} = {masked}")

    if settings.AGENTIC_RAG_WORKER_MODEL:
        print(f"  {'AGENTIC_RAG_WORKER_MODEL':25s} = {settings.AGENTIC_RAG_WORKER_MODEL} (全局覆盖)")

    print()
    print("  模型分配:")
    coordinator_cfg = settings.planner_llm
    worker_cfg = settings.tool_llm
    image_cfg = settings.image_llm
    print(f"    Coordinator = {coordinator_cfg['model_id']}  ({coordinator_cfg['base_url']})")
    print(f"    Worker      = {worker_cfg['model_id']}  ({worker_cfg['base_url']})")
    print(f"    Image       = {image_cfg['model_id']}  ({image_cfg['base_url']})")

    missing = [n for n, v in keys if not v and n != "TAVILY_API_KEY"]
    if missing:
        print(f"\n  【错误】缺少必要密钥: {', '.join(missing)}")
        print("  请在项目根目录的 .env 文件中填入对应值。")
        return False

    return True


TOOL_LABELS = {
    "spawn_worker": "启动 Worker",
    "send_message": "继续 Worker",
    "TaskStop": "终止 Worker",
}


async def _stream_coordinator(app, messages, session_id: str) -> bool:
    """将消息送入 LangGraph 并流式打印 Coordinator 输出。

    使用 astream_events(version="v2") 捕获两类事件：
    - on_chat_model_stream: LLM 逐 token 输出 → 实时打印，消除卡顿
    - on_chain_end (coordinator 节点): 捕获最终 AIMessage 的 tool_calls

    Returns:
        True 如果 Coordinator 派出了新 Worker（spawn_worker / send_message）。
    """
    from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk

    config = {"configurable": {"thread_id": session_id}}
    msg = HumanMessage(content=messages) if isinstance(messages, str) else messages
    has_more = False
    printing = False  # 是否正在流式打印中

    async for event in app.astream_events(
        {"messages": [msg]}, config=config, version="v2"
    ):
        kind = event["event"]

        # ── LLM 逐 token 流式输出 ──
        if kind == "on_chat_model_stream":
            # 只捕获 coordinator 节点内的 LLM 输出
            node = event.get("metadata", {}).get("langgraph_node")
            if node != "coordinator":
                continue

            chunk = event["data"].get("chunk")
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                if not printing:
                    print("[Coordinator]: ", end="", flush=True)
                    printing = True
                print(chunk.content, end="", flush=True)

        # ── coordinator 节点执行完毕 ──
        elif kind == "on_chain_end":
            if event.get("name") != "coordinator":
                continue

            output = event["data"].get("output", {})
            out_msgs = output.get("messages", [])
            if not out_msgs:
                continue

            last_msg = out_msgs[-1]
            if not isinstance(last_msg, AIMessage):
                continue

            # 处理 tool_calls
            if last_msg.tool_calls:
                if printing:
                    print()  # 流式文本后换行
                    printing = False
                for tc in last_msg.tool_calls:
                    label = TOOL_LABELS.get(tc["name"], tc["name"])
                    print(f"  → {label}: {tc['args']}")
                    if tc["name"] in ("spawn_worker", "send_message"):
                        has_more = True
            elif printing:
                print()  # 纯文本输出后换行1

                printing = False

    # 保底换行
    if printing:
        print()

    return has_more


async def _wait_for_workers(app, session_id: str, deadline: float):
    """串行等待所有 Worker 完成。每次从消息队列取通知 → 喂给 Coordinator → 循环。

    不再使用独立的 _listen_loop 协程，所有逻辑在主协程中串行执行，消除竞态。
    """
    import json as _json
    from core.message_queue import global_message_queue
    from core.task_manager import global_task_manager

    last_status_count = -1

    while time.time() < deadline:
        # 检查是否还有活着的 Worker
        running = [t for t in global_task_manager.active_tasks.values() if t.status == "running"]
        if not running:
            # 没有 running 任务了，但队列里可能还有未消费的通知
            if global_message_queue.queue.qsize() == 0:
                return

        if len(running) != last_status_count:
            if running:
                print(f"\n[系统] 等待 {len(running)} 个 Worker 结果（最多 {int(deadline - time.time())} 秒）...")
            last_status_count = len(running)

        # 从队列取通知（带超时，以便定期检查 deadline 和 running 状态）
        try:
            item = await asyncio.wait_for(global_message_queue.dequeue(), timeout=3.0)
        except asyncio.TimeoutError:
            continue

        session_id_from_q = item["session_id"]
        notification = item["value"]

        # 解析 worker_id，处理完后标记完成
        worker_id = ""
        try:
            worker_id = _json.loads(notification).get("task_id", "")
        except Exception:
            pass

        wrapped = (
            f"[WORKER_NOTIFICATION]\n"
            f"```json\n{notification}\n```\n"
            f"[/WORKER_NOTIFICATION]\n\n"
            f"请结合以上数据，继续推进工作。"
        )

        print()
        try:
            await _stream_coordinator(app, wrapped, session_id_from_q)
        except Exception as e:
            print(f"\n[系统] 处理 Worker 通知时出错: {e}")

        # 通知处理完毕，标记该 Worker 为已完成
        if worker_id:
            task = global_task_manager.get_task(worker_id)
            if task:
                task.status = "completed"

        last_status_count = -1  # 下一轮重新打印状态

    # ── 超时：kill 剩余 Worker，让 Coordinator 总结现有结果 ──
    running = [
        (tid, t) for tid, t in global_task_manager.active_tasks.items()
        if t.status == "running"
    ]
    if running:
        for tid, task in running:
            task.future.cancel()
            print(f"[系统] 已终止超时 Worker: {tid}")

        killed_list = ", ".join(tid for tid, _ in running)
        summary_prompt = (
            f"[系统通知]\n"
            f"以下 Worker 因超时已被终止: {killed_list}\n"
            f"请基于目前已收到的结果，直接为用户总结。不要继续等待或追问。\n"
            f"[/系统通知]"
        )
        try:
            await _stream_coordinator(app, summary_prompt, session_id)
        except Exception as e:
            print(f"\n[系统] 超时总结出错: {e}")


def _cleanup_workers():
    """取消所有仍在运行的后台 Worker。"""
    from core.task_manager import global_task_manager
    for task_id, info in list(global_task_manager.active_tasks.items()):
        if info.status == "running":
            info.future.cancel()
            print(f"[系统] 已终止后台 Worker: {task_id}")


async def main():
    from server.agent.grapy import build_agent

    if not check_config():
        sys.exit(1)

    app = await build_agent()
    session_id = f"session_{int(time.time())}"

    print()
    print("=" * 60)
    print(" 交互模式 — 直接输入问题 | /clear 清空上下文 | /exit 退出")
    print("=" * 60)

    try:
        while True:
            try:
                query = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[系统] 退出")
                break

            if not query:
                continue
            if query == "/exit":
                print("[系统] 退出")
                break
            if query == "/clear":
                session_id = f"interactive_{int(time.time())}"
                _cleanup_workers()
                print("[系统] 上下文已清空")
                continue

            has_workers = await _stream_coordinator(app, query, session_id)

            if has_workers:
                await _wait_for_workers(app, session_id, time.time() + 90)
    finally:
        _cleanup_workers()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else None

    # 命令行参数优先，否则显示菜单
    if mode in ("ingest", "--ingest", "-i"):
        run_ingestion()
    elif mode in ("chat", "--chat", "-c"):
        asyncio.run(main())
    else:
        choice = show_menu()
        if choice == "ingest":
            run_ingestion()
        elif choice == "chat":
            asyncio.run(main())
