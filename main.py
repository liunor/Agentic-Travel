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


async def _stream_coordinator(app, messages, session_id: str, memory_context=None) -> bool:
    """将消息送入 LangGraph 并流式打印 Coordinator 输出。

    使用 astream_events(version="v2") 捕获两类事件：
    - on_chat_model_stream: LLM 逐 token 输出 → 实时打印，消除卡顿
    - on_chain_end (coordinator 节点): 捕获最终 AIMessage 的 tool_calls

    memory_context: 上一轮预取好的记忆上下文 SystemMessage，直接注入本轮消息。
    """
    from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk

    config = {"configurable": {"thread_id": session_id}}
    msg = HumanMessage(content=messages) if isinstance(messages, str) else messages
    has_more = False
    printing = False  # 是否正在流式打印中

    # 组装消息：记忆上下文（上一轮预取）→ 用户提问
    input_msgs = []
    if memory_context:
        input_msgs.append(memory_context)
    input_msgs.append(msg)

    async for event in app.astream_events(
        {"messages": input_msgs}, config=config, version="v2"
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


async def _async_input(prompt: str) -> str:
    """在线程池中执行阻塞的 input()，保证 asyncio 事件循环在等待用户输入时仍然活跃。
    这是保证 asyncio.create_task() 后台任务（如记忆提取）能在用户输入等待期间正常运行的关键。
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def main():
    from server.agent.grapy import build_agent
    from server.agent.session_storage import (
        get_active_session_id,
        create_new_session,
        switch_to_new_session,
        save_session_on_exit,
        load_session,
        list_sessions,
        global_session_storage,
    )

    if not check_config():
        sys.exit(1)

    app, _db_conn = await build_agent()

    # 会话持久化：尝试恢复上次活跃会话，若无则创建新会话
    resumed_id = get_active_session_id()
    if resumed_id:
        session_id = resumed_id
        print(f"\n[系统] 已恢复上次会话: {session_id}")
    else:
        session_id = create_new_session()
        print(f"\n[系统] 已创建新会话: {session_id}")

    # 追踪后台记忆提取任务，防止被 GC 丢弃或在 finally 前未完成
    _memory_tasks: set = set()

    # 记忆预取缓存：上一轮预取完成的结果，本轮直接注入，零额外延迟
    _memory_state: dict = {"context": None}

    print()
    print("=" * 60)
    print(" 交互模式 — 直接输入问题 | /sessions 历史会话 | /load <id> 加载会话 | /new 新建 | /clear 清空 | /exit 退出")
    print("=" * 60)

    try:
        while True:
            try:
                query = await _async_input("\n你: ")
                query = query.strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[系统] 退出")
                break

            if not query:
                continue
            if query == "/exit":
                print("[系统] 正在保存会话...")
                save_session_on_exit()
                print("[系统] 退出")
                break
            if query == "/new":
                _cleanup_workers()
                await global_session_storage.flush()
                new_id = switch_to_new_session()
                # 清除内存中的已写入 UUID 记录，否则新会话不会写入
                global_session_storage._written_uuids.clear()
                print(f"[系统] 已创建新会话: {new_id}")
                session_id = new_id
                _memory_state["context"] = None
                continue
            if query == "/clear":
                _cleanup_workers()
                await global_session_storage.flush()
                new_id = switch_to_new_session()
                global_session_storage._written_uuids.clear()
                print("[系统] 上下文已清空")
                session_id = new_id
                _memory_state["context"] = None
                continue
            if query == "/sessions":
                sessions = list_sessions()
                if not sessions:
                    print("[系统] 暂无历史会话")
                else:
                    print("\n  历史会话列表:")
                    print(f"  {'会话ID':<30} {'消息数':<8} {'最后活跃':<20} {'状态'}")
                    print("  " + "-" * 75)
                    from datetime import datetime
                    for s in sessions:
                        ts = datetime.fromtimestamp(s["last_active"]).strftime("%m-%d %H:%M:%S")
                        marker = "← 当前" if s["is_active"] else ""
                        print(f"  {s['session_id']:<30} {s['message_count']:<8} {ts:<20} {marker}")
                    print()
                continue
            if query.startswith("/load"):
                parts = query.split(maxsplit=1)
                if len(parts) < 2:
                    print("[系统] 用法: /load <session_id>")
                    continue
                target_id = parts[1].strip()
                _cleanup_workers()
                await global_session_storage.flush()
                global_session_storage._written_uuids.clear()
                if load_session(target_id):
                    session_id = target_id
                    _memory_state["context"] = None
                    print(f"[系统] 已切换到会话: {session_id}")
                else:
                    print(f"[系统] 会话 {target_id} 的 JSONL 文件不存在，无法加载")
                continue

            # 获取本轮对话前的消息长度，以便后台记忆提取器精确捕获增量对话消息
            config = {"configurable": {"thread_id": session_id}}
            try:
                state = await app.aget_state(config)
                all_messages = state.values.get("messages", [])
                last_processed_len = len(all_messages)
            except Exception:
                last_processed_len = 0
                all_messages = []

            # 上下文窗口管理：裁剪超过 200K 的旧消息（保留 SystemMessage）
            from server.agent.session_storage import trim_context
            if all_messages:
                trimmed = trim_context(all_messages)
                if len(trimmed) < len(all_messages):
                    await app.aupdate_state(config, {"messages": trimmed})
                    all_messages = trimmed
                    last_processed_len = len(trimmed)

            # 启动下一轮记忆预取（与 Coordinator LLM 并行执行）
            from langchain_core.messages import HumanMessage
            from server.memory.injection import get_memory_context_message
            prefetch_msgs = list(all_messages) + [HumanMessage(content=query)]

            async def _prefetch_and_store():
                try:
                    _memory_state["context"] = await get_memory_context_message(prefetch_msgs)
                except Exception:
                    _memory_state["context"] = None

            _prefetch_task = asyncio.create_task(_prefetch_and_store())

            # 用上一轮预取好的记忆上下文注入本轮（第一轮为 None，冷启动）
            has_workers = await _stream_coordinator(app, query, session_id, memory_context=_memory_state["context"])
            _memory_state["context"] = None  # 已消费，清空等下一轮预取结果

            if has_workers:
                await _wait_for_workers(app, session_id, time.time() + 90)

            # 对话轮次结束，触发后台旅行记忆自动提取与持久化
            from server.memory.extractor import extract_travel_memories
            task = asyncio.create_task(extract_travel_memories(app, session_id, last_processed_len))
            _memory_tasks.add(task)
            task.add_done_callback(_memory_tasks.discard)

            # 追加增量会话日志到 JSONL 文件中
            try:
                final_state = await app.aget_state(config)
                from server.agent.session_storage import record_transcript
                await record_transcript(session_id, final_state.values.get("messages", []))
            except Exception as e:
                print(f"[系统] 写入会话日志失败: {e}")


    finally:
        _cleanup_workers()
        # 等待所有仍在运行的记忆提取任务完成，避免进程退出时任务被强制取消
        if _memory_tasks:
            print(f"[系统] 等待 {len(_memory_tasks)} 个记忆提取任务完成...")
            await asyncio.gather(*_memory_tasks, return_exceptions=True)

        print("[系统] 正在将会话日志刷入磁盘...")
        await global_session_storage.flush()
        save_session_on_exit()
        await _db_conn.close()
        print("[系统] 数据库连接已关闭")


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
