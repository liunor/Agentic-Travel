"""
Worker 生命周期管理工具模块。

该模块实现了 Coordinator 派发和管理后台 Worker 的核心工具集，是
"发号施令→后台执行→结果回传"这一完整闭环的载体。
主要功能包括：
- 提供 `spawn_worker` 工具：根据技能黄页创建全新的 Worker 并异步启动沙箱循环。
- 提供 `send_message` 工具：从磁盘恢复已存在 Worker 的对话历史并追加指令继续执行。
- 提供 `_execute_sandbox_loop` 内部沙箱引擎：负责 Worker 的 ReAct 循环
  （模型思考 → 工具调用 → 结果回写），包含超时控制、遥测统计、插队消息处理。
- 提供 `_background_task_wrapper` 壳函数：包裹沙箱循环并在结束时将结果通知
  投递到全局消息队列，唤醒 Coordinator。

Classes:
    SpawnWorkerInput
        `spawn_worker` 工具的 Pydantic 入参模型。字段：
        - agent_name: 技能包名或基础工具名，必须在黄页中存在。
        - directive: 详尽的自然语言操作指令。
        - model: 可选的模型覆盖，留空走默认解析链。

    SendMessageInput
        `send_message` 工具的 Pydantic 入参模型。字段：
        - to_agent_id: 目标 Worker 的 Task-ID。
        - message: 追加的指令或纠正要求。

Functions:
    _execute_sandbox_loop(worker_id, session_id, messages, allowed_tools_objects, model_name) -> str
        异步沙箱执行循环。最多 6 轮 ReAct 迭代，单次最长 60 秒。
        每轮先清空 pending_messages 处理 Coordinator 插队指令，再调用
        LLM 推理并执行工具。返回 JSON 通知字符串，
        内含遥测数据（总 token 数、工具调用次数、耗时毫秒）。

    _background_task_wrapper(worker_id, session_id, initial_messages, allowed_tools, model_name) -> None
        沙箱外壳协程。捕获 `_execute_sandbox_loop` 的所有异常并兜底，
        最终将结果通知通过 `global_message_queue.enqueue` 投递到主消息队列。

    spawn_worker(agent_name, directive, config, model) -> str
        LangChain `@tool`。根据 agent_name 查询技能黄页获取 SOP Prompt
        与工具白名单，解析模型（env → tool param → yaml defaults → inherit），
        组装初始消息（System + Human），写入磁盘持久化，
        通过 `asyncio.create_task` 启动后台沙箱，瞬间返回 "started" 通知。

    send_message(to_agent_id, message, config) -> str
        LangChain `@tool`。从 JSONL 磁盘恢复 Worker 历史消息与模型名，
        追加新 HumanMessage 并以原模型重新启动后台沙箱，支持跨轮次延续上下文。

Dependencies:
    - `core.skill_loader.skill_loader`: 查询技能名是否合法、获取 SOP Prompt 与工具白名单。
    - `core.tool_registry.physical_tool_manager`: 按白名单提取 Worker 可用的实体工具。
    - `core.task_manager.global_task_manager`: 注册任务 future、查询运行中任务、投递插队消息。
    - `core.message_queue.global_message_queue`: 沙箱结束后向 Coordinator 投递结果通知。
    - `server.agent.llm_factory`: 创建 Worker LLM 实例（支持动态模型解析）。
    - `server.agent.node.session_storage`: 磁盘持久化（元数据 + JSONL 对话记录）。
    - `configs.settings.settings`: 模型名解析（_resolve_model_name）。

Side effects:
    - 每次 spawn/send_message 会在 `.data/sessions/<session_id>/subagents/<worker_id>/`
      下创建 metadata.json 和 transcript.jsonl 文件。
    - 通过 `asyncio.create_task` 创建的后台任务不阻塞调用方，Coordinator 立即返回。
    - 沙箱超时（60s）或轮次耗尽（6 轮）均不会导致进程崩溃，错误通过通知 JSON 回传。
"""
import asyncio
import time
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from core.task_manager import global_task_manager
from core.message_queue import global_message_queue
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from core.skill_loader import skill_loader
from server.agent.llm_factory import get_worker_llm, get_tool_llm
from core.tool_registry import physical_tool_manager
from server.agent.node.session_storage import (
    write_agent_metadata,
    record_sidechain_transcript,
    get_transcript,
    read_agent_metadata
)
from utils.logger import get_logger
from utils.uuid import create_agent_id
from schemas.models import WorkerNotificationSpec, WorkerUsageSpec

logger = get_logger("shiliu.tools.worker")


class SpawnWorkerInput(BaseModel):
    agent_name: str = Field(..., description="必须是黄页中存在的技能包名或基础工具名。")
    directive: str = Field(..., description="详尽的操作指令，需包含所有参数。")
    model: str = Field(default="", description="可选模型覆盖，如 deepseek-v3.2、doubao-1-6-flash。留空则走默认解析链。")


class SendMessageInput(BaseModel):
    to_agent_id: str = Field(..., description="要继续对话的 Worker 的 Task-ID。")
    message: str = Field(..., description="追加的明确指令或纠正要求。")


async def _execute_sandbox_loop(
    worker_id: str,
    session_id: str,
    messages: list,
    allowed_tools_objects: list,
    model_name: str = "",
) -> str:
    """沙箱 ReAct 循环：模型思考 → 工具调用 → 结果回写。

    Args:
        worker_id: Worker 唯一标识符（UUID）。
        session_id: 所属会话 ID，用于磁盘持久化定位。
        messages: 初始消息列表（SystemMessage + HumanMessage），循环中追加 ToolMessage。
        allowed_tools_objects: Worker 被授权使用的工具对象列表。
        model_name: 已解析的 provider 名称。为空时走默认解析链（get_tool_llm）。

    Returns:
        JSON 字符串（WorkerNotificationSpec），包含执行结果和遥测数据。
    """
    worker_logger = logger.bind(worker_id=worker_id, session_id=session_id)

    llm = get_worker_llm(tool_specified_model=model_name) if model_name else get_tool_llm()
    if allowed_tools_objects:
        llm = llm.bind_tools(allowed_tools_objects)
    tools_by_name = {t.name: t for t in allowed_tools_objects}

    start_time = time.time()
    total_tokens_used = 0
    tool_uses_count = 0

    try:
        async def query_loop():
            nonlocal total_tokens_used, tool_uses_count

            for turn in range(6):
                # 每轮思考前清空 Coordinator 插队消息
                task_info = global_task_manager.get_task(worker_id)
                if task_info and not task_info.pending_messages.empty():
                    while not task_info.pending_messages.empty():
                        new_msg = task_info.pending_messages.get_nowait()
                        interruption_msg = HumanMessage(
                            content=f"【Coordinator 追加指令】：\n{new_msg}\n请优先处理此指令！"
                        )
                        messages.append(interruption_msg)
                        record_sidechain_transcript(session_id, worker_id, [interruption_msg])
                        worker_logger.warning("接收到运行中插队指令", new_msg=new_msg)

                response = await llm.ainvoke(messages)
                messages.append(response)

                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    total_tokens_used += response.usage_metadata.get("total_tokens", 0)

                record_sidechain_transcript(session_id, worker_id, [response])

                if not response.tool_calls:
                    return response.content

                for tool_call in response.tool_calls:
                    tool_uses_count += 1
                    t_name, t_args, t_id = tool_call["name"], tool_call["args"], tool_call["id"]
                    worker_logger.info("发起工具调用", tool_name=t_name)

                    if t_name in tools_by_name:
                        try:
                            tool_result_content = await tools_by_name[t_name].ainvoke(t_args)
                        except Exception as e:
                            tool_result_content = f"执行出错: {str(e)}"
                    else:
                        tool_result_content = f"权限拒绝: 不允许使用工具 {t_name}"

                    tool_msg = ToolMessage(content=str(tool_result_content), tool_call_id=t_id)
                    messages.append(tool_msg)
                    record_sidechain_transcript(session_id, worker_id, [tool_msg])

            return "强制停止：调用工具超过最大次数(6次)。"

        final_answer = await asyncio.wait_for(query_loop(), timeout=60.0)
        duration_ms = int((time.time() - start_time) * 1000)
        worker_logger.info("Worker 执行成功", duration_ms=duration_ms, tokens=total_tokens_used)

        notif = WorkerNotificationSpec(
            task_id=worker_id,
            status="completed",
            summary="工作节点（Worker）成功执行了任务",
            result=final_answer,
            usage=WorkerUsageSpec(total_tokens=total_tokens_used, tool_uses=tool_uses_count, duration_ms=duration_ms),
        )
        return notif.model_dump_json(exclude_none=True)

    except asyncio.TimeoutError:
        duration_ms = int((time.time() - start_time) * 1000)
        worker_logger.error("Worker 执行超时被强制杀死")
        notif = WorkerNotificationSpec(
            task_id=worker_id,
            status="killed",
            summary="执行超时（超过 60 秒）",
            usage=WorkerUsageSpec(total_tokens=total_tokens_used, tool_uses=tool_uses_count, duration_ms=duration_ms),
        )
        return notif.model_dump_json(exclude_none=True)

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        worker_logger.exception("Worker 执行内部崩溃")
        notif = WorkerNotificationSpec(
            task_id=worker_id,
            status="failed",
            summary=f"Worker 执行内部崩溃: {str(e)}",
            usage=WorkerUsageSpec(total_tokens=total_tokens_used, tool_uses=tool_uses_count, duration_ms=duration_ms),
        )
        return notif.model_dump_json(exclude_none=True)


async def _background_task_wrapper(
    worker_id: str,
    session_id: str,
    initial_messages: list,
    allowed_tools: list,
    model_name: str = "",
):
    """包裹沙箱循环，结束时将结果投递到全局消息队列。

    Args:
        worker_id: Worker 唯一标识符（UUID）。
        session_id: 所属会话 ID。
        initial_messages: 初始消息列表。
        allowed_tools: Worker 被授权使用的工具对象列表。
        model_name: 已解析的 provider 名称。
    """
    try:
        notification_json = await _execute_sandbox_loop(
            worker_id, session_id, initial_messages, allowed_tools, model_name
        )
    except Exception as e:
        notif = WorkerNotificationSpec(
            task_id=worker_id,
            status="failed",
            summary=f"沙箱外壳崩溃 {str(e)}",
        )
        notification_json = notif.model_dump_json(exclude_none=True)

    await global_message_queue.enqueue(session_id=session_id, message=notification_json)


@tool("spawn_worker", args_schema=SpawnWorkerInput)
async def spawn_worker(agent_name: str, directive: str, config: RunnableConfig, model: str = "") -> str:
    """启动一个新的 Worker 执行任务。使用此工具时，Worker 没有之前的记忆。

    Worker 模型解析优先级（由高到低）：
    1. AGENTIC_RAG_WORKER_MODEL 环境变量（全局覆盖）
    2. 本函数的 model 参数（Coordinator 调用时动态指定）
    3. agent_config.yaml 中 agents.<name>.model
    4. agent_config.yaml 中 defaults.worker
    5. inherit → 继承 Coordinator 当前模型

    Args:
        agent_name: 技能黄页中存在的技能包名或基础工具名。
        directive: 详尽的操作指令，需包含所有参数。
        config: LangChain RunnableConfig，从中提取 thread_id 作为 session_id。
        model: 可选的模型覆盖，如 deepseek-v3.2、doubao-1-6-flash。留空走默认解析链。

    Returns:
        JSON 字符串（WorkerNotificationSpec），包含 task_id 和 "started" 状态。
    """
    from configs.settings import settings

    session_id = config.get("configurable", {}).get("thread_id", "default_session")
    worker_id = create_agent_id(agent_name)

    resolved_model = settings._resolve_model_name(agent_name, model or None)

    sop_prompt = ""
    allowed_tools_objects = []

    if skill_loader.is_skill(agent_name):
        skill = skill_loader.skills[agent_name]
        sop_prompt = skill.prompt_sop
        allowed_tools_objects = physical_tool_manager.get_worker_tools(skill.allowed_tools)
    elif skill_loader.is_base_tool(agent_name):
        sop_prompt = "你是一个精确的工具执行特工。直接使用工具获取数据，不要推理。"
        allowed_tools_objects = physical_tool_manager.get_worker_tools([agent_name])
    else:
        return WorkerNotificationSpec(
            task_id=worker_id, status="failed", summary="未知特工"
        ).model_dump_json(exclude_none=True)

    import datetime
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')

    system_instruction = f"""[后台特工硬性约束]
停止，请先仔细阅读本协议。
你是一个被独立唤醒的后台执行特工（Worker 子进程），你不是与用户对话的主控系统。
你的执行结果只发给主控系统，绝不会被真实用户直接看到。

不可违背的硬性规则：
1. 【禁止闲聊】：绝对不要问好、不要提问、不要建议下一步操作。
2. 【禁止主观发挥】：不要在回复中加入任何自我评价或多余的解释。
3. 【直接执行】：直接且仅使用你被授予的工具来完成指令。如果同时拥有 search_knowledge_base 和其他搜索工具，必须优先使用 search_knowledge_base。
4. 【强制输出格式】：你的回复【必须】严格以"执行范围："开头。
5. 【陈述事实】：只汇报客观事实结果，汇报完毕后立刻停止。
[/后台特工硬性约束]

[当前系统环境]
当前真实时间：{current_time}
[/当前系统环境]

[专家领域与标准操作流程 (SOP)]
{sop_prompt}
[/专家领域与标准操作流程 (SOP)]"""

    initial_messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=f"【任务指令】：\n{directive}")
    ]

    write_agent_metadata(session_id, worker_id, {
        "agentType": agent_name,
        "directive": directive,
        "model": resolved_model,
    })
    record_sidechain_transcript(session_id, worker_id, initial_messages)

    bg_task = asyncio.create_task(
        _background_task_wrapper(worker_id, session_id, initial_messages, allowed_tools_objects, resolved_model)
    )

    global_task_manager.register_task(
        task_id=worker_id,
        task_type=agent_name,
        command=directive,
        future=bg_task,
    )

    return WorkerNotificationSpec(
        task_id=worker_id,
        status="started",
        summary="任务已在后台启动。如需终止，请使用 TaskStop 工具。"
    ).model_dump_json(exclude_none=True)


@tool("send_message", args_schema=SendMessageInput)
async def send_message(to_agent_id: str, message: str, config: RunnableConfig) -> str:
    """继续一个已经存在的 Worker。发送追加指令，它将带着之前的上下文记忆继续工作。

    会从元数据中恢复该 Worker 创建时使用的模型，保证上下文一致性。

    Args:
        to_agent_id: 目标 Worker 的 Task-ID。
        message: 追加的明确指令或纠正要求。
        config: LangChain RunnableConfig，从中提取 thread_id 作为 session_id。

    Returns:
        JSON 字符串（WorkerNotificationSpec），包含 task_id 和状态摘要。
    """
    session_id = config.get("configurable", {}).get("thread_id", "default_session")

    # 情况 A：Worker 正在运行 → 插队投递，不重启
    task = global_task_manager.get_task(to_agent_id)
    if task and task.status == "running":
        global_task_manager.queue_message(to_agent_id, message)
        return WorkerNotificationSpec(
            task_id=to_agent_id,
            status="started",
            summary="消息已投递至运行中的 Worker，将在其下一轮工具调用前处理。"
        ).model_dump_json(exclude_none=True)

    # 情况 B：Worker 已停止 → 从磁盘恢复并重新启动
    try:
        messages = get_transcript(session_id, to_agent_id)
        metadata = read_agent_metadata(session_id, to_agent_id)
    except FileNotFoundError:
        return WorkerNotificationSpec(
            task_id=to_agent_id, status="failed", summary="找不到该 Worker 的历史记忆。"
        ).model_dump_json(exclude_none=True)

    new_human_msg = HumanMessage(content=f"【追加指令】：\n{message}")
    messages.append(new_human_msg)
    record_sidechain_transcript(session_id, to_agent_id, [new_human_msg])

    agent_name = metadata.get("agentType", "unknown")
    resolved_model = metadata.get("model", "")
    allowed_tools_objects = []

    if skill_loader.is_skill(agent_name):
        allowed_tools_objects = physical_tool_manager.get_worker_tools(skill_loader.skills[agent_name].allowed_tools)
    elif skill_loader.is_base_tool(agent_name):
        allowed_tools_objects = physical_tool_manager.get_worker_tools([agent_name])

    bg_task = asyncio.create_task(
        _background_task_wrapper(to_agent_id, session_id, messages, allowed_tools_objects, resolved_model)
    )

    global_task_manager.register_task(
        task_id=to_agent_id,
        task_type=agent_name,
        command=message,
        future=bg_task,
    )

    return WorkerNotificationSpec(
        task_id=to_agent_id,
        status="started",
        summary="任务已成功从断点唤醒并在后台继续执行。如需终止，请使用 TaskStop 工具。"
    ).model_dump_json(exclude_none=True)
