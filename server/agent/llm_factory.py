import openai
from typing import Any
from configs.settings import settings
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AIMessage
from langchain_core.language_models import LanguageModelInput


class _DeepSeekThinkingModel(ChatDeepSeek):
    """在官方 ChatDeepSeek 基础上补充 reasoning_content 回传。

    DeepSeek thinking 模式要求：当 AI 回复中包含 reasoning_content 时，
    后续请求必须在对应的 assistant message 中原样回传该字段。

    ChatDeepSeek 已正确：
    - 提取 reasoning_content 到 AIMessage.additional_kwargs（_create_chat_result）
    - 处理 assistant content 必须为 string 的限制
    - 处理 tool message content 序列化

    但它未在序列化请求时回传 reasoning_content，此类补充该逻辑。
    """

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        # 1. 先拿到原始消息列表（用于后续查找 reasoning_content）
        messages = self._convert_input(input_).to_messages()

        # 2. 调用父类完成标准序列化（包括 ChatDeepSeek 的 content 格式修正）
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # 3. 将 reasoning_content 注入回 payload 中对应的 assistant 消息
        payload_msgs = payload.get("messages", [])
        for i, msg in enumerate(messages):
            if (
                isinstance(msg, AIMessage)
                and i < len(payload_msgs)
                and payload_msgs[i].get("role") == "assistant"
            ):
                reasoning = msg.additional_kwargs.get("reasoning_content")
                if reasoning:
                    payload_msgs[i]["reasoning_content"] = reasoning

        return payload


def _build_llm(conf: dict) -> ChatOpenAI:
    """从配置字典构建 LLM 实例。

    conf 中可选字段：
    - model_kwargs: 标准 OpenAI 参数，合并到请求顶层（如 reasoning_effort）
    - extra_body: 提供商自定义参数，嵌套在 extra_body 键下（如 DeepSeek thinking）

    路由逻辑：
    - DeepSeek 官方 API → _DeepSeekThinkingModel（正确处理 reasoning_content 回传）
    - 其他提供商（火山引擎等）→ ChatOpenAI
    """
    model_kwargs = conf.get("model_kwargs") or {}
    extra_body = conf.get("extra_body")
    base_url = conf.get("base_url", "")

    is_deepseek = "deepseek.com" in base_url

    if is_deepseek:
        # DeepSeek 官方 API — 使用 ChatDeepSeek 子类
        # 注意：ChatDeepSeek 使用 api_base 参数名（非 base_url）
        kwargs = {
            "model": conf["model_id"],
            "api_base": base_url,
            "api_key": conf["api_key"],
            **model_kwargs,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        return _DeepSeekThinkingModel(**kwargs)
    else:
        # 其他提供商（火山引擎 Ark 等）— 使用标准 ChatOpenAI
        kwargs = {
            "model": conf["model_id"],
            "base_url": base_url,
            "api_key": conf["api_key"],
            **model_kwargs,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        return ChatOpenAI(**kwargs)


def get_planner_llm() -> ChatOpenAI:
    """Coordinator 专用 LLM。"""
    return _build_llm(settings.planner_llm)


def get_tool_llm() -> ChatOpenAI:
    """Worker 默认 LLM（走完整解析链，包括 AGENTIC_RAG_WORKER_MODEL 环境变量）。"""
    return _build_llm(settings.tool_llm)


def get_worker_llm(agent_name: str | None = None, tool_specified_model: str | None = None) -> ChatOpenAI:
    """Worker 专用 LLM，支持三层优先级模型解析。"""
    conf = settings.resolve_worker_llm(agent_name, tool_specified_model)
    return _build_llm(conf)
