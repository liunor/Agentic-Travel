import yaml
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置，从 .env 读取密钥，从 agent_config.yaml 读取模型/工具配置。

    模型解析优先级（Worker）：
        1. AGENTIC_RAG_WORKER_MODEL 环境变量（全局覆盖）
        2. spawn_worker 调用时传入的 model 参数
        3. agent_config.yaml 中 agents.<name>.model
        4. agent_config.yaml 中 defaults.worker
        5. inherit → 使用 Coordinator 模型

    Coordinator 模型：
        agent_config.yaml 中 defaults.coordinator，无其他覆盖层级。
    """

    BASE_DIR: Path = BASE_DIR

    # -- 密钥 — 全部来自 .env --
    volcengine_API_KEY: str           # 火山引擎 Ark（doubao、deepseek-v3.2、seedream）
    DEEPSEEK_API_KEY: str             # DeepSeek 官方 API（deepseek-v4-pro / v4-flash）
    QWEATHER_API_KEY: str             # 和风天气
    AMAP_API_KEY: str                 # 高德地图
    TAVILY_API_KEY: str = ""          # Tavily 搜索（可选）
    aliyun_API_KEY: str               # 阿里云（embedding + reranker）
    AGENTIC_RAG_WORKER_MODEL: str = ""  # 全局 Worker 模型覆盖（可选）

    _config: dict = {}

    def __init__(self, **values):
        super().__init__(**values)
        yaml_path = BASE_DIR / "configs" / "agent_config.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    # ============================================================
    # 内部工具方法
    # ============================================================
    def _get_api_key(self, provider_cfg: dict) -> str:
        """根据 provider 的 api_key_env 字段读取对应的 env var。"""
        env_name = provider_cfg.get("api_key_env", "volcengine_API_KEY")
        return getattr(self, env_name, "")

    def _get_llm_config(self, provider_name: str) -> dict:
        """根据 provider 名返回 {model_id, base_url, api_key, extra_params} 字典。"""
        providers = self._config.get("providers", {})
        if provider_name not in providers:
            raise KeyError(
                f"Provider '{provider_name}' 未在 agent_config.yaml 的 providers 中定义。"
                f" 可用: {list(providers.keys())}"
            )
        conf = providers[provider_name].copy()
        conf["api_key"] = self._get_api_key(conf)
        return conf

    def _resolve_model_name(
        self,
        agent_name: str | None = None,
        tool_specified_model: str | None = None,
    ) -> str:
        """解析 Worker 的 provider 名称（按优先级 1→5）。"""
        # 1. 全局环境变量
        if self.AGENTIC_RAG_WORKER_MODEL:
            return self.AGENTIC_RAG_WORKER_MODEL

        # 2. Tool 调用时指定的 model
        if tool_specified_model:
            return tool_specified_model

        # 3. Agent 定义中的 model
        agents = self._config.get("agents", {})
        if agent_name and agent_name in agents:
            agent_model = agents[agent_name].get("model")
            if agent_model and agent_model != "inherit":
                return agent_model

        # 4. defaults.worker
        defaults = self._config.get("defaults", {})
        default_worker = defaults.get("worker", "inherit")
        if default_worker != "inherit":
            return default_worker

        # 5. inherit → Coordinator 模型
        return defaults.get("coordinator", "deepseek-v4-pro")

    # ============================================================
    # LLM 配置属性
    # ============================================================
    @property
    def planner_llm(self) -> dict:
        """Coordinator 的 LLM 配置。"""
        model_name = self._config.get("defaults", {}).get("coordinator", "deepseek-v4-pro")
        return self._get_llm_config(model_name)

    @property
    def tool_llm(self) -> dict:
        """Worker 的默认 LLM 配置（走完整解析链）。"""
        model_name = self._resolve_model_name()
        return self._get_llm_config(model_name)

    @property
    def image_llm(self) -> dict:
        """图像生成模型配置。"""
        model_name = self._config.get("defaults", {}).get("image", "seedream-image")
        return self._get_llm_config(model_name)

    def resolve_worker_llm(
        self,
        agent_name: str | None = None,
        tool_specified_model: str | None = None,
    ) -> dict:
        """解析 Worker 的 LLM 配置（公开入口）。"""
        model_name = self._resolve_model_name(agent_name, tool_specified_model)
        return self._get_llm_config(model_name)

    # ============================================================
    # 第三方工具 base_url
    # ============================================================
    @property
    def amap_base_url(self) -> str:
        return (
            self._config.get("tools", {})
            .get("amap", {})
            .get("base_url", "https://restapi.amap.com/v3")
        )

    @property
    def qweather_base_url(self) -> str:
        return (
            self._config.get("tools", {})
            .get("qweather", {})
            .get("base_url", "https://m42vhfja65.re.qweatherapi.com")
        )

    # ============================================================
    # RAG 知识库配置
    # ============================================================
    @property
    def rag_config(self) -> dict:
        return self._config.get("rag", {})

    @property
    def rag_data_path(self) -> Path:
        return BASE_DIR / self.rag_config.get("data_path", "Data")

    @property
    def rag_chunk_max_size(self) -> int:
        return self.rag_config.get("chunk_max_size", 2000)

    @property
    def rag_embed_config(self) -> dict:
        cfg = self.rag_config.get("embedding", {}).copy()
        cfg["api_key"] = getattr(self, cfg.get("api_key_env", "aliyun_API_KEY"), "")
        return cfg

    @property
    def rag_rerank_config(self) -> dict:
        cfg = self.rag_config.get("reranker", {}).copy()
        cfg["api_key"] = getattr(self, cfg.get("api_key_env", "aliyun_API_KEY"), "")
        return cfg

    @property
    def rag_chroma_path(self) -> str:
        return str(BASE_DIR / self.rag_config.get("chroma", {}).get("path", ".data/chroma_db"))

    @property
    def rag_chroma_collection(self) -> str:
        return self.rag_config.get("chroma", {}).get("collection_name", "kb_collection")

    @property
    def rag_docstore_path(self) -> str:
        return str(BASE_DIR / self.rag_config.get("docstore", {}).get("path", ".data/docstore"))

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")


settings = Settings()
