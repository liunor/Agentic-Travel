"""RAG 知识库检索工具 — 供 Worker 调用。"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from utils.logger import get_logger
from server.rag.retrieval import get_retriever

logger = get_logger("shiliu.rag.tool")


class SearchKBInput(BaseModel):
    query: str = Field(..., description="明确的搜索查询语句")


@tool("search_knowledge_base", args_schema=SearchKBInput)
def search_knowledge_base(query: str) -> str:
    """搜索本地知识库，返回最相关的文档片段（含来源标注）。

    用于查找已入库的本地资料，如景区介绍、历史文化、攻略指南等。
    不适用于联网搜索、实时天气、地图查询等外部 API 场景。
    """
    logger.info("知识库检索", query=query)

    try:
        retriever = get_retriever()
        nodes = retriever.retrieve(query, top_k=20, rerank_top_n=5)
    except Exception as e:
        logger.error(f"检索失败: {e}")
        return f"知识库检索异常: {e}"

    if not nodes:
        return "未在知识库中找到相关内容，建议尝试不同关键词或使用联网搜索。"

    parts = []
    for i, node in enumerate(nodes, 1):
        source = node.metadata.get("file_name", "未知来源")
        path = node.metadata.get("header_path", source)
        score = node.metadata.get("rerank_score", 0)
        parts.append(
            f"--- 片段 {i} (相关度: {score:.3f}) ---\n"
            f"来源: {source}\n"
            f"路径: {path}\n"
            f"内容:\n{node.text}\n"
        )

    return "\n".join(parts)
