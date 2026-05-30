import httpx
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from configs.settings import settings
from utils.logger import get_logger

logger = get_logger("shiliu.tools.web_search")

TAVILY_API_KEY = settings.TAVILY_API_KEY
TAVILY_BASE_URL = "https://api.tavily.com/search"


class WebSearchInput(BaseModel):
    query: str = Field(..., description="搜索关键词或问题")
    max_results: int = Field(default=5, description="返回结果数量，默认5条")


class SearchResultItem(BaseModel):
    title: str
    url: str
    content: str
    score: Optional[float] = None


class WebSearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    total_results: int


class WebSearchError(BaseModel):
    error: str


@tool("web_search", args_schema=WebSearchInput)
async def web_search(query: str, max_results: int = 5) -> str:
    """【仅兜底使用】搜索引擎，查询突发新闻和通用百科。只能在天气、地图、本地知识库等预设工具均无结果或报错时才能调用，不可作为首选工具。"""
    logger.info("发起联网搜索", query=query)
    if not TAVILY_API_KEY:
        return WebSearchError(error="TAVILY_API_KEY 未配置").model_dump_json()

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.post(TAVILY_BASE_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

            results = [
                SearchResultItem(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    score=r.get("score"),
                )
                for r in data.get("results", [])
            ]

            response = WebSearchResponse(
                query=query,
                results=results,
                total_results=len(results),
            )
            logger.info("搜索完成", query=query, count=len(results))
            return response.model_dump_json()

        except Exception as e:
            logger.exception("搜索请求异常", error=str(e))
            return WebSearchError(error=f"搜索请求异常: {str(e)}").model_dump_json()
