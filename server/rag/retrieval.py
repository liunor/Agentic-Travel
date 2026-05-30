"""
RAG 检索管线：双路召回（Vector + BM25）→ RRF 融合 → DashScope 重排序。
"""

import os
from typing import List, Optional

import httpx
import chromadb
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever

from configs.settings import settings
from utils.logger import get_logger

logger = get_logger("shiliu.rag.retrieval")

class RAGRetriever:
    def __init__(self):
        self._index = None
        self._bm25: Optional[BM25Retriever] = None
        self._loaded = False

    def _ensure_loaded(self):
        """
        确保检索器已加载索引和 BM25 模型，支持懒加载。
        Returns:
                None
        """
        if self._loaded:
            return
        chroma_path = settings.rag_chroma_path
        docstore_dir = settings.rag_docstore_path

        # ---- ChromaDB ----
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_or_create_collection(settings.rag_chroma_collection)
        vector_store = ChromaVectorStore(chroma_collection=collection)

        # ---- docstore ----
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=docstore_dir
        )

        # ---- embedding ----
        embed_cfg = settings.rag_embed_config
        api_key = embed_cfg["api_key"]
        # OpenAIEmbedding / openai 包内部可能降级查 OPENAI_API_KEY 环境变量
        os.environ.setdefault("OPENAI_API_KEY", api_key)
        embed_model = OpenAIEmbedding(
            api_key=api_key,
            api_base=embed_cfg["base_url"],
            model_name=embed_cfg["model_id"],
        )

        LlamaSettings.embed_model = embed_model

        self._index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )

        # ---- BM25 ----
        all_nodes = list(storage_context.docstore.docs.values())
        self._bm25 = BM25Retriever.from_defaults(
            nodes=all_nodes, similarity_top_k=20,
        )

        self._loaded = True
        logger.info(f"检索器就绪，共 {len(all_nodes)} 个节点。")

    def retrieve(self, query: str, top_k: int = 20, rerank_top_n: int = 5) -> List[TextNode]:
        """
        Args:
            query: 用户查询文本
            top_k: 向量检索和 BM25 各自返回的候选数量
            rerank_top_n: DashScope 重排序后返回的最终结果数量

        Returns:
            List[TextNode]: 最终返回给用户的文本节点列表
        """
        self._ensure_loaded()

        if not self._bm25:
            return []

        # ---- 双路 RRF 融合 ----
        vector_retriever = self._index.as_retriever(similarity_top_k=top_k)

        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, self._bm25],
            similarity_top_k=top_k,
            num_queries=1,
            mode="reciprocal_rerank",
        )
        fused = fusion_retriever.retrieve(query)
        if not fused:
            return []

        # ---- DashScope 重排序 ----
        reranked = self._rerank(query, fused, rerank_top_n)
        return reranked


    def _rerank(
        self, query: str, nodes: List[NodeWithScore], top_n: int
    ) -> List[TextNode]:
        """
        Args:
            query: 用户查询文本
            nodes: 融合后待重排序的节点列表，包含原始节点和初步相关度分数
            top_n: 最终返回的节点数量，通常小于等于输入节点数量

        Returns:
            List[TextNode]: 重排序后的节点列表，包含原始文本和新的相关度分数（如有）
        """
        cfg = settings.rag_rerank_config
        documents = [n.node.get_content() for n in nodes]

        payload = {
            "model": cfg.get("model_id", "gte-rerank"),
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": min(top_n, len(documents)), "return_documents": True},
        }
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        }

        try:
            resp = httpx.post(cfg["base_url"], json=payload, headers=headers, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Reranker 调用失败，降级为原始融合结果: {e}")
            return [n.node for n in nodes[:top_n]]

        # 兼容 OpenAI-compatible（results 顶层）和 DashScope 原生（output.results）
        if "output" in data:
            results = data["output"].get("results", [])
        else:
            results = data.get("results", [])
        out = []
        for r in results:
            idx = r.get("index", 0)
            if idx < len(nodes):
                n = nodes[idx].node
                n.metadata["rerank_score"] = r.get("relevance_score", 0)
                out.append(n)
        return out


_retriever: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    """
    Returns:
        RAGRetriever: 单例的 RAGRetriever 实例，支持全局共享和懒加载
    """
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever
