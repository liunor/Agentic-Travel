"""
RAG 入库管线：MD 层级切分 → 去重 → embedding → ChromaDB 持久化。

独立运行: python -m server.rag.ingestion
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict

import chromadb
from llama_index.core import Document
from llama_index.core import StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from configs.settings import settings
from utils.logger import get_logger

logger = get_logger("shiliu.rag.ingestion")

INGESTED_LOG = settings.BASE_DIR / ".data" / "ingested_files.json"


def _compute_md5(file_path: str) -> str:
    """
    计算文件的 MD5 哈希值，用于去重判断。

    Args:
        file_path: 文件路径

    Returns:
        str: 文件内容的 MD5 哈希值，用于去重判断
    """
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_ingested() -> Dict[str, str]:
    """
    加载已入库文件的 MD5 记录，返回 {file_path: md5} 的字典。

    Returns:
        Dict[str, str]: 已入库文件的 MD5 记录，格式为 {file_path: md5}
    """
    if INGESTED_LOG.exists():
        try:
            return json.loads(INGESTED_LOG.read_text("utf-8"))
        except Exception:
            return {}
    return {}


def _save_ingested(records: Dict[str, str]):
    """
    保存已入库文件的 MD5 记录到磁盘，格式为 {file_path: md5}。
    Args:
        records: 已入库文件的 MD5 记录，格式为 {file_path: md5}

    Returns:
        None
    """
    INGESTED_LOG.parent.mkdir(parents=True, exist_ok=True)
    INGESTED_LOG.write_text(json.dumps(records, ensure_ascii=False, indent=2), "utf-8")


def _parse_md_to_nodes(file_path: str) -> List[TextNode]:
    """
    按 Markdown 标题层级切分，为每个节点注入 header_path 元数据。

    Args:
        file_path: Markdown 文件路径

    Returns:
        List[TextNode]: 切分后的文本节点列表，每个节点包含 text 和 metadata（file_name, header_path）

    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    file_name = os.path.basename(file_path)
    nodes: List[TextNode] = []
    header_stack: Dict[int, str] = {}
    buf: List[str] = []

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.+)", line)
        if m:
            chunk_text = "".join(buf).strip()
            if chunk_text:
                path = " > ".join(
                    [file_name] + [h for _, h in sorted(header_stack.items())]
                )
                nodes.append(TextNode(
                    text=chunk_text,
                    metadata={"file_name": file_name, "header_path": path},
                ))

            level = len(m.group(1))
            header_stack[level] = m.group(2).strip()
            for lv in list(header_stack):
                if lv > level:
                    del header_stack[lv]
            buf = [line]
        else:
            buf.append(line)

    chunk_text = "".join(buf).strip()
    if chunk_text:
        path = " > ".join(
            [file_name] + [h for _, h in sorted(header_stack.items())]
        )
        nodes.append(TextNode(
            text=chunk_text,
            metadata={"file_name": file_name, "header_path": path},
        ))

    return nodes


def run_ingestion():
    """
    RAG 入库管线：MD 层级切分 → 去重 → embedding → ChromaDB 持久化。

    Returns:
        None
    """
    data_dir = settings.rag_data_path
    if not data_dir.exists():
        logger.warning(f"数据目录不存在: {data_dir}")
        return

    md_files = list(data_dir.rglob("*.md"))
    if not md_files:
        logger.info("Data/ 下没有 .md 文件，跳过入库。")
        return

    ingested = _load_ingested()

    # ---- MD5 去重 ----
    new_files = []
    for fp in md_files:
        fp_str = str(fp)
        md5 = _compute_md5(fp_str)
        if ingested.get(fp_str) == md5:
            continue
        new_files.append((fp_str, md5))

    if not new_files:
        logger.info("所有 .md 文件均已入库，无需处理。")
        return

    logger.info(f"发现 {len(new_files)} 个新文件，开始入库...")

    # ---- Embedding 模型 ----
    embed_cfg = settings.rag_embed_config
    api_key = embed_cfg["api_key"]
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    embed_model = OpenAIEmbedding(
        api_key=api_key,
        api_base=embed_cfg["base_url"],
        model_name=embed_cfg["model_id"],
        embed_batch_size=embed_cfg.get("batch_size", 10),
    )

    # ---- ChromaDB + docstore ----
    chroma_path = settings.rag_chroma_path
    docstore_dir = settings.rag_docstore_path
    os.makedirs(chroma_path, exist_ok=True)
    os.makedirs(docstore_dir, exist_ok=True)

    db = chromadb.PersistentClient(path=chroma_path)
    collection = db.get_or_create_collection(settings.rag_chroma_collection)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # 加载已有 docstore（如果有），否则新建
    try:
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=docstore_dir
        )
    except Exception:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

    total_nodes = 0
    for fp_str, md5 in new_files:
        file_name = os.path.basename(fp_str)
        logger.info(f"处理: {file_name}")

        try:
            raw_nodes = _parse_md_to_nodes(fp_str)
        except Exception as e:
            logger.error(f"解析失败 {file_name}: {e}")
            continue

        # ---- 兜底切分：单节点 > chunk_max_size 再切一刀 ----
        final_nodes: List[TextNode] = []
        for n in raw_nodes:
            if len(n.text) > settings.rag_chunk_max_size:
                doc = Document(text=n.text, metadata=n.metadata)
                sub_nodes = splitter.get_nodes_from_documents([doc])
                for sub in sub_nodes:
                    sub.metadata.update(n.metadata)
                final_nodes.extend(sub_nodes)
            else:
                final_nodes.append(n)

        # ---- 注入 doc_id + embedding ----
        texts = [n.text for n in final_nodes]
        embeddings = embed_model.get_text_embedding_batch(texts)
        for n, emb in zip(final_nodes, embeddings):
            n.metadata["doc_id"] = md5
            n.embedding = emb

        # ---- 增量写入 ----
        vector_store.add(final_nodes)

        for n in final_nodes:
            n.embedding = None
        storage_context.docstore.add_documents(final_nodes)
        storage_context.persist(persist_dir=docstore_dir)

        ingested[fp_str] = md5
        total_nodes += len(final_nodes)
        logger.info(f"入库完成: {file_name} → {len(final_nodes)} 个节点")

    _save_ingested(ingested)
    logger.info(f"全部入库完成，共处理 {len(new_files)} 个文件，{total_nodes} 个节点。")


if __name__ == "__main__":
    run_ingestion()
