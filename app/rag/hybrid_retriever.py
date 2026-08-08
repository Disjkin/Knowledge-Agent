"""混合检索模块 - BM25 + 向量检索 + RRF 融合。

将关键词检索（BM25）与语义检索（向量）的结果通过 Reciprocal Rank Fusion 融合，
兼顾精确匹配和语义理解。
"""
import asyncio
import logging
from typing import List, Tuple

import numpy as np
from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)

# BM25 索引缓存（进程级单例）
_bm25_index = None
_bm25_chunks: List[Document] = []


def _build_bm25_index():
    """从 Chroma 加载所有文档构建 BM25 索引（惰性初始化）。"""
    global _bm25_index, _bm25_chunks

    if _bm25_index is not None:
        return

    try:
        from rank_bm25 import BM25Okapi

        from app.rag.retriever import get_vectorstore

        db = get_vectorstore()
        data = db.get()

        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        if not documents:
            logger.warning("BM25 索引构建：知识库为空")
            return

        _bm25_chunks = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(documents, metadatas)
        ]

        # 中文分词：按字符切分（简单但对中文关键词匹配有效）
        tokenized = [list(doc.page_content) for doc in _bm25_chunks]
        _bm25_index = BM25Okapi(tokenized)

        logger.info(f"BM25 索引构建完成，共 {len(_bm25_chunks)} 个文档")
    except Exception as e:
        logger.error(f"BM25 索引构建失败: {e}")
        _bm25_index = None


def bm25_search(query: str, k: int = 10) -> List[Tuple[Document, float]]:
    """BM25 关键词检索。

    Args:
        query: 查询文本
        k: 返回数量

    Returns:
        [(Document, score), ...] 按分数从高到低
    """
    _build_bm25_index()

    if _bm25_index is None or not _bm25_chunks:
        return []

    # 中文按字符分词
    tokenized_query = list(query)
    scores = _bm25_index.get_scores(tokenized_query)

    # 取 top-k（score > 0）
    top_indices = np.argsort(scores)[::-1][:k]

    results: list[tuple[Document, float]] = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append((_bm25_chunks[idx], float(scores[idx])))

    return results


def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[Document, float]]],
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """Reciprocal Rank Fusion 融合多路检索结果。

    RRF 公式: score(d) = Σ 1/(k + rank(d))

    Args:
        result_lists: 多路检索结果列表，每路是 [(Document, score), ...]
        k: RRF 参数，默认 60

    Returns:
        融合后的 [(Document, score), ...] 按融合分数从高到低
    """
    doc_scores: dict = {}
    doc_map: dict = {}

    for result_list in result_lists:
        for rank, (doc, _score) in enumerate(result_list):
            # 用 chunk_id 或 page_content hash 作为唯一标识
            key = doc.metadata.get("chunk_id", hash(doc.page_content))
            if key not in doc_map:
                doc_map[key] = doc
                doc_scores[key] = 0.0
            doc_scores[key] += 1.0 / (k + rank + 1)

    # 按融合分数排序
    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    return [(doc_map[key], doc_scores[key]) for key in sorted_keys]


async def hybrid_search(
    query: str,
    k: int = None,
) -> List[Tuple[Document, float]]:
    """混合检索：向量 + BM25 + RRF 融合。

    Args:
        query: 查询文本
        k: 最终返回数量（实际返回 candidate_k 供后续 rerank）

    Returns:
        [(Document, score), ...] 按融合分数从高到低
    """
    from app.rag.retriever import search_with_scores

    if k is None:
        k = settings.retrieve_k

    # 候选池扩大（用于 rerank）
    candidate_k = settings.rerank_candidate_k

    # 并行执行两路检索
    vector_task = search_with_scores(query, candidate_k)
    bm25_task = asyncio.to_thread(bm25_search, query, candidate_k)

    vector_results, bm25_results = await asyncio.gather(
        vector_task, bm25_task, return_exceptions=True
    )

    # 处理异常
    if isinstance(vector_results, Exception):
        logger.warning(f"向量检索失败: {vector_results}")
        vector_results = []
    if isinstance(bm25_results, Exception):
        logger.warning(f"BM25 检索失败: {bm25_results}")
        bm25_results = []

    if not vector_results and not bm25_results:
        return []

    # RRF 融合
    result_lists: list[list[tuple[Document, float]]] = []
    if vector_results:
        result_lists.append(vector_results)
    if bm25_results:
        result_lists.append(bm25_results)

    fused = reciprocal_rank_fusion(result_lists)
    return fused[:candidate_k]


def reset_bm25_index():
    """重置 BM25 索引缓存（重新入库后调用）。"""
    global _bm25_index, _bm25_chunks
    _bm25_index = None
    _bm25_chunks = []
