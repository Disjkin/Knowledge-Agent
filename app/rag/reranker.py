"""重排模块 - Cross-Encoder 重排序。

用 cross-encoder 模型对检索候选集重新打分排序，
提升最终送入 LLM 的上下文质量。

与双塔模型（bge-m3 embedding）不同，cross-encoder 让 query 和 document
在 attention 层交互，精度显著更高，是 RAG 系统的标准组件。
"""
import logging
from typing import List, Tuple

import numpy as np
from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)

# 模型缓存（进程级单例）
_reranker_model = None


def _get_reranker():
    """获取（或惰性初始化）重排模型。"""
    global _reranker_model

    if _reranker_model is not None:
        return _reranker_model

    try:
        from sentence_transformers import CrossEncoder

        model_name = settings.reranker_model
        # 优先检查 ModelScope 缓存（国内下载走的 ModelScope）
        import os
        ms_cache = os.path.expanduser(
            f"~/.cache/modelscope/models/{model_name.replace('/', '--')}/snapshots/master"
        )
        if os.path.isdir(ms_cache) and os.path.exists(os.path.join(ms_cache, "config.json")):
            model_path = ms_cache
        else:
            model_path = model_name

        _reranker_model = CrossEncoder(model_path, device="cpu")
        logger.info(f"重排模型加载完成: {model_path}")
    except Exception as e:
        logger.error(f"重排模型加载失败: {e}")
        _reranker_model = None

    return _reranker_model


def rerank(
    query: str,
    docs: List[Document],
    top_k: int = None,
) -> List[Tuple[Document, float]]:
    """用 cross-encoder 对候选文档重排。

    Args:
        query: 查询文本
        docs: 候选文档列表（来自混合检索的候选池）
        top_k: 返回前 top_k 个，默认取 settings.reranker_top_k

    Returns:
        [(Document, score), ...] 按重排分数从高到低
    """
    if top_k is None:
        top_k = settings.reranker_top_k

    if not docs:
        return []

    model = _get_reranker()
    if model is None:
        # 模型不可用，按原始顺序返回
        logger.warning("重排模型不可用，按原始顺序返回")
        return [(doc, 0.0) for doc in docs[:top_k]]

    try:
        # 构造 query-doc pairs
        pairs = [(query, doc.page_content) for doc in docs]

        # 批量打分
        scores = model.predict(pairs, batch_size=16)
        scores = np.array(scores)

        # 按分数降序排序，取 top_k
        sorted_indices = np.argsort(scores)[::-1][:top_k]

        results: list[tuple[Document, float]] = []
        for idx in sorted_indices:
            results.append((docs[idx], float(scores[idx])))

        return results
    except Exception as e:
        logger.error(f"重排失败: {e}，返回原始顺序")
        return [(doc, 0.0) for doc in docs[:top_k]]
