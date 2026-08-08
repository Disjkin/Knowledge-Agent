"""从已有 Chroma 集合构建 top-k 向量检索器。

注意：使用 Chroma 原生 API（不用 langchain-chroma 封装），
因为 langchain-chroma 在 bge-m3 嵌入下会异常卡住。
"""
import asyncio
from functools import lru_cache

from app.config import settings


def get_vectorstore():
    """返回 Chroma 原生集合（供启动时判断是否空库）。"""
    import chromadb

    client = chromadb.PersistentClient(path=settings.persist_dir)
    col = client.get_or_create_collection(
        settings.collection_name, metadata={"hnsw:space": "cosine"}
    )
    return col


async def search_with_scores(
    query: str, k: int = None
) -> list[tuple["Document", float]]:
    """带相关性分数的向量检索。

    使用 Chroma 原生 query API（cosine 距离），
    返回 (Document, score) 列表，score 越高越相关（score = 1 - distance）。

    Args:
        query: 查询文本
        k: 返回数量，默认取 settings.retrieve_k

    Returns:
        [(Document, float), ...] 按分数从高到低排列
    """
    from langchain_core.documents import Document  # noqa: F811  用于类型注解

    from app.llm.factory import get_embeddings

    if k is None:
        k = settings.retrieve_k

    def _search():
        col = get_vectorstore()
        emb = get_embeddings()
        query_vec = emb.embed_query(query)
        results = col.query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        docs: list[tuple[Document, float]] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for i in range(len(ids)):
            doc = Document(
                page_content=documents[i] if i < len(documents) else "",
                metadata=metadatas[i] if i < len(metadatas) else {},
            )
            # cosine 距离转相似度分数: score = 1 - distance
            dist = distances[i] if i < len(distances) else 1.0
            score = max(0.0, 1.0 - dist)
            docs.append((doc, score))
        return docs

    # 同步方法，用 asyncio 包装
    return await asyncio.to_thread(_search)
