"""检索器 - 封装检索逻辑"""

import logging
from typing import List

from .document_loader import Document
from .embeddings import EmbeddingModel
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """检索器，负责从向量库中检索相关文档"""

    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel, top_k: int = 5):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """检索与query相关的文档"""
        k = top_k or self.top_k

        # 1. 查询向量化
        query_embedding = self.embedding_model.embed_query(query)

        # 2. 向量检索
        results = self.vector_store.search(query_embedding, top_k=k)

        # 3. 转换为Document
        documents = []
        for r in results:
            doc = Document(
                content=r["content"],
                metadata={**r.get("metadata", {}), "relevance_score": 1 - r.get("distance", 0)},
            )
            documents.append(doc)

        logger.info(f"检索到 {len(documents)} 条相关文档")
        return documents
