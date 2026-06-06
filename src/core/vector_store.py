"""向量数据库封装 - 基于ChromaDB的本地向量存储"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class VectorStore:
    """向量数据库接口，使用ChromaDB"""

    def __init__(self, persist_directory: str = "./data/chroma", collection_name: str = "knowledge"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _init_client(self):
        """延迟初始化ChromaDB客户端"""
        if self._client is None:
            try:
                import chromadb
                self._client = chromadb.PersistentClient(path=self.persist_directory)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"ChromaDB初始化完成，集合: {self.collection_name}")
            except Exception as e:
                logger.error(f"ChromaDB初始化失败: {e}")
                raise

    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: Optional[List[str]] = None,
    ):
        """添加向量到数据库"""
        self._init_client()
        if ids is None:
            import hashlib
            ids = [hashlib.md5(t.encode()).hexdigest()[:16] for t in texts]

        self._collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"已添加 {len(texts)} 条向量")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[dict]:
        """相似度检索"""
        self._init_client()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = []
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                doc = {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                }
                documents.append(doc)
        return documents

    def delete(self, ids: List[str]):
        """删除指定向量"""
        self._init_client()
        self._collection.delete(ids=ids)
        logger.info(f"已删除 {len(ids)} 条向量")

    def count(self) -> int:
        """获取向量总数"""
        self._init_client()
        return self._collection.count()

    def clear(self):
        """清空集合"""
        self._init_client()
        # 删除并重建集合
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("向量集合已清空")
