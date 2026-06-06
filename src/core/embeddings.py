"""嵌入模型封装 - 将文本转换为向量表示"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """嵌入模型接口，基于sentence-transformers"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"加载嵌入模型: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info("嵌入模型加载完成")
            except Exception as e:
                logger.error(f"嵌入模型加载失败: {e}")
                raise

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化"""
        self._load_model()
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """单条查询向量化"""
        self._load_model()
        embedding = self._model.encode([query], show_progress_bar=False)
        return embedding[0].tolist()
