"""嵌入模型封装 - 优先使用API，也支持轻量本地模型"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# 内置的极轻量模型（~80MB，之前的 1/6 大小）
FALLBACK_LOCAL_MODEL = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """嵌入模型接口：优先 API，回退本地轻量模型"""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
    ):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key
        self.model = model
        self._local_model = None
        self._use_api = bool(base_url and api_key and model)

    @property
    def is_loaded(self) -> bool:
        if self._use_api:
            return True
        return self._local_model is not None

    def load_model(self):
        """预加载（仅本地模式生效）"""
        if not self._use_api:
            self._ensure_local_model()

    def _ensure_local_model(self):
        """确保本地模型已加载"""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"加载本地嵌入模型: {FALLBACK_LOCAL_MODEL}（~80MB，首次自动下载）...")
            self._local_model = SentenceTransformer(FALLBACK_LOCAL_MODEL)
            logger.info("本地嵌入模型就绪")

    def _embed_via_api(self, texts: List[str]) -> List[List[float]]:
        """通过API向量化"""
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=60, max_retries=1)
        response = client.embeddings.create(input=texts, model=self.model)
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [e.embedding for e in sorted_data]

    def _embed_locally(self, texts: List[str]) -> List[List[float]]:
        """通过本地模型向量化"""
        self._ensure_local_model()
        embeddings = self._local_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """批量文本向量化"""
        if not texts:
            return []

        if self._use_api:
            try:
                return self._embed_via_api(texts)
            except Exception as e:
                logger.warning(f"API嵌入失败，回退本地模型: {e}")
                self._use_api = False  # 后续走本地

        return self._embed_locally(texts)

    def embed_query(self, query: str) -> List[float]:
        """单条查询向量化"""
        results = self.embed([query])
        return results[0]
