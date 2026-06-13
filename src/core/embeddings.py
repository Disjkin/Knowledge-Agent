"""嵌入模型封装 - 优先使用API，也支持本地GPU模型"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# 本地嵌入模型（中文优化，~1.3GB，效果好）
FALLBACK_LOCAL_MODEL = "./models/bge-large-zh-v1.5"

# 批量大小，避免显存溢出
LOCAL_BATCH_SIZE = 64


class EmbeddingModel:
    """嵌入模型接口：优先 API，回退本地GPU模型"""

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
        """确保本地模型已加载（自动检测GPU）"""
        if self._local_model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"加载本地嵌入模型: {FALLBACK_LOCAL_MODEL}（设备: {device}，首次自动下载）...")
            self._local_model = SentenceTransformer(FALLBACK_LOCAL_MODEL, device=device)
            logger.info("本地嵌入模型就绪")

    def _embed_via_api(self, texts: List[str]) -> List[List[float]]:
        """通过API向量化"""
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=60, max_retries=1)
        response = client.embeddings.create(input=texts, model=self.model)
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [e.embedding for e in sorted_data]

    def _embed_locally(self, texts: List[str], progress_callback=None) -> List[List[float]]:
        """通过本地模型向量化（分批处理）"""
        self._ensure_local_model()
        total = len(texts)
        all_embeddings = []

        for i in range(0, total, LOCAL_BATCH_SIZE):
            batch = texts[i:i + LOCAL_BATCH_SIZE]
            batch_embeddings = self._local_model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings.tolist())
            done = min(i + LOCAL_BATCH_SIZE, total)
            logger.info(f"向量化进度: {done}/{total}")
            if progress_callback:
                progress_callback(done, total)

        return all_embeddings

    def embed(self, texts: List[str], show_progress: bool = False, progress_callback=None) -> List[List[float]]:
        """批量文本向量化，progress_callback(done, total) 用于报告进度"""
        if not texts:
            return []

        if self._use_api:
            try:
                return self._embed_via_api(texts)
            except Exception as e:
                logger.warning(f"API嵌入失败，回退本地模型: {e}")
                self._use_api = False  # 后续走本地

        return self._embed_locally(texts, progress_callback=progress_callback)

    def embed_query(self, query: str) -> List[float]:
        """单条查询向量化"""
        results = self.embed([query])
        return results[0]
