"""配置层 - 用 pydantic-settings 读取 .env。"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

# 本机所有嵌入/重排模型均已缓存到本地（HF hub + ModelScope）。
# 强制 HuggingFace 离线模式，避免 sentence-transformers 每次加载都联网校验
# huggingface.co（国内不可达，会导致启动卡死数分钟）。
# 若日后需要下载新模型，临时在环境变量里覆盖这两项即可。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 服务
    host: str = "127.0.0.1"
    port: int = 8000

    # LLM 提供者: openai | anthropic
    llm_provider: str = "openai"

    # OpenAI 兼容
    openai_api_key: str = "sk-placeholder"
    openai_base_url: str = "https://api.deepseek.com"
    openai_model: str = "deepseek-chat"
    # 推理模型（前端「深度思考」开关用，如 deepseek-reasoner；留空则回落到 openai_model）
    openai_reasoning_model: str = ""

    # Anthropic
    anthropic_api_key: str = "sk-ant-placeholder"
    anthropic_model: str = "claude-sonnet-4-20250514"

    # 嵌入: local | openai
    embedding_provider: str = "local"
    embedding_backend: str = "onnx"             # pytorch | onnx（bge-m3 用 onnx 加速）
    local_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"
    local_embedding_int8: bool = False             # INT8 量化（bge-m3 等大模型在 GPU 上启用）

    # RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieve_k: int = 5

    # 查询改写
    enable_query_rewrite: bool = True              # Multi-Query 改写
    query_rewrite_n: int = 4                       # 改写子查询数量
    enable_hyde: bool = False                      # HyDE 假设文档检索

    # 混合检索
    enable_bm25: bool = True                       # 启用 BM25 关键词检索

    # 重排
    enable_reranker: bool = True                   # Cross-Encoder 重排
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_top_k: int = 5                        # 重排后返回数量
    rerank_candidate_k: int = 20                   # 重排前候选池大小
    enable_unified_reranker: bool = True           # 网搜后对 KB+web 统一 cross-encoder 重排

    # 路径（相对项目根）
    data_dir: str = "data"
    persist_dir: str = "chroma_db"
    collection_name: str = "knowledge_base"

    # ── 网搜兜底 ──
    web_search_enabled: bool = False
    web_search_provider: str = "doubao"          # doubao | tavily | duckduckgo
    tavily_api_key: str = ""
    doubao_search_api_key: str = ""
    doubao_search_url: str = "https://open.feedcoopapi.com/search_api/global_search"
    web_search_max_results: int = 5
    web_search_timeout: int = 15
    web_search_proxy: str = ""                   # 网搜代理地址，如 http://127.0.0.1:7897
    relevance_threshold: float = 0.3             # 本地检索最高分低于此值 -> 走网搜
    web_search_variance_threshold: float = 0.05  # top-k 分数方差低于此值 -> 走网搜

    # ── 链路追踪 ──
    trace_enabled: bool = True
    trace_dir: str = "logs/traces"
    trace_max_field_len: int = 2000


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
