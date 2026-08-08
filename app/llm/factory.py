"""LLM + Embeddings 多模型工厂 - 根据 settings 创建对应实例，统一缓存。"""
from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=None)
def get_llm(model_override: str = ""):
    """根据 LLM_PROVIDER 返回带 streaming=True 的聊天模型。

    model_override: 仅 openai 兼容 provider 生效，
    用于「深度思考」开关切换到推理模型（如 deepseek-reasoner）。
    """
    provider = settings.llm_provider.lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            streaming=True,
            temperature=0.2,
        )

    # 默认 openai 兼容（DeepSeek / 通义 / 智谱 / OpenAI ...）
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_override or settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        streaming=True,
        temperature=0.2,
    )


def get_generation_llm(deep_think: bool = False):
    """最终生成用的 LLM。

    deep_think=True 且配置了 openai_reasoning_model 时切换到推理模型；
    否则回落到当前默认模型（前端开关表现为无感）。
    """
    if (
        deep_think
        and settings.llm_provider.lower() == "openai"
        and settings.openai_reasoning_model
    ):
        return get_llm(settings.openai_reasoning_model)
    return get_llm()


def _resolve_model_path(model_name: str) -> str:
    """优先从 ModelScope 缓存加载，找不到则回退 HuggingFace。"""
    import os

    ms_cache = os.path.expanduser(
        f"~/.cache/modelscope/models/{model_name.replace('/', '--')}/snapshots/master"
    )
    if os.path.isdir(ms_cache) and os.path.exists(os.path.join(ms_cache, "config.json")):
        return ms_cache
    return model_name


class _QuantizedEmbeddings:
    """INT8 量化的本地嵌入模型。

    GPU 环境：使用 bitsandbytes 8bit 量化加载（load_in_8bit=True），
        bge-m3 约 568MB VRAM，RTX 2060 6GB 绰绰有余。
    CPU 环境：使用 PyTorch dynamic quantization 对 Linear 层做 INT8 量化。
    实现 LangChain Embeddings 接口（embed_documents / embed_query）。
    """

    def __init__(self, model_name: str):
        import torch
        from sentence_transformers import SentenceTransformer

        model_path = _resolve_model_path(model_name)
        use_cuda = torch.cuda.is_available()

        if use_cuda:
            # GPU: 用 bitsandbytes 8bit 量化
            try:
                self._model = SentenceTransformer(
                    model_path,
                    device="cuda",
                    model_kwargs={"load_in_8bit": True},
                )
                print(f"[factory] bge-m3 加载到 GPU (bitsandbytes INT8)")
            except Exception as e:
                # bitsandbytes 不可用 -> FP16 回退
                print(f"[factory] bitsandbytes 8bit 加载失败 ({e})，回退 FP16")
                self._model = SentenceTransformer(
                    model_path,
                    device="cuda",
                    model_kwargs={"torch_dtype": torch.float16},
                )
                print(f"[factory] bge-m3 加载到 GPU (FP16)")
        else:
            # CPU: PyTorch dynamic quantization
            self._model = SentenceTransformer(model_path, device="cpu")
            self._model[0].auto_model = torch.quantization.quantize_dynamic(
                self._model[0].auto_model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            print(f"[factory] bge-m3 加载到 CPU (PyTorch INT8 动态量化)")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()


@lru_cache(maxsize=None)
def get_embeddings():
    """根据 EMBEDDING_PROVIDER 返回嵌入模型。"""
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    # 默认 local - ONNX 后端（bge-m3 在 CPU 上比 PyTorch 快 5-10 倍）
    if getattr(settings, "embedding_backend", "pytorch").lower() == "onnx":
        try:
            from app.llm.onnx_embeddings import get_onnx_embeddings

            return get_onnx_embeddings()
        except Exception as e:
            print(f"[factory] ONNX 嵌入加载失败: {e}，回退 PyTorch 后端")
            # fall through to pytorch backend

    model_name = settings.local_embedding_model
    try:
        if getattr(settings, "local_embedding_int8", False):
            return _QuantizedEmbeddings(model_name)
    except Exception as e:
        print(f"[factory] 加载 {model_name}（INT8）失败: {e}")
        # 回退到 bge-small-zh-v1.5（已缓存）
        fallback = "BAAI/bge-small-zh-v1.5"
        print(f"[factory] 回退到 {fallback}")
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=fallback,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
