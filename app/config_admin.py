"""运行时配置读写 —— 从 .env 读取、写回 .env、即时生效。"""
import os  # noqa: F401  (保留以备扩展)
from pathlib import Path

# .env 路径（项目根）
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# 我们管理的 key
OPENAI_KEYS = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]
ANTHROPIC_KEYS = ["ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"]
LLM_PROVIDER_KEY = "LLM_PROVIDER"


def _read_env() -> dict:
    """读 .env 为 dict（不解析，保留原始文本结构）。"""
    if not ENV_PATH.exists():
        return {}
    out = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            k, v = stripped.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def mask_key(key: str) -> str:
    """对 API key 做前端掩码：仅显示首 4 末 4。"""
    if not key or len(key) <= 10:
        return "•" * 12
    return key[:4] + "•" * (len(key) - 8) + key[-4:]


def get_settings() -> dict:
    """返回前端可读的设置（key 带掩码）。"""
    env = _read_env()
    provider = env.get(LLM_PROVIDER_KEY, "openai").lower()
    if provider == "anthropic":
        raw_key = env.get("ANTHROPIC_API_KEY", "")
    else:
        raw_key = env.get("OPENAI_API_KEY", "")

    return {
        "provider": provider,
        "api_key_masked": mask_key(raw_key),
        "base_url": env.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
        "model": (
            env.get("ANTHROPIC_MODEL", "")
            if provider == "anthropic"
            else env.get("OPENAI_MODEL", "deepseek-chat")
        ),
    }


def apply_settings(payload: dict) -> None:
    """把前端提交的设置写回 .env，并清除各缓存让下次请求即时生效。"""
    env = _read_env()
    provider = (payload.get("provider") or "openai").lower()

    env[LLM_PROVIDER_KEY] = provider

    # api_key：undefined / 空 表示保留原值；掩码表示没改
    new_key = payload.get("api_key")
    if new_key and "•" not in new_key:
        if provider == "anthropic":
            env["ANTHROPIC_API_KEY"] = new_key
        else:
            env["OPENAI_API_KEY"] = new_key

    if provider == "openai":
        if payload.get("base_url"):
            env["OPENAI_BASE_URL"] = payload["base_url"]
        if payload.get("model"):
            env["OPENAI_MODEL"] = payload["model"]
    else:
        if payload.get("model"):
            env["ANTHROPIC_MODEL"] = payload["model"]

    # 写回
    _write_env(env)

    # 清除缓存 → 下一次提问用新配置重建 LLM/embeddings
    _clear_caches()


def _write_env(env: dict) -> None:
    """把管理范围内的 key 写回 .env，保留注释和其他行。"""
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    managed = {LLM_PROVIDER_KEY} | set(OPENAI_KEYS) | set(ANTHROPIC_KEYS)
    out = []
    seen = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        seen.add(key)
        if key in managed:
            if key in env:
                out.append(f"{key}={env[key]}")
            # 管理范围内但 env 里没值 → 跳过（被删除）
        else:
            out.append(line)
    # 追加新 key
    for k in managed:
        if k not in seen and k in env:
            out.append(f"{k}={env[k]}")

    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


def _clear_caches() -> None:
    """清除 LLM / embeddings / settings 的 lru_cache，使新配置生效。"""
    import app.config as cfg
    import app.llm.factory as factory
    import app.rag.retriever as retriever

    cfg.get_settings.cache_clear()
    factory.get_llm.cache_clear()
    factory.get_embeddings.cache_clear()
    try:
        retriever.get_retriever.cache_clear()
    except Exception:
        pass


def test_connection(payload: dict) -> dict:
    """用给定配置实际测一次模型调用，返回 {ok, reply|error}。"""
    from langchain_core.messages import HumanMessage

    provider = (payload.get("provider") or "openai").lower()

    # 确定测试用 key：掩码或 undefined → 先从当前 .env 取一份
    api_key = payload.get("api_key", "")
    if not api_key or "•" in api_key:
        cur = _read_env()
        api_key = cur.get(
            "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY", ""
        )

    model = payload.get("model", "").strip()
    if not model:
        return {"ok": False, "error": "请先填写模型名称"}

    try:
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(model=model, api_key=api_key, max_tokens=64, temperature=0)
        else:
            from langchain_openai import ChatOpenAI

            base_url = payload.get("base_url", "").strip()
            llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url or None,
                max_tokens=64,
                temperature=0,
            )

        resp = llm.invoke([HumanMessage(content="ping")])
        text = getattr(resp, "content", str(resp))
        if isinstance(text, list):
            text = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in text
            )
        return {"ok": True, "reply": str(text).strip()[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)}
