"""FastAPI 后端 — 路由 + SSE 流式 + 静态托管 + 启动时自动索引。"""
import asyncio
import json
import os
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models.schemas import ChatRequest
import app.config_admin as config_admin

# 项目根（main.py 往上两层 → 项目根）
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"


# ── 启动时：若向量库为空则自动索引 ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.rag.ingest import ingest
    from app.rag.retriever import get_vectorstore

    vs = get_vectorstore()
    try:
        existing = vs.get()
        count = len(existing.get("ids", []) or [])
    except Exception:
        count = 0

    if count == 0:
        print("[startup] 知识库为空，开始自动索引 data/ ...")
        try:
            stats = ingest()
            print(f"[startup] 索引完成: {stats}")
        except Exception as e:
            print(f"[startup] 自动索引失败: {e}")
    else:
        print(f"[startup] 知识库已有 {count} 个片段，跳过索引。")
        # 检测嵌入模型是否变更（维度不兼容时 query 会报错 → 重建）
        try:
            from app.llm.factory import get_embeddings

            query_vec = get_embeddings().embed_query("测试")
            vs.query(
                query_embeddings=[query_vec],
                n_results=1,
                include=["documents"],
            )
        except Exception:
            print("[startup] 检测到嵌入模型变更，重建知识库…")
            try:
                vs.delete(ids=vs.get()["ids"])
                stats = ingest()
                print(f"[startup] 重建完成: {stats}")
            except Exception as e:
                print(f"[startup] 重建失败: {e}")
    yield


app = FastAPI(title="个人知识库助手", version="1.0.0", lifespan=lifespan)

# CORS（本地开发允许全部来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── SSE 帧构造 ──
def sse_frame(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── 路由 ──
@app.get("/api/health")
async def health():
    return {"status": "ok", "model": _current_model()}


def _current_model() -> str:
    if settings.llm_provider.lower() == "anthropic":
        return f"anthropic:{settings.anthropic_model}"
    return f"openai:{settings.openai_model}@{settings.openai_base_url}"


@app.get("/api/documents")
async def list_documents():
    """返回已索引的文档名列表。"""
    from app.rag.retriever import get_vectorstore

    vs = get_vectorstore()
    try:
        data = vs.get()
        sources = set()
        for m in data.get("metadatas", []) or []:
            if m and "source" in m:
                sources.add(m["source"])
        return {"documents": sorted(sources), "total": len(sources)}
    except Exception as e:
        return {"documents": [], "total": 0, "error": str(e)}


@app.post("/api/open-data-folder")
async def open_data_folder():
    """在文件管理器中打开 data/ 文件夹。"""
    data_path = (BASE_DIR / settings.data_dir).resolve()
    data_path.mkdir(exist_ok=True)
    opened = False
    err = None
    try:
        # 显式调 explorer，避免后台进程调 startfile 无效
        if os.name == "nt":
            subprocess.Popen(
                ["explorer", str(data_path)],
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            opened = True
        elif hasattr(os, "startfile"):
            os.startfile(str(data_path))
            opened = True
        else:
            subprocess.Popen(["xdg-open", str(data_path)])
            opened = True
    except Exception as e:
        err = str(e)
    return {"status": "ok" if opened else "error", "path": str(data_path), "error": err}


# ── 运行时模型设置（无需重启） ──
from starlette.requests import Request as _StarletteRequest


@app.get("/api/settings")
async def read_settings():
    return config_admin.get_settings()


@app.post("/api/settings")
async def save_settings(req: _StarletteRequest):
    try:
        body = await req.json()
        payload = {k: body[k] for k in ("provider", "api_key", "base_url", "model") if k in body}
        config_admin.apply_settings(payload)
        return {"ok": True, "settings": config_admin.get_settings()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/settings/test")
async def test_settings(req: _StarletteRequest):
    try:
        body = await req.json()
        payload = {k: body[k] for k in ("provider", "api_key", "base_url", "model") if k in body}
        return config_admin.test_connection(payload)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/ingest")
async def trigger_ingest(clear: bool = False):
    """手动触发重建知识库。

    clear=true 时先清空全部向量再从头索引（用于删除文件后清理孤儿片段、
    或想干净重来）；默认 false 走增量判重。
    """
    from app.rag.ingest import ingest, reset_bm25_index

    stats = ingest(clear=clear)
    reset_bm25_index()  # 重建 BM25 索引缓存
    return {"status": "ok", "cleared": clear, **stats}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """SSE 流式聊天。"""
    from app.rag.chain import stream_chat

    async def event_stream():
        try:
            async for event, data in stream_chat(
                req.message,
                req.history,
                web_search=req.web_search,
                deep_think=req.deep_think,
            ):
                yield sse_frame(event, data)
        except Exception as e:
            yield sse_frame("error", {"message": f"服务异常: {e}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )


# ── Trace 查询接口 ──
@app.get("/api/traces")
async def list_traces_api(limit: int = 20, offset: int = 0):
    """按时间倒序返回 trace 摘要列表。"""
    from app.tracing.reader import list_traces

    return {"traces": list_traces(limit=limit, offset=offset)}


@app.get("/api/traces/{trace_id}")
async def get_trace_api(trace_id: str):
    """返回指定 trace 的完整 JSON。"""
    from app.tracing.reader import get_trace

    trace = get_trace(trace_id)
    if trace is None:
        return {"error": "trace not found", "trace_id": trace_id}
    return trace


# ── 静态前端托管 ──
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
