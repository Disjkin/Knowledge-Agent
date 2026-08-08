"""链路追踪模块 - 记录每次查询的完整调用链。

一次 query 一条 trace，步骤用上下文管理器计时。
trace 以 JSONL 格式按天滚动存储，请求结束时一次性写入。
"""

import contextlib
import json
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


# ── 工具函数 ──────────────────────────────────────────────

def _truncate(obj: Any, max_len: int) -> Any:
    """递归截断长文本字段，超过 max_len 的字符串截断后追加 '...[truncated]'。"""
    if isinstance(obj, str):
        if len(obj) > max_len:
            return obj[:max_len] + "...[truncated]"
        return obj
    elif isinstance(obj, dict):
        return {k: _truncate(v, max_len) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_truncate(item, max_len) for item in obj]
    else:
        return obj


# ── 步骤上下文管理器 ─────────────────────────────────────

class _StepContext(contextlib.AbstractContextManager):
    """单个步骤的上下文管理器。

    __enter__ 返回自身，调用方可通过 set_outputs() 设置输出；
    __exit__ 时计算耗时并记录状态。若 with 块内抛异常，
    status 置为 "error" 且记录异常信息，异常本身不被抑制。
    """

    def __init__(self, seq: int, name: str, inputs: dict) -> None:
        self._seq: int = seq
        self._name: str = name
        self._inputs: dict = inputs
        self._outputs: Optional[dict] = None
        self._started_at: Optional[str] = None
        self._start_dt: Optional[datetime] = None
        self._duration_ms: int = 0
        self._status: str = "ok"
        self._error: Optional[str] = None

    def set_outputs(self, outputs: dict) -> None:
        """设置该步骤的输出数据，在 with 块内调用。"""
        self._outputs = outputs

    def __enter__(self) -> "_StepContext":
        now = datetime.now()
        self._started_at = now.astimezone().isoformat()
        self._start_dt = now
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # 计算耗时（整数毫秒）
        if self._start_dt is not None:
            self._duration_ms = int(
                (datetime.now() - self._start_dt).total_seconds() * 1000
            )
        # 异常 -> error 状态
        if exc_type is not None:
            self._status = "error"
            self._error = str(exc_val) if exc_val is not None else "Unknown error"
        # 不抑制异常，让其继续传播
        return False

    def to_dict(self, max_len: int) -> dict:
        """转换为可序列化的字典（应用字段截断）。"""
        return {
            "seq": self._seq,
            "name": self._name,
            "started_at": self._started_at,
            "duration_ms": self._duration_ms,
            "inputs": _truncate(self._inputs, max_len),
            "outputs": _truncate(self._outputs, max_len) if self._outputs is not None else None,
            "status": self._status,
            "error": _truncate(self._error, max_len) if self._error else None,
        }


class _NoopStepContext(contextlib.AbstractContextManager):
    """空操作步骤上下文，零开销。"""

    def __enter__(self) -> "_NoopStepContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def set_outputs(self, outputs: dict) -> None:
        pass


# ── Tracer ───────────────────────────────────────────────

class Tracer:
    """一次 query 一条 trace。步骤用上下文管理器计时。"""

    # 类级别锁，所有 Tracer 实例共享，保护文件写入
    _write_lock = threading.Lock()

    def __init__(self, query: str, history_len: int) -> None:
        self._trace_id: str = uuid.uuid4().hex
        self._created_at: str = datetime.now().astimezone().isoformat()
        self._start_dt: datetime = datetime.now()
        self._query: str = query
        self._history_len: int = history_len
        self._model: Optional[str] = None
        self._used_web_search: bool = False
        self._answer: Optional[str] = None
        self._status: str = "ok"
        self._error: Optional[str] = None
        self._step_contexts: List[_StepContext] = []
        self._step_counter: int = 0

    @property
    def trace_id(self) -> str:
        """公开 trace_id 供外部读取。"""
        return self._trace_id

    def step(self, name: str, inputs: dict) -> _StepContext:
        """创建一个步骤上下文管理器。

        返回的对象可在 with 块内调用 set_outputs() 设置输出。
        """
        self._step_counter += 1
        ctx = _StepContext(self._step_counter, name, inputs)
        self._step_contexts.append(ctx)
        return ctx

    def set_result(self, answer: str, used_web_search: bool, model: str) -> None:
        """设置查询结果。"""
        self._answer = answer
        self._used_web_search = used_web_search
        self._model = model
        self._status = "ok"

    def set_error(self, message: str) -> None:
        """设置错误信息。"""
        self._error = message
        self._status = "error"

    def save(self) -> None:
        """组装完整 trace JSON 并追加写入当天 JSONL 文件。"""
        from app.config import settings

        max_len = settings.trace_max_field_len

        # 计算总耗时（整数毫秒）
        duration_ms = int(
            (datetime.now() - self._start_dt).total_seconds() * 1000
        )

        # 组装完整 trace
        trace: Dict[str, Any] = {
            "trace_id": self._trace_id,
            "created_at": self._created_at,
            "duration_ms": duration_ms,
            "query": _truncate(self._query, max_len),
            "history_len": self._history_len,
            "model": self._model,
            "used_web_search": self._used_web_search,
            "status": self._status,
            "error": _truncate(self._error, max_len) if self._error else None,
            "steps": [ctx.to_dict(max_len) for ctx in self._step_contexts],
            "answer": _truncate(self._answer, max_len) if self._answer else None,
        }

        # 解析 trace_dir（相对路径基于项目根 = app/ 的父目录）
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        trace_dir = settings.trace_dir
        if not os.path.isabs(trace_dir):
            trace_dir = os.path.join(project_root, trace_dir)

        # 确保目录存在
        os.makedirs(trace_dir, exist_ok=True)

        # 当天 JSONL 文件（按天滚动）
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = os.path.join(trace_dir, f"{today}.jsonl")

        # 加锁写入一行 JSON（避免并发写交错）
        line = json.dumps(trace, ensure_ascii=False)
        with Tracer._write_lock:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


# ── NoopTracer（trace_enabled=False 时使用）──────────────

class NoopTracer:
    """空操作 Tracer，trace_enabled=False 时使用。零开销。"""

    trace_id: Optional[str] = None

    def step(self, name: str, inputs: dict) -> _NoopStepContext:
        return _NoopStepContext()

    def set_result(self, answer: str, used_web_search: bool, model: str) -> None:
        pass

    def set_error(self, message: str) -> None:
        pass

    def save(self) -> None:
        pass


# ── 工厂函数 ─────────────────────────────────────────────

def get_tracer(query: str, history_len: int) -> Union[Tracer, NoopTracer]:
    """工厂函数；trace_enabled=False 时返回空操作实现。"""
    from app.config import settings

    if settings.trace_enabled:
        return Tracer(query, history_len)
    return NoopTracer()
