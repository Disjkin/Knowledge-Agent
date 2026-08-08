"""Trace 查询读取模块 - 从 JSONL 日志中读取和搜索 trace 数据。

提供列表查询（list_traces）和单条查询（get_trace）两个接口，只读不写。
trace 日志存储在 {项目根}/logs/traces/YYYY-MM-DD.jsonl，每行一个完整 trace JSON。
"""

import json
from pathlib import Path
from typing import Optional


# ── 内部工具函数 ──────────────────────────────────────────

def _get_trace_dir() -> Path:
    """获取 trace 日志目录的绝对路径。

    从 settings.trace_dir 读取，相对路径基于项目根解析。
    项目根 = app/tracing/reader.py 往上三层。
    """
    from app.config import settings

    project_root = Path(__file__).resolve().parent.parent.parent
    trace_dir = Path(settings.trace_dir)
    if not trace_dir.is_absolute():
        trace_dir = project_root / trace_dir
    return trace_dir


def _get_jsonl_files(trace_dir: Path, max_days: int = 30) -> list[Path]:
    """获取 trace_dir 下最近 max_days 天的 JSONL 文件，按文件名（日期）倒序排列。

    文件名格式为 YYYY-MM-DD.jsonl，按名称倒序即按日期从新到旧。
    若目录不存在返回空列表。
    """
    if not trace_dir.is_dir():
        return []

    # glob 所有 .jsonl 文件，按文件名倒序（最新的在前）
    files = sorted(trace_dir.glob("*.jsonl"), reverse=True)
    # 只取最近 max_days 个文件（每天最多一个文件）
    return files[:max_days]


def _parse_trace_line(line: str) -> Optional[dict]:
    """解析单行 JSON，失败时返回 None。"""
    try:
        return json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None


def _make_summary(trace: dict) -> dict:
    """从完整 trace 中提取摘要字段。

    query 截断到前 80 字符，其余字段原样返回。
    """
    query = trace.get("query", "")
    if len(query) > 80:
        query = query[:80]
    return {
        "trace_id": trace.get("trace_id"),
        "created_at": trace.get("created_at"),
        "query": query,
        "duration_ms": trace.get("duration_ms"),
        "used_web_search": trace.get("used_web_search"),
        "status": trace.get("status"),
    }


# ── 公开接口 ──────────────────────────────────────────────

def list_traces(limit: int = 20, offset: int = 0) -> list[dict]:
    """按时间倒序返回 trace 摘要列表。

    从最近几天的 JSONL 文件中读取，每条只返回摘要字段：
    - trace_id
    - created_at
    - query（截断到前 80 字符）
    - duration_ms
    - used_web_search
    - status

    Args:
        limit: 返回条数（默认 20）
        offset: 跳过条数（默认 0，用于分页）

    Returns:
        摘要字典列表，按 created_at 倒序排列
    """
    trace_dir = _get_trace_dir()
    files = _get_jsonl_files(trace_dir)

    summaries: list[dict] = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    trace = _parse_trace_line(line)
                    if trace is None:
                        continue
                    summaries.append(_make_summary(trace))
        except OSError:
            # 文件读取失败时跳过该文件
            continue

    # 按 created_at 倒序排列（None/缺失值排到最后）
    summaries.sort(key=lambda s: s.get("created_at") or "", reverse=True)

    # 分页切片
    return summaries[offset : offset + limit]


def get_trace(trace_id: str) -> Optional[dict]:
    """根据 trace_id 查找完整 trace。

    遍历最近几天的 JSONL 文件，找到匹配的 trace_id。

    Args:
        trace_id: 要查找的 trace ID

    Returns:
        完整的 trace JSON dict，找不到返回 None
    """
    trace_dir = _get_trace_dir()
    files = _get_jsonl_files(trace_dir)

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    trace = _parse_trace_line(line)
                    if trace is None:
                        continue
                    if trace.get("trace_id") == trace_id:
                        return trace
        except OSError:
            # 文件读取失败时跳过该文件
            continue

    return None
