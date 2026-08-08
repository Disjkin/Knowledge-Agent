"""查询改写模块 - Multi-Query + HyDE。

通过 LLM 将用户原始查询改写为多个不同角度的子查询，
或生成假设性文档来提升向量检索的召回率。
"""
import logging
from typing import List

from langchain_core.messages import HumanMessage

from app.config import settings

logger = logging.getLogger(__name__)

MULTI_QUERY_PROMPT = """你是一个查询改写助手。请将用户的问题改写为 {n} 个不同角度的子查询，用于从知识库中检索相关信息。

要求：
1. 每个子查询独立一行，不要编号。
2. 从不同角度表达相同的查询意图（如：换用同义词、聚焦不同方面、换用疑问句式）。
3. 保持原意，不要扩展到不相关的话题。
4. 输出 {n} 行，不要有其他内容。

用户问题：{query}

子查询："""

HYDE_PROMPT = """请根据以下问题，写一段 200 字左右的假设性回答文档。这段文档会被用来从知识库中检索相关信息，所以请尽量用陈述句描述可能包含答案的内容，即使你不确定答案是否正确。

问题：{query}

假设性回答："""


async def multi_query_rewrite(query: str, llm, n: int = None) -> List[str]:
    """将原始查询改写为 n 个子查询（含原始查询）。

    Args:
        query: 用户原始查询
        llm: LLM 实例（LangChain ChatModel）
        n: 子查询数量，默认取 settings.query_rewrite_n

    Returns:
        子查询列表（第一个元素为原始查询，后续为改写查询）
    """
    if n is None:
        n = settings.query_rewrite_n

    try:
        prompt = MULTI_QUERY_PROMPT.format(n=n, query=query)
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        lines = [
            line.strip()
            for line in resp.content.strip().split("\n")
            if line.strip()
        ]
        # 去掉可能的编号前缀（1. 2. 1、 等）
        cleaned: list[str] = []
        for line in lines:
            if len(line) > 2 and line[0].isdigit() and line[1] in ".、)":
                line = line[2:].strip()
            cleaned.append(line)

        # 确保原始查询在列表首位
        if query not in cleaned:
            cleaned.insert(0, query)

        return cleaned[: n + 1]  # 原始 + n 个改写
    except Exception as e:
        logger.warning(f"Multi-query 改写失败: {e}，回退到原始查询")
        return [query]


async def generate_hyde_document(query: str, llm) -> str:
    """生成假设性文档用于 HyDE 检索。

    Args:
        query: 用户原始查询
        llm: LLM 实例

    Returns:
        假设性文档文本（失败时回退到原始查询）
    """
    try:
        prompt = HYDE_PROMPT.format(query=query)
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        return resp.content.strip()
    except Exception as e:
        logger.warning(f"HyDE 生成失败: {e}")
        return query
