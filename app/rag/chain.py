"""RAG 链 + 流式输出。
事件序列：sources(kb) -> [status -> sources(web)] -> reasoning × N -> token × N -> done （异常时 error）。
trace 步骤：query_rewrite -> hybrid_retrieve -> rerank -> [web_search] -> llm_stream -> reply。
"""
import statistics
from typing import AsyncGenerator, List, Tuple
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app.config import settings

# ── 系统提示词：语言跟随 + 引用 + 抗幻觉 + 网搜 ──
SYSTEM_PROMPT = """你是知识库助手。请严格依据下面【参考资料】回答用户问题。

规则：
1. 用提问的语言回答（中文问则中文答，英文问则英文答）。
2. 回答必须附上依据，本地资料引用格式如 [来源: 文件名#片段编号]，网络资料引用格式如 [来源: 标题](URL)。
3. 若参考资料中没有答案，请明确告知"知识库中未找到相关信息"，绝不要编造。
4. 回答要简洁、有条理。
5. 若使用了网络搜索资料，回答末尾需注明"以上内容包含网络搜索结果，请注意甄别时效性"。"""

# ── 上下文格式化 ──
def format_context(kb_chunks: List[Document], web_chunks: List[Document]) -> str:
    """把本地知识库片段和网络搜索片段拼成带编号的参考资料文本。"""
    kb_pieces: list[str] = []
    for i, c in enumerate(kb_chunks, 1):
        src = c.metadata.get("source", "未知")
        cid = c.metadata.get("chunk_id", i - 1)
        kb_pieces.append(f"[片段 {i}] 来源: {src}#{cid}\n{c.page_content}")

    web_pieces: list[str] = []
    for i, c in enumerate(web_chunks, 1):
        title = c.metadata.get("source", "未知")
        url = c.metadata.get("url", "")
        web_pieces.append(f"[网络资料 {i}] 来源: {title} ({url})\n{c.page_content}")

    sections: list[str] = []
    if kb_pieces:
        sections.append("【本地知识库资料】\n" + "\n\n".join(kb_pieces))
    if web_pieces:
        sections.append("【网络搜索资料】\n" + "\n\n".join(web_pieces))
    return "\n\n".join(sections)


def chunks_to_sources(chunks: List[Document]) -> List[dict]:
    """把 chunk 列表转成前端可用的引用卡片数据。"""
    return [
        {
            "index": i + 1,
            "source": c.metadata.get("source", "未知"),
            "file_name": c.metadata.get("file_name", ""),
            "chunk_id": c.metadata.get("chunk_id", i),
            "snippet": c.page_content[:300],
            "type": c.metadata.get("type", "kb"),
            "url": c.metadata.get("url", ""),
            "score": round(c.metadata.get("score", 0.0), 4),
        }
        for i, c in enumerate(chunks)
    ]


def _web_results_to_docs(results: list[dict]) -> list[Document]:
    """把网搜结果列表转为 Document 对象。"""
    docs: list[Document] = []
    for i, r in enumerate(results):
        url = r.get("url", "")
        try:
            domain = urlparse(url).netloc
        except Exception:
            domain = ""
        docs.append(Document(
            page_content=r.get("content", ""),
            metadata={
                "source": r.get("title", ""),
                "file_name": domain,
                "chunk_id": i,
                "type": "web",
                "url": url,
            },
        ))
    return docs


def _get_model_str() -> str:
    """返回当前 LLM 的标识字符串。"""
    if settings.llm_provider == "openai":
        return f"openai:{settings.openai_model}@{settings.openai_base_url}"
    elif settings.llm_provider == "anthropic":
        return f"anthropic:{settings.anthropic_model}"
    return settings.llm_provider


def _dedup_by_chunk_id(docs_with_scores: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    """按 chunk_id 去重，保留分数最高的。"""
    seen: dict = {}
    for doc, score in docs_with_scores:
        key = doc.metadata.get("chunk_id", id(doc))
        if key not in seen or score > seen[key][1]:
            seen[key] = (doc, score)
    return list(seen.values())


# ── 主入口：异步 SSE 流 ──
async def stream_chat(
    question: str,
    history: list = None,
    web_search: bool | None = None,
    deep_think: bool = False,
) -> AsyncGenerator[Tuple[str, object], None]:
    """异步生成 SSE 事件: (event_name, data)。

    完整流程：
    1. 查询改写（Multi-Query / HyDE）
    2. 混合检索（向量 + BM25 + RRF 融合）
    3. Cross-Encoder 重排
    4. 网搜判定（分数阈值 + 方差判定；web_search 参数可手动 强制/关闭）
    5. LLM 流式生成（deep_think=True 且配置推理模型时用推理模型）
    """
    from app.llm.factory import get_llm, get_generation_llm
    from app.rag.retriever import search_with_scores
    from app.tools.web_search import search_web
    from app.tracing.tracer import get_tracer

    tracer = get_tracer(question, len(history or []))
    model_str = _get_model_str()

    try:
        llm = get_llm()                      # 查询改写等辅助调用用默认模型
        gen_llm = get_generation_llm(deep_think)  # 最终生成：深度思考时可能切推理模型

        # ════════════════════════════════════════
        # 步骤1: 查询改写（Multi-Query）
        # ════════════════════════════════════════
        sub_queries: list[str] = [question]
        if settings.enable_query_rewrite:
            with tracer.step("query_rewrite", {
                "original_query": question,
                "mode": "multi_query",
                "n": settings.query_rewrite_n,
            }) as step:
                from app.rag.query_transform import multi_query_rewrite
                sub_queries = await multi_query_rewrite(
                    question, llm, settings.query_rewrite_n
                )
                step.set_outputs({
                    "sub_queries": sub_queries,
                    "count": len(sub_queries),
                })

        # ════════════════════════════════════════
        # 步骤2: 混合检索（向量 + BM25 + RRF）
        # ════════════════════════════════════════
        kb_results: list[tuple[Document, float]] = []
        candidate_k = settings.rerank_candidate_k

        with tracer.step("hybrid_retrieve", {
            "sub_queries": sub_queries,
            "candidate_k": candidate_k,
            "bm25_enabled": settings.enable_bm25,
        }) as step:

            if settings.enable_bm25:
                # ── 混合检索模式：对每个子查询做 向量+BM25+RRF ──
                from app.rag.hybrid_retriever import hybrid_search, reciprocal_rank_fusion

                all_fused: list[tuple[Document, float]] = []
                for sq in sub_queries:
                    fused = await hybrid_search(sq, candidate_k)
                    all_fused.extend(fused)

                # 多子查询结果合并后去重
                kb_results = _dedup_by_chunk_id(all_fused)
                # 按 RRF 分数降序
                kb_results.sort(key=lambda x: x[1], reverse=True)
                kb_results = kb_results[:candidate_k]
            else:
                # ── 纯向量检索模式（降级）──
                all_vec: list[tuple[Document, float]] = []
                for sq in sub_queries:
                    results = await search_with_scores(sq, candidate_k)
                    all_vec.extend(results)
                kb_results = _dedup_by_chunk_id(all_vec)
                kb_results.sort(key=lambda x: x[1], reverse=True)
                kb_results = kb_results[:candidate_k]

            max_score = kb_results[0][1] if kb_results else 0.0
            step.set_outputs({
                "hit_count": len(kb_results),
                "max_score": round(max_score, 4),
                "chunks": [
                    {
                        "source": d.metadata.get("source", "?"),
                        "score": round(s, 4),
                        "chunk_id": d.metadata.get("chunk_id", "?"),
                        "snippet": d.page_content[:200],
                    }
                    for d, s in kb_results[:10]  # trace 里只记前 10
                ],
            })

        # ════════════════════════════════════════
        # 步骤3: Cross-Encoder 重排
        # ════════════════════════════════════════
        if settings.enable_reranker and kb_results:
            with tracer.step("rerank", {
                "model": settings.reranker_model,
                "candidate_count": len(kb_results),
                "top_k": settings.reranker_top_k,
            }) as step:
                from app.rag.reranker import rerank
                candidate_docs = [doc for doc, _ in kb_results]
                reranked = rerank(question, candidate_docs, settings.reranker_top_k)
                # 用重排分数更新 kb_results
                kb_results = reranked
                max_score = kb_results[0][1] if kb_results else 0.0
                step.set_outputs({
                    "reranked_count": len(kb_results),
                    "top_score": round(max_score, 4),
                    "chunks": [
                        {
                            "source": d.metadata.get("source", "?"),
                            "score": round(s, 4),
                            "chunk_id": d.metadata.get("chunk_id", "?"),
                            "snippet": d.page_content[:200],
                        }
                        for d, s in kb_results
                    ],
                })

        # 提取 Document 并设置 metadata
        kb_docs: List[Document] = []
        for doc, score in kb_results:
            doc.metadata["type"] = "kb"
            doc.metadata["score"] = score
            kb_docs.append(doc)

        max_score = kb_results[0][1] if kb_results else 0.0

        # 推送 KB 来源卡片
        if kb_docs:
            yield ("sources", {"chunks": chunks_to_sources(kb_docs), "type": "kb"})

        # ════════════════════════════════════════
        # 步骤4: 判定是否网搜（手动开关 > 分数阈值 + 方差判定）
        #   web_search: False=仅本地 / True=强制网搜 / None=自动判定
        # ════════════════════════════════════════
        need_web = False
        web_reason = ""
        if settings.web_search_enabled and web_search is not False:
            if web_search is True:
                need_web = True
                web_reason = "用户手动开启智能搜索（强制网搜）"
            elif not kb_results:
                need_web = True
                web_reason = "无检索结果"
            elif max_score < settings.relevance_threshold:
                need_web = True
                web_reason = f"最高分 {max_score:.4f} < 阈值 {settings.relevance_threshold}"
            else:
                # 方差判定：top-k 分数方差过低说明没有突出命中
                scores = [s for _, s in kb_results]
                if len(scores) >= 3:
                    variance = statistics.pvariance(scores)
                    if variance < settings.web_search_variance_threshold:
                        need_web = True
                        web_reason = f"分数方差 {variance:.6f} < 阈值 {settings.web_search_variance_threshold}"

        web_docs: List[Document] = []
        used_web_search = False
        if need_web:
            yield ("status", {
                "stage": "web_search",
                "message": "本地知识库检索质量不足，正在联网搜索…",
            })
            with tracer.step("web_search", {
                "query": question,
                "provider": settings.web_search_provider,
                "max_results": settings.web_search_max_results,
                "reason": web_reason,
            }) as step:
                results = await search_web(question, settings.web_search_max_results)
                web_docs = _web_results_to_docs(results)
                step.set_outputs({
                    "hit_count": len(results),
                    "results": [
                        {
                            "title": r["title"],
                            "url": r["url"],
                            "snippet": r["snippet"][:200],
                        }
                        for r in results
                    ],
                })
                used_web_search = True

            if web_docs:
                yield ("sources", {"chunks": chunks_to_sources(web_docs), "type": "web"})

        # ════════════════════════════════════════
        # 步骤5: 合并上下文 + LLM 流式
        # ════════════════════════════════════════
        all_docs = kb_docs + web_docs
        if not all_docs:
            not_found_msg = "知识库中未找到相关信息。"
            yield ("token", not_found_msg)
            with tracer.step("reply", {
                "answer": not_found_msg,
                "used_web_search": used_web_search,
                "kb_sources": len(kb_docs),
                "web_sources": len(web_docs),
            }) as step:
                tracer.set_result(not_found_msg, used_web_search, model_str)
                step.set_outputs({
                    "trace_id": tracer.trace_id,
                    "answer_chars": len(not_found_msg),
                })
            yield ("done", {
                "trace_id": tracer.trace_id,
                "used_web_search": used_web_search,
            })
            return

        context_text = format_context(kb_docs, web_docs)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "【参考资料】\n{context}\n\n【用户问题】\n{question}"),
            ]
        )
        prompt_value = await prompt.ainvoke(
            {"context": context_text, "question": question}
        )

        answer_parts: list[str] = []
        with tracer.step("llm_stream", {
            "model": model_str,
            "context_chars": len(context_text),
            "prompt_messages": 2,
        }) as step:
            async for chunk in gen_llm.astream(prompt_value):
                # 思考过程（DeepSeek/LongCat 走 additional_kwargs.reasoning_content，
                # Claude 走 additional_kwargs.thinking）
                kwargs = getattr(chunk, "additional_kwargs", None) or {}
                reasoning = (
                    kwargs.get("reasoning_content")
                    or kwargs.get("thinking")
                    or kwargs.get("reasoning")
                )
                if reasoning:
                    yield ("reasoning", reasoning)

                # 正文 token
                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    text = "".join(
                        (c.get("text", "") if isinstance(c, dict) else str(c))
                        for c in content
                        if not isinstance(c, dict) or c.get("type") in ("text", "")
                    )
                else:
                    text = content
                if text:
                    answer_parts.append(text)
                    yield ("token", text)

            step.set_outputs({
                "answer_chars": len("".join(answer_parts)),
                "finish": "done",
            })

        # ════════════════════════════════════════
        # 步骤6: 最终回复
        # ════════════════════════════════════════
        answer = "".join(answer_parts)
        with tracer.step("reply", {
            "answer": answer,
            "used_web_search": used_web_search,
            "kb_sources": len(kb_docs),
            "web_sources": len(web_docs),
        }) as step:
            tracer.set_result(answer, used_web_search, model_str)
            step.set_outputs({
                "trace_id": tracer.trace_id,
                "answer_chars": len(answer),
            })
        yield ("done", {
            "trace_id": tracer.trace_id,
            "used_web_search": used_web_search,
        })

    except GeneratorExit:
        # 客户端断开连接 -- 记录后必须重新抛出（generator 协议要求）
        tracer.set_error("客户端断开连接")
        raise
    except Exception as e:
        tracer.set_error(str(e))
        yield ("error", {"message": f"处理失败: {e}"})
    finally:
        tracer.save()
