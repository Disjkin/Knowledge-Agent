"""端到端测试：验证 6 项优化后的完整 RAG 流程。"""
import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")


async def main():
    from app.config import settings

    print(f"嵌入后端: {settings.embedding_backend} / {settings.local_embedding_model}")
    print(f"查询改写: {settings.enable_query_rewrite} (n={settings.query_rewrite_n})")
    print(f"BM25: {settings.enable_bm25}, Reranker: {settings.enable_reranker}")
    print(f"chunk_size: {settings.chunk_size}, overlap: {settings.chunk_overlap}")
    print(f"网搜: {settings.web_search_enabled}, 方差阈值: {settings.web_search_variance_threshold}")
    print()

    from app.rag.chain import stream_chat

    queries = [
        "格律诗音响公司是怎么成立的？",
        "丁元英是一个怎样的人？",
        "叶晓明和刘冰是什么关系？",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        print(f"{'='*60}")

        answer_parts = []
        kb_sources = []
        used_web = False
        async for event, data in stream_chat(query, []):
            if event == "token":
                answer_parts.append(data)
            elif event == "sources":
                if isinstance(data, dict):
                    for c in data.get("chunks", []):
                        kb_sources.append(
                            f"  [{c.get('chunk_id')}] score={c.get('score', 0):.4f} | {c.get('snippet', '')[:60]}..."
                        )
            elif event == "status":
                print(f"  状态: {data.get('message', '')}")
            elif event == "done":
                used_web = data.get("used_web_search", False)

        print(f"\n检索来源 ({len(kb_sources)} 条):")
        for s in kb_sources[:8]:
            print(s)
        if used_web:
            print("  (包含网络搜索)")
        print(f"\nA: {''.join(answer_parts)}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
