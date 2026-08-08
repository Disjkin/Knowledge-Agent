"""重建知识库：删除旧 collection（512维 bge-small）-> 用 bge-m3 FP32/DML 重新入库 + 验证。

切换嵌入模型时维度会变（bge-small 512 -> bge-m3 1024），Chroma 集合维度是
首次写入时固定的，仅 delete 记录不会重置维度，所以必须整删 collection 重建。

Run: .venv/Scripts/python.exe reingest_bgem3.py
"""
import time


def main():
    import numpy as np
    import chromadb
    from app.config import settings
    from app.rag.ingest import ingest, reset_bm25_index
    from app.rag.retriever import get_vectorstore
    from app.llm.factory import get_embeddings

    # 1. 删除旧 collection（维度不兼容，必须整删重建）
    client = chromadb.PersistentClient(path=settings.persist_dir)
    try:
        client.delete_collection(settings.collection_name)
        print(f"[reingest] 已删除旧 collection: {settings.collection_name}")
    except Exception as e:
        print(f"[reingest] 删除 collection（可能不存在）: {e}")

    # 2. 预热嵌入模型（加载 FP32/DML session）
    print("[reingest] 加载嵌入模型 ...")
    t0 = time.perf_counter()
    emb = get_embeddings()
    print(f"[reingest] 嵌入模型就绪 ({time.perf_counter()-t0:.1f}s) type={type(emb).__name__}")

    # 3. 全量入库
    print("[reingest] 开始入库 ...")
    t0 = time.perf_counter()
    stats = ingest(clear=False)  # collection 已是空的，无需再 clear
    print(f"[reingest] 入库完成 ({time.perf_counter()-t0:.1f}s): {stats}")
    reset_bm25_index()

    # 4. 验证：嵌入测试 query + 检索
    q = "知识库检索测试"
    v = emb.embed_query(q)
    arr = np.asarray(v)
    print(f"[reingest] 验证: query嵌入 dim={arr.shape[0]} norm={np.linalg.norm(arr):.4f} "
          f"has_nan={bool(np.isnan(arr).any())}")

    col = get_vectorstore()
    print(f"[reingest] collection count={col.count()}")
    res = col.query(
        query_embeddings=[v], n_results=3,
        include=["documents", "distances", "metadatas"],
    )
    for i, (doc, dist, meta) in enumerate(zip(
        res["documents"][0], res["distances"][0], res["metadatas"][0]
    )):
        print(f"  hit{i}: score={1-dist:.4f} src={meta.get('source','?')}#{meta.get('chunk_id','?')} | {doc[:60]}")


if __name__ == "__main__":
    main()
