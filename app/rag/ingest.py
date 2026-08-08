"""文档入库 — 扫描 data/ → 加载 → 分块 → 嵌入 → 写 Chroma。
含 hash 判重：重复运行时未改动的文件会被跳过。
"""
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from app.config import settings

# PyMuPDF PDF loader（中文提取质量远高于 pypdf）
try:
    from langchain_community.document_loaders import PyMuPDFLoader
except ImportError:
    from langchain_community.document_loaders import PyPDFLoader as PyMuPDFLoader  # 兜底


# ── 文件扩展 → loader 映射 ──
LOADERS = {
    ".pdf": PyMuPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".markdown": UnstructuredMarkdownLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}

# 中文感知分块分隔符（从粗到细）
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", ".", "!", "?", ";", " ", ""]


def _file_hash(path: Path) -> str:
    """按内容 + mtime 算 hash，判断文件是否改动。"""
    h = hashlib.md5()
    stat = path.stat()
    h.update(f"{stat.st_size}:{stat.st_mtime}".encode())
    return h.hexdigest()


def scan_files(data_dir: Path) -> List[Path]:
    """返回 data/ 下所有被支持的文件路径。"""
    files: List[Path] = []
    for ext in LOADERS:
        files.extend(data_dir.rglob(f"*{ext}"))
    # 去重 + 排序，保证确定性
    return sorted(set(files))


def load_file(path: Path) -> List[Document]:
    """用对应 loader 加载单个文件。"""
    ext = path.suffix.lower()
    loader_cls = LOADERS.get(ext)
    if loader_cls is None:
        return []

    # TextLoader / Markdown 指定 utf-8，避免 Windows 中文乱码
    if ext in (".txt", ".md", ".markdown"):
        loader = loader_cls(str(path), encoding="utf-8")
    else:
        loader = loader_cls(str(path))

    docs = loader.load()
    # 注入 source metadata（存相对路径，方便 UI 展示）
    rel = path.relative_to(Path(settings.data_dir)).as_posix()
    for d in docs:
        d.metadata["source"] = rel
        d.metadata["file_name"] = path.name
    return docs


def split_docs(docs: List[Document]) -> List[Document]:
    """用中文感知分隔符分块。"""
    splitter = RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    # 给每个 chunk 打编号
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def _collection_has_data(db) -> bool:
    """检查 Chroma 集合是否已有数据。"""
    try:
        data = db.get()
        return len(data.get("ids", [])) > 0
    except Exception:
        return False


def _clear_collection(db):
    """清空 Chroma 集合中的所有数据（换嵌入模型时必须）。"""
    try:
        data = db.get()
        ids = data.get("ids", [])
        if ids:
            db.delete(ids=ids)
            print(f"[ingest] 已清除旧向量 {len(ids)} 条（嵌入模型变更）")
    except Exception as e:
        print(f"[ingest] 清除旧向量失败: {e}")


def ingest(clear: bool = False) -> Dict[str, object]:
    """全量入库。返回统计信息。

    Args:
        clear: True 时先清空集合全部向量再从头索引（用于删除文件后
            清理孤儿片段，或想干净重来）。默认 False 走增量 hash 判重。

    返回字典：
      files_total: 扫描到多少文件
      files_indexed: 实际入库多少（跳过未改的）
      chunks_generated: 总 chunk 数
      errors: 出错文件列表

    注意：使用 Chroma 原生 API（不用 langchain-chroma 封装），
    因为 langchain-chroma 的 add_documents 在 bge-m3 嵌入下会异常卡住。
    """
    import chromadb

    from app.llm.factory import get_embeddings

    data_dir = Path(settings.data_dir)
    data_dir.mkdir(exist_ok=True)

    files = scan_files(data_dir)
    embeddings = get_embeddings()

    # Chroma 原生客户端
    client = chromadb.PersistentClient(path=settings.persist_dir)
    col = client.get_or_create_collection(
        settings.collection_name, metadata={"hnsw:space": "cosine"}
    )

    if clear:
        # 清空后重建：删除集合中全部向量，再从头索引 data/
        try:
            count = col.count()
            if count > 0:
                col.delete(ids=col.get()["ids"])
                print(f"[ingest] 清空重建：已清除旧向量 {count} 条", flush=True)
        except Exception as e:
            print(f"[ingest] 清空失败: {e}", flush=True)
        existing_meta: Dict[str, str] = {}
    else:
        # 已入库文件的 hash（从现有 metadata 反查）
        existing_meta = _collect_source_hashes_native(col)

        # 如果集合非空但 hash 表为空（说明维度不兼容导致 get 异常或数据损坏），
        # 主动清除重建
        if not existing_meta:
            try:
                count = col.count()
                if count > 0:
                    col.delete(ids=col.get()["ids"])
                    print(f"[ingest] 已清除旧向量 {count} 条（嵌入模型变更）", flush=True)
            except Exception as e:
                print(f"[ingest] 清除旧向量失败: {e}", flush=True)

    stats = {
        "files_total": len(files),
        "files_indexed": 0,
        "chunks_generated": 0,
        "errors": [],
    }

    for f in files:
        try:
            fh = _file_hash(f)
            rel = f.relative_to(data_dir).as_posix()
            # 若 hash 与上次一致 → 跳过
            if existing_meta.get(rel) == fh:
                continue

            docs = load_file(f)
            if not docs:
                continue
            chunks = split_docs(docs)
            if not chunks:
                continue

            # 注入 file_hash，便于下次判重
            for c in chunks:
                c.metadata["file_hash"] = fh

            # 分批写入（每批 8 个 chunks：embedding + 原生 API add）
            BATCH = 8
            for i in range(0, len(chunks), BATCH):
                batch = chunks[i : i + BATCH]
                texts = [c.page_content for c in batch]
                vecs = embeddings.embed_documents(texts)
                col.add(
                    ids=[f"{rel}#{c.metadata.get('chunk_id', i + j)}" for j, c in enumerate(batch)],
                    embeddings=vecs,
                    documents=texts,
                    metadatas=[c.metadata for c in batch],
                )
                print(f"[ingest] {rel}: {min(i+BATCH, len(chunks))}/{len(chunks)} chunks written", flush=True)

            # 判重表也更新
            existing_meta[rel] = fh
            stats["files_indexed"] += 1
            stats["chunks_generated"] += len(chunks)

        except Exception as e:  # 单个文件出错不阻断整体
            stats["errors"].append({"file": str(f), "error": str(e)})

    return stats


def _collect_source_hashes_native(col) -> Dict[str, str]:
    """从现有 Chroma 记录收集 source → file_hash 映射（原生 API）。"""
    result: Dict[str, str] = {}
    try:
        data = col.get(include=["metadatas"])
        for meta in data.get("metadatas", []) or []:
            if not meta:
                continue
            src = meta.get("source")
            fh = meta.get("file_hash")
            if src and fh:
                result[src] = fh
    except Exception:
        pass
    return result


def reset_bm25_index():
    """入库后重置 BM25 索引缓存。"""
    try:
        from app.rag.hybrid_retriever import reset_bm25_index as _reset
        _reset()
    except ImportError:
        pass


def _collect_source_hashes(db) -> Dict[str, str]:
    """从现有 Chroma 记录收集 source → file_hash 映射。"""
    result: Dict[str, str] = {}
    try:
        data = db.get()  # {"ids", "metadatas", "documents"}
        for meta in data.get("metadatas", []) or []:
            if not meta:
                continue
            src = meta.get("source")
            fh = meta.get("file_hash")
            if src and fh:
                result[src] = fh
    except Exception:
        pass
    return result
