"""文本分割器 - 将长文档分割为适合向量化的小块"""

import logging
from typing import List

from .document_loader import Document

logger = logging.getLogger(__name__)


class TextSplitter:
    """文本分割器，支持按段落、句子、字符数三级分割策略"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将Document列表分割为更小的块"""
        result = []
        for doc in documents:
            chunks = self.split(doc.content)
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata["chunk_index"] = i
                result.append(Document(content=chunk, metadata=metadata))
        logger.info(f"分割完成: {len(documents)} 个文档 → {len(result)} 个文本块")
        return result

    def split(self, text: str) -> List[str]:
        """分割文本为chunks，优先按段落，其次按句子，最后按字符"""
        if len(text) <= self.chunk_size:
            return [text]

        # 第一步：按段落分割
        paragraphs = self._split_by_paragraph(text)

        # 第二步：对过长的段落按句子分割
        sentences = []
        for para in paragraphs:
            if len(para) > self.chunk_size:
                sentences.extend(self._split_by_sentence(para))
            else:
                sentences.append(para)

        # 第三步：合并小块，确保不超过chunk_size
        chunks = self._merge_chunks(sentences)

        return chunks

    def _split_by_paragraph(self, text: str) -> List[str]:
        """按段落分割（双换行符）"""
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_by_sentence(self, text: str) -> List[str]:
        """按句子分割"""
        import re
        # 中英文句子分隔符
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def _merge_chunks(self, pieces: List[str]) -> List[str]:
        """将小块合并为不超过chunk_size的大块，支持overlap"""
        if not pieces:
            return []

        chunks = []
        current_chunk = ""

        for piece in pieces:
            # 如果当前块加上新块不超限，合并
            if len(current_chunk) + len(piece) + 1 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n{piece}" if current_chunk else piece
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk)
                # 如果单个piece就超限，强制按字符切割
                if len(piece) > self.chunk_size:
                    sub_chunks = self._split_by_chars(piece)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    # 保留overlap部分
                    if self.chunk_overlap > 0 and chunks:
                        overlap_text = current_chunk[-self.chunk_overlap:] if current_chunk else ""
                        current_chunk = f"{overlap_text}\n{piece}"
                    else:
                        current_chunk = piece

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_by_chars(self, text: str) -> List[str]:
        """按字符数强制分割（兜底策略）"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        return chunks
