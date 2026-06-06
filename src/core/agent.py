"""Agent核心 - 编排整个RAG流程"""

import logging
from typing import List, Optional

from .document_loader import Document
from .embeddings import EmbeddingModel
from .llm import LLMInterface
from .retriever import Retriever
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# RAG提示词模板
RAG_PROMPT_TEMPLATE = """你是一个知识库助手。根据以下检索到的文档内容回答用户的问题。

要求：
1. 基于提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，明确告知用户
3. 回答要准确、简洁、有条理
4. 在回答末尾标注信息来源

检索到的文档内容：
{context}

对话历史：
{history}

用户问题：{question}

请回答："""


class ConversationMemory:
    """对话记忆"""

    def __init__(self, max_history: int = 10):
        self.history: List[dict] = []
        self.max_history = max_history

    def add(self, question: str, answer: str):
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_history_text(self) -> str:
        if not self.history:
            return "无"
        lines = []
        for h in self.history:
            lines.append(f"用户: {h['question']}")
            lines.append(f"助手: {h['answer']}")
        return "\n".join(lines)

    def clear(self):
        self.history.clear()


class Agent:
    """RAG Agent核心"""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMInterface,
        prompt_template: str = RAG_PROMPT_TEMPLATE,
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template
        self.memory = ConversationMemory()

    def ask(self, question: str, with_history: bool = True) -> dict:
        """
        回答问题

        Returns:
            {
                "answer": str,       # 回答内容
                "sources": list,     # 来源文档列表
            }
        """
        logger.info(f"收到问题: {question}")

        # 1. 检索相关文档
        docs = self.retriever.retrieve(question)
        logger.info(f"检索到 {len(docs)} 条相关文档")

        # 2. 组装上下文
        context = self._build_context(docs)

        # 3. 构建提示词
        history_text = self.memory.get_history_text() if with_history else "无"
        prompt = self.prompt_template.format(
            context=context,
            history=history_text,
            question=question,
        )

        # 4. 调用LLM
        messages = [{"role": "user", "content": prompt}]
        answer = self.llm.chat(messages)

        # 5. 保存对话历史
        self.memory.add(question, answer)

        # 6. 提取来源信息
        sources = self._extract_sources(docs)

        return {
            "answer": answer,
            "sources": sources,
        }

    def _build_context(self, docs: List[Document]) -> str:
        """组装检索上下文"""
        if not docs:
            return "未找到相关文档。"

        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", "未知文件")
            score = doc.metadata.get("relevance_score", 0)
            context_parts.append(
                f"[文档{i}] 来源: {source} (相关度: {score:.2f})\n{doc.content}"
            )
        return "\n\n---\n\n".join(context_parts)

    def _extract_sources(self, docs: List[Document]) -> List[dict]:
        """提取来源信息"""
        sources = []
        seen = set()
        for doc in docs:
            source = doc.metadata.get("source", "")
            if source and source not in seen:
                seen.add(source)
                sources.append({
                    "file_name": doc.metadata.get("file_name", ""),
                    "file_type": doc.metadata.get("file_type", ""),
                    "source": source,
                })
        return sources

    def clear_memory(self):
        """清空对话历史"""
        self.memory.clear()
