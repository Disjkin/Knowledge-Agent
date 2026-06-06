#!/usr/bin/env python3
"""Knowledge Agent - 主入口"""

import argparse
import logging
import sys
from pathlib import Path

# 确保项目根目录在path中
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from src.core.config import config
from src.core.document_loader import DocumentLoader
from src.core.text_splitter import TextSplitter
from src.core.embeddings import EmbeddingModel
from src.core.vector_store import VectorStore
from src.core.retriever import Retriever
from src.core.llm import LLMInterface
from src.core.agent import Agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_agent() -> Agent:
    """构建Agent实例"""
    embedding_model = EmbeddingModel(
        model_name=config.get("embedding.model_name"),
        device=config.get("embedding.device"),
    )
    vector_store = VectorStore(
        persist_directory=config.get("vector_store.persist_directory"),
        collection_name=config.get("vector_store.collection_name"),
    )
    retriever = Retriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
        top_k=config.get("retrieval.top_k"),
    )
    llm = LLMInterface(
        provider=config.get("llm.provider"),
        model=config.get("llm.model"),
        api_key=config.get("llm.api_key"),
        temperature=config.get("llm.temperature"),
        max_tokens=config.get("llm.max_tokens"),
    )
    return Agent(retriever=retriever, llm=llm)


def cmd_index(args):
    """索引文档"""
    doc_loader = DocumentLoader()
    splitter = TextSplitter(
        chunk_size=config.get("text_splitter.chunk_size"),
        chunk_overlap=config.get("text_splitter.chunk_overlap"),
    )
    embedding_model = EmbeddingModel(
        model_name=config.get("embedding.model_name"),
        device=config.get("embedding.device"),
    )
    vector_store = VectorStore(
        persist_directory=config.get("vector_store.persist_directory"),
        collection_name=config.get("vector_store.collection_name"),
    )

    path = Path(args.path)
    if path.is_file():
        docs = doc_loader.load(str(path))
    elif path.is_dir():
        docs = doc_loader.load_directory(str(path))
    else:
        logger.error(f"路径不存在: {args.path}")
        return

    if not docs:
        logger.warning("未加载到任何文档")
        return

    chunks = splitter.split_documents(docs)
    logger.info(f"文档分割完成: {len(docs)} 个文档 → {len(chunks)} 个文本块")

    texts = [c.content for c in chunks]
    embeddings = embedding_model.embed(texts)
    metadatas = [c.metadata for c in chunks]
    vector_store.add(texts=texts, embeddings=embeddings, metadatas=metadatas)

    logger.info(f"✅ 索引完成！当前向量总数: {vector_store.count()}")


def cmd_ask(args):
    """提问"""
    agent = build_agent()
    result = agent.ask(args.question)
    print(f"\n📋 回答:\n{result['answer']}")
    if result["sources"]:
        print(f"\n📎 来源:")
        for src in result["sources"]:
            print(f"  - {src['file_name']} ({src['file_type']})")


def cmd_chat(args):
    """交互式对话"""
    agent = build_agent()
    print("🧠 Knowledge Agent 交互模式 (输入 'quit' 退出, 'clear' 清空历史)")
    print("-" * 50)

    while True:
        try:
            question = input("\n👤 你: ").strip()
            if not question:
                continue
            if question.lower() == "quit":
                print("👋 再见！")
                break
            if question.lower() == "clear":
                agent.clear_memory()
                print("🗑️ 对话历史已清空")
                continue

            result = agent.ask(question)
            print(f"\n🧠 助手: {result['answer']}")
            if result["sources"]:
                print("📎 来源:", ", ".join(s["file_name"] for s in result["sources"]))

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


def cmd_web(args):
    """启动Web界面"""
    import subprocess
    web_app = ROOT_DIR / "src" / "web" / "app.py"
    port = args.port or 8501
    logger.info(f"启动Web界面: http://localhost:{port}")
    subprocess.run(["streamlit", "run", str(web_app), "--server.port", str(port)])


def main():
    parser = argparse.ArgumentParser(description="Knowledge Agent - 私有文件检索智能助手")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # index 命令
    index_parser = subparsers.add_parser("index", help="索引文档")
    index_parser.add_argument("path", help="文件或目录路径")
    index_parser.set_defaults(func=cmd_index)

    # ask 命令
    ask_parser = subparsers.add_parser("ask", help="提问")
    ask_parser.add_argument("question", help="问题内容")
    ask_parser.set_defaults(func=cmd_ask)

    # chat 命令
    chat_parser = subparsers.add_parser("chat", help="交互式对话")
    chat_parser.set_defaults(func=cmd_chat)

    # web 命令
    web_parser = subparsers.add_parser("web", help="启动Web界面")
    web_parser.add_argument("--port", type=int, default=8501, help="端口号")
    web_parser.set_defaults(func=cmd_web)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
