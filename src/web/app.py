"""Streamlit Web界面 - Knowledge Agent"""

import logging
import sys
from pathlib import Path

import streamlit as st

# 添加项目根目录到path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.core.config import config
from src.core.document_loader import DocumentLoader
from src.core.text_splitter import TextSplitter
from src.core.embeddings import EmbeddingModel
from src.core.vector_store import VectorStore
from src.core.retriever import Retriever
from src.core.llm import LLMInterface
from src.core.agent import Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_session_state():
    """初始化Streamlit session state"""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_count" not in st.session_state:
        st.session_state.indexed_count = 0


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


def index_documents(uploaded_files):
    """将上传的文件建立索引"""
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

    # 保存文件到本地
    doc_dir = Path(config.get("documents.directory"))
    doc_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    for uploaded_file in uploaded_files:
        # 保存文件
        file_path = doc_dir / uploaded_file.name
        file_path.write_bytes(uploaded_file.getbuffer())

        # 加载文档
        docs = doc_loader.load(str(file_path))
        if not docs:
            st.warning(f"⚠️ 无法加载文件: {uploaded_file.name}")
            continue

        # 分割文本
        chunks = splitter.split_documents(docs)
        if not chunks:
            st.warning(f"⚠️ 文件内容为空: {uploaded_file.name}")
            continue

        # 向量化并存储
        texts = [c.content for c in chunks]
        embeddings = embedding_model.embed(texts)
        metadatas = [c.metadata for c in chunks]
        vector_store.add(texts=texts, embeddings=embeddings, metadatas=metadatas)

        total_chunks += len(chunks)
        logger.info(f"已索引: {uploaded_file.name} ({len(chunks)} 个文本块)")

    return total_chunks


def main():
    st.set_page_config(
        page_title="Knowledge Agent",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 Knowledge Agent")
    st.caption("个人私有知识库智能助手 - 基于RAG的文档检索与问答")

    init_session_state()

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 配置")

        # LLM配置
        st.subheader("LLM模型")
        provider = st.selectbox(
            "Provider",
            LLMInterface.list_providers(),
            index=LLMInterface.list_providers().index(config.get("llm.provider", "openai")),
        )
        model = st.selectbox(
            "Model",
            LLMInterface.list_models(provider),
        )
        api_key = st.text_input("API Key", type="password", value=config.get("llm.api_key", ""))

        st.divider()

        # 文档上传
        st.subheader("📁 文档管理")
        uploaded_files = st.file_uploader(
            "上传文档",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("📥 建立索引", use_container_width=True):
            with st.spinner("正在处理文档..."):
                chunks_count = index_documents(uploaded_files)
                st.session_state.indexed_count += chunks_count
                st.success(f"✅ 索引完成！共处理 {chunks_count} 个文本块")

        st.divider()

        # 状态信息
        st.subheader("📊 状态")
        st.metric("已索引文本块", st.session_state.indexed_count)

        if st.button("🗑️ 清空对话历史", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.agent:
                st.session_state.agent.clear_memory()
            st.rerun()

    # 主区域 - 对话界面
    st.subheader("💬 对话")

    # 初始化Agent
    if st.session_state.agent is None:
        try:
            st.session_state.agent = build_agent()
        except Exception as e:
            st.error(f"Agent初始化失败: {e}")
            st.info("请确保已安装所有依赖并配置好API密钥")
            return

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 来源文档"):
                    for src in msg["sources"]:
                        st.write(f"- {src['file_name']} ({src['file_type']})")

    # 用户输入
    if question := st.chat_input("输入你的问题..."):
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Agent回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    result = st.session_state.agent.ask(question)
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("📎 来源文档"):
                            for src in sources:
                                st.write(f"- {src['file_name']} ({src['file_type']})")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    error_msg = f"回答出错: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


if __name__ == "__main__":
    main()
