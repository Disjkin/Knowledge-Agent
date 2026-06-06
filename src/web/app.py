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

ENV_PATH = ROOT_DIR / ".env"


# ===== 持久化配置 =====

def read_env_vars() -> dict:
    """从 .env 文件读取配置值"""
    values = {"LLM_BASE_URL": "", "LLM_API_KEY": "", "LLM_MODEL": ""}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                if key in values:
                    values[key] = val.strip()
    return values


def save_env_var(key: str, value: str):
    """保存单个配置项到 .env 文件"""
    content = ""
    if ENV_PATH.exists():
        content = ENV_PATH.read_text(encoding="utf-8")
    lines = content.splitlines()

    found = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k, _, _ = stripped.partition("=")
            if k.strip() == key:
                lines[i] = f"{key}={value}"
                found = True
                break

    if not found:
        lines.append(f"{key}={value}")

    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ===== 资源缓存（避免每次交互重新加载）=====

def get_embedding_model(llm_base_url: str, llm_api_key: str):
    """获取嵌入模型（优先用独立配置，否则复用LLM配置）"""
    emb_base_url = config.get("embedding.base_url", "") or llm_base_url
    emb_api_key = config.get("embedding.api_key", "") or llm_api_key
    return EmbeddingModel(
        base_url=emb_base_url,
        api_key=emb_api_key,
        model=config.get("embedding.model", ""),
    )


@st.cache_resource(show_spinner=False)
def get_vector_store():
    """获取向量库（缓存）"""
    return VectorStore(
        persist_directory=config.get("vector_store.persist_directory"),
        collection_name=config.get("vector_store.collection_name"),
    )


# ===== Session State =====

def init_session_state():
    """初始化Streamlit session state"""
    env_vars = read_env_vars()

    defaults = {
        "agent": None,
        "messages": [],
        "indexed_count": 0,
        "llm_base_url": env_vars.get("LLM_BASE_URL", ""),
        "llm_api_key": env_vars.get("LLM_API_KEY", ""),
        "llm_model": env_vars.get("LLM_MODEL", ""),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def build_agent(base_url: str, api_key: str, model: str) -> Agent:
    """构建Agent实例"""
    embedding_model = get_embedding_model(base_url, api_key)
    vector_store = get_vector_store()
    retriever = Retriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
        top_k=config.get("retrieval.top_k"),
    )
    llm = LLMInterface(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=config.get("llm.temperature"),
        max_tokens=config.get("llm.max_tokens"),
        timeout=config.get("llm.timeout", 60),
    )
    return Agent(retriever=retriever, llm=llm)


def index_documents(uploaded_files, base_url: str, api_key: str) -> int:
    """将上传的文件建立索引，返回处理的文本块数量"""
    embedding_model = get_embedding_model(base_url, api_key)
    vector_store = get_vector_store()

    doc_dir = Path(config.get("documents.directory"))
    doc_dir.mkdir(parents=True, exist_ok=True)

    # 阶段1：加载文档
    doc_loader = DocumentLoader()
    splitter = TextSplitter(
        chunk_size=config.get("text_splitter.chunk_size"),
        chunk_overlap=config.get("text_splitter.chunk_overlap"),
    )

    all_chunks = []
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

        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    # 阶段2：向量化（带进度条）
    texts = [c.content for c in all_chunks]
    metadatas = [c.metadata for c in all_chunks]

    progress_bar = st.progress(0, text="🔢 正在向量化文本...")
    embeddings = embedding_model.embed(texts, show_progress=False)
    progress_bar.progress(60, text="📥 正在存入向量库...")

    # 阶段3：存储
    vector_store.add(texts=texts, embeddings=embeddings, metadatas=metadatas)
    progress_bar.progress(100, text="✅ 索引完成")
    progress_bar.empty()

    return len(all_chunks)


# ===== 主页面 =====

def main():
    st.set_page_config(
        page_title="Knowledge Agent",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 Knowledge Agent")
    st.caption("个人私有知识库智能助手 - 基于RAG的文档检索与问答")

    init_session_state()

    # ===== 侧边栏 =====
    with st.sidebar:
        st.header("⚙️ 配置")

        st.subheader("LLM 设置")
        base_url = st.text_input(
            "API地址",
            value=st.session_state.llm_base_url,
            placeholder="如 https://api.deepseek.com",
            key="input_base_url",
        )
        api_key = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.llm_api_key,
            key="input_api_key",
        )
        model = st.text_input(
            "模型名称 (可选)",
            value=st.session_state.llm_model,
            placeholder="留空则使用服务端默认",
            key="input_model",
        )

        if st.button("🔄 应用配置", use_container_width=True):
            if not base_url:
                st.error("请填写API地址")
            elif not api_key:
                st.error("请填写API Key")
            else:
                save_env_var("LLM_BASE_URL", base_url)
                save_env_var("LLM_API_KEY", api_key)
                save_env_var("LLM_MODEL", model)
                st.session_state.llm_base_url = base_url
                st.session_state.llm_api_key = api_key
                st.session_state.llm_model = model
                st.session_state.agent = None
                config.reload()
                st.success("✅ 配置已保存")

        st.divider()

        st.subheader("📁 文档管理")
        uploaded_files = st.file_uploader(
            "上传文档 (PDF/Word/TXT/Markdown)",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("📥 建立索引", use_container_width=True):
            try:
                chunks_count = index_documents(
                    uploaded_files,
                    st.session_state.llm_base_url,
                    st.session_state.llm_api_key,
                )
                if chunks_count > 0:
                    st.session_state.indexed_count += chunks_count
                    st.success(f"✅ 索引完成！处理 {chunks_count} 个文本块")
                else:
                    st.warning("⚠️ 未处理任何内容")
            except Exception as e:
                st.error(f"索引失败: {e}")

        st.divider()

        st.subheader("📊 状态")
        st.metric("已索引文本块", st.session_state.indexed_count)

        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.agent:
                st.session_state.agent.clear_memory()
            st.rerun()

    # ===== 主区域 - 对话 =====
    st.subheader("💬 对话")

    # 初始化Agent
    if st.session_state.agent is None:
        b_url = st.session_state.llm_base_url
        a_key = st.session_state.llm_api_key
        m_name = st.session_state.llm_model

        if not b_url or not a_key:
            st.info("👈 请先在左侧配置 API地址 和 API Key，然后点击「应用配置」")
            return

        try:
            st.session_state.agent = build_agent(b_url, a_key, m_name)
            st.success("✅ 模型就绪，可以开始对话")
        except Exception as e:
            st.error(f"初始化失败: {e}")
            return

    # 历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 来源"):
                    for src in msg["sources"]:
                        st.write(f"- {src['file_name']}")

    # 用户输入
    if question := st.chat_input("输入你的问题..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                # 检索
                with st.spinner("🔍 检索相关知识..."):
                    docs = st.session_state.agent.retriever.retrieve(question)
                    context = st.session_state.agent._build_context(docs)
                    history_text = st.session_state.agent.memory.get_history_text()
                    prompt = st.session_state.agent.prompt_template.format(
                        context=context, history=history_text, question=question,
                    )
                    messages = [{"role": "user", "content": prompt}]

                # 流式回答
                placeholder = st.empty()
                full_answer = ""
                for token in st.session_state.agent.llm.chat_stream(messages):
                    full_answer += token
                    placeholder.markdown(full_answer + "▌")

                placeholder.markdown(full_answer)

                # 保存记忆
                st.session_state.agent.memory.add(question, full_answer)
                sources = st.session_state.agent._extract_sources(docs)

                if sources:
                    with st.expander("📎 来源"):
                        for src in sources:
                            st.write(f"- {src['file_name']}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_answer,
                    "sources": sources,
                })

            except Exception as e:
                error_msg = f"❌ {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })


if __name__ == "__main__":
    main()
