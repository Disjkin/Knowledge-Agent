# 🧠 Knowledge Agent

个人私有知识库智能助手 —— 基于 RAG（检索增强生成）技术的文档检索与智能问答系统。

## ✨ 功能特性

- **多格式文档支持** — 支持 PDF、Word (.docx)、TXT、Markdown 文件导入
- **语义向量检索** — 基于 Embedding + ChromaDB 的高效语义检索
- **智能问答** — 基于 RAG 技术，结合检索结果与 LLM 生成准确回答
- **多轮对话** — 支持上下文连续对话，自动维护对话历史
- **引用溯源** — 回答时标注信息来源文档
- **多模型兼容** — 支持 OpenAI、DeepSeek、Ollama 等任意兼容 OpenAI API 的模型
- **Web 界面** — 基于 Streamlit 的可视化交互界面
- **本地私有** — 所有数据本地存储，不上传云端

## 📁 项目结构

```
agent/
├── main.py                    # CLI 主入口
├── config.yaml                # 全局配置文件
├── .env.example               # 环境变量示例
├── requirements.txt           # Python 依赖
├── src/
│   ├── core/                  # 核心模块
│   │   ├── agent.py           # RAG Agent 编排
│   │   ├── config.py          # 配置管理
│   │   ├── document_loader.py # 文档加载器
│   │   ├── text_splitter.py   # 文本分块
│   │   ├── embeddings.py      # Embedding 模型
│   │   ├── vector_store.py    # 向量数据库 (ChromaDB)
│   │   ├── retriever.py       # 检索器
│   │   └── llm.py             # LLM 接口
│   └── web/
│       └── app.py             # Streamlit Web 界面
└── data/
    ├── chroma/                # 向量数据库存储
    └── documents/             # 文档存储目录
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Disjkin/Knowledge-Agent.git
cd Knowledge-Agent
```

### 2. 创建虚拟环境

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制示例配置文件并填入你的 API 信息：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# LLM 配置 - 任选一种
# OpenAI
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=gpt-4

# DeepSeek
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=deepseek-chat

# Ollama (本地部署，无需API Key)
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2
```

### 5. 启动使用

**Web 界面（推荐）：**

```bash
python main.py web
```

浏览器访问 `http://localhost:8501`

**命令行模式：**

```bash
# 索引文档
python main.py index ./data/documents

# 提问
python main.py ask "这个文档讲了什么？"

# 交互式对话
python main.py chat

# 查看配置状态
python main.py config
```

## 📦 依赖说明

| 依赖 | 用途 |
|------|------|
| `langchain` / `langchain-community` | LLM 编排框架 |
| `chromadb` | 本地向量数据库 |
| `openai` | OpenAI 兼容 API 客户端 |
| `sentence-transformers` | Embedding 模型支持 |
| `pypdf2` | PDF 文档解析 |
| `python-docx` | Word 文档解析 |
| `markdown` | Markdown 文档解析 |
| `streamlit` | Web 界面框架 |
| `pyyaml` | YAML 配置解析 |
| `python-dotenv` | 环境变量加载 |

## ⚙️ 配置说明

所有配置项在 [config.yaml](config.yaml) 中：

```yaml
embedding:              # Embedding 模型配置（留空则复用 LLM 配置）
vector_store:           # 向量数据库配置
  persist_directory: "./data/chroma"
  collection_name: "knowledge"
llm:                    # LLM 模型配置
  temperature: 0.7
  max_tokens: 2048
  timeout: 60
text_splitter:          # 文本分块配置
  chunk_size: 500
  chunk_overlap: 50
retrieval:              # 检索配置
  top_k: 5
```

## 📋 系统要求

- Python 3.9+
- 操作系统：Windows / macOS / Linux
- 任一 LLM API 服务（OpenAI、DeepSeek、Ollama 等）

## 📄 License

MIT
