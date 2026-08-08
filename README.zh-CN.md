# 个人知识库助手

本地运行的 **RAG 知识库助手**：把文档（PDF / Markdown / DOCX / TXT）放进 `data/`，系统自动分块 + 向量化入库，通过 Web 聊天界面提问，从知识库检索相关片段后调用 LLM 推理回答。

- **多模型可配置**：通过 `.env` 自由切换 DeepSeek / OpenAI / 智谱 GLM / 通义千问 等 OpenAI 兼容模型，以及 Anthropic Claude
- **中文友好**：中文分块、BGE 中文嵌入模型、中英文自动切换回答
- **流式输出**：ChatGPT 风格 Web UI，逐字显示答案与参考来源卡片
- **本地运行**：向量库本地持久化，无需云账号
- **网搜兜底**：本地知识库未命中时自动联网搜索（Tavily / DuckDuckGo），UI 区分来源
- **链路追踪**：每次问答的完整调用链落盘保存（JSONL），支持 API 查询排查

## 功能特性

- 支持 PDF / Markdown / DOCX / TXT 四类文件（中文 + 英文）
- 智能分块（中文标点感知，不会从句中切断）
- 本地嵌入模型（`BAAI/bge-small-zh-v1.5`），无需联网即可检索
- 答案附引用来源（文件名 + 片段编号 + 原文片段预览）
- 重建索引时未改动文件自动跳过（hash 判重）
- 深色 / 浅色主题跟随系统

## 快速开始

### 1. 安装依赖

推荐用 **uv**（契合你现有工作流）：

```bash
cd /path/to/your/project
uv venv
uv pip install -r requirements.txt
```

或用 venv：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ `sentence-transformers` 会拉取 PyTorch（约 2GB），首次安装较慢。若体积敏感，可在 README 底部 **嵌入替代方案** 换用 `fastembed`（中文精度略低）。

### 2. 配置

```bash
cp .env.example .env
```

编辑 `.env`，**至少填一项** 模型提供商的 API key：

```env
# 默认 DeepSeek
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-你的key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat

# 切换 Claude：
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-你的key
# ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

### 3. 放入文档

把文档放进 `data/` 目录（支持子文件夹）：

```
data/
├── 产品手册.pdf
├── 会议纪要.md
├── 规范.docx
└── 笔记.txt
```

### 4. 启动

```bash
bash start.sh
```

或手动：

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

浏览器打开 **http://127.0.0.1:8000**

> 首次启动若 `data/` 有文档会自动索引；日志会显示 "已索引 N 个文档、M 个片段"。

### 5. 使用

- 在底部输入框打字提问，Enter 发送
- 答案流式出现，同时显示参考来源卡片
- 新增 / 改动文档后，点左侧「🔄 重建」重新索引
- 切换模型：改 `.env` 的 `LLM_PROVIDER` 后重启

## 配置项一览

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` / `PORT` | `127.0.0.1` / `8000` | 服务监听地址 |
| `LLM_PROVIDER` | `openai` | `openai` 或 `anthropic` |
| `OPENAI_API_KEY` | — | OpenAI 兼容接口 key（DeepSeek / OpenAI / 智谱 / 通义） |
| `OPENAI_BASE_URL` | `https://api.deepseek.com` | 接口地址 |
| `OPENAI_MODEL` | `deepseek-chat` | 模型名 |
| `ANTHROPIC_API_KEY` | — | Claude key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Claude 模型 |
| `EMBEDDING_PROVIDER` | `local` | `local`（本地 BGE）或 `openai` |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 本地嵌入模型 |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI 兼容嵌入模型名 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `600` / `80` | 分块大小 / 重叠 |
| `RETRIEVE_K` | `5` | 每次取 top-k 个片段 |
| `WEB_SEARCH_ENABLED` | `false` | 网搜兜底总开关 |
| `WEB_SEARCH_PROVIDER` | `tavily` | 网搜供应商：`tavily` 或 `duckduckgo` |
| `TAVILY_API_KEY` | - | Tavily API key（网搜供应商为 tavily 时必填） |
| `WEB_SEARCH_MAX_RESULTS` | `5` | 网搜最大返回条数 |
| `WEB_SEARCH_TIMEOUT` | `10` | 网搜超时秒数 |
| `RELEVANCE_THRESHOLD` | `0.3` | 本地检索最高分低于此值时触发网搜 |
| `TRACE_ENABLED` | `true` | 链路追踪开关 |
| `TRACE_DIR` | `logs/traces` | trace 日志目录 |
| `TRACE_MAX_FIELD_LEN` | `2000` | trace 中长文本字段截断长度 |

## 项目结构

```
个人知识库助手/
├── data/                   # 放入知识库文档
├── chroma_db/              # 向量库持久化（自动生成）
├── logs/traces/            # 链路追踪日志（自动生成，JSONL 按天滚动）
├── app/
│   ├── main.py             # FastAPI 后端
│   ├── config.py           # 读 .env
│   ├── config_admin.py     # 运行时模型设置
│   ├── llm/factory.py      # 多模型工厂
│   ├── rag/
│   │   ├── ingest.py       # 文档入库
│   │   ├── retriever.py    # 向量检索（带分数）
│   │   └── chain.py        # RAG 链 + 网搜兜底 + 流式
│   ├── tools/
│   │   └── web_search.py   # 网搜模块（Tavily / DuckDuckGo）
│   ├── tracing/
│   │   ├── tracer.py       # 链路追踪（JSONL 落盘）
│   │   └── reader.py       # trace 查询读取
│   ├── models/schemas.py   # Pydantic 模型
│   └── static/             # 前端
├── .env                    # 配置（git 忽略）
├── requirements.txt
└── start.sh / start.bat
```

## 嵌入替代方案

若不想拉 PyTorch（~2GB），可换 `fastembed`：

```bash
pip install fastembed  # 替代 sentence-transformers
```

并在 `.env`：

```env
LOCAL_EMBEDDING_MODEL=Qdrant/bge-small-en-v1.5
```

然后改 `app/llm/factory.py` 中 `get_embeddings()` 加载方式（把 `HuggingFaceEmbeddings` 换为 `FastEmbedEmbeddings`）。中文检索精度会略降，但安装体积大幅减小。

## 常见问题

**Q: 中文 PDF 提取乱码？**  
A: 本项目默认用 PyMuPDF，已针对 CJK 优化。若仍有问题请确保 PDF 内嵌字体。

**Q: 答案总是 "知识库中未找到相关信息"？**  
A: 检查 `.env` 嵌入模型是否正确加载；尝试重建索引；确认 `CHUNK_SIZE` 没有过小（600 对中文合适）。

**Q: 想接入国产模型（智谱 / 通义）？**  
A: 改 `.env` 的 `OPENAI_BASE_URL` + `OPENAI_MODEL` + key（LLM_PROVIDER 保持 `openai` 即可，它们都兼容 OpenAI Chat Completions 格式）。

## License

MIT
