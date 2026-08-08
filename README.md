# Knowledge-Agent · 本地 RAG 知识库助手

<div align="center">

**把文档丢进 `data/`，自动分块入库 → 混合检索 + 重排 → LLM 流式作答，本地知识库命中不足时自动联网兜底。**

FastAPI · LangChain · ChromaDB · BGE-M3 · SSE 流式 Web UI

</div>

---

## ✨ 特性

- **混合检索**：向量检索（BGE-M3）+ BM25 关键词检索，经 **RRF 融合**，兼顾语义与精确匹配
- **查询改写**：Multi-Query 多角度子查询改写 + HyDE 假设文档检索，提升召回率
- **Cross-Encoder 重排**：`bge-reranker-base` 对候选池精排，提升送入 LLM 的上下文质量
- **网搜兜底**：本地检索质量不足（分数阈值 / 方差判定）时自动联网，支持 **豆包 / Tavily / DuckDuckGo** 三供应商，UI 明确区分来源
- **多模型可配**：OpenAI 兼容接口（DeepSeek / 通义千问 / 智谱 / OpenAI）与 Anthropic Claude 自由切换，运行时改配置免重启
- **深度思考**：可切换推理模型（如 `deepseek-reasoner`），前端展示思考过程
- **流式输出**：ChatGPT 风格 Web UI，逐字显示答案 + 参考来源卡片
- **链路追踪**：每次问答的完整调用链落盘（JSONL 按天滚动），支持 API 查询排查
- **中文友好**：中文感知分块、中文嵌入模型、中英文自动跟随作答
- **本地运行**：向量库本地持久化，嵌入模型离线加载，无需云账号

## 🧠 工作流程

```
用户提问
   │
   ▼
[1] 查询改写   Multi-Query 改写为多个子查询（可选 HyDE）
   │
   ▼
[2] 混合检索   对每个子查询并行做 向量 + BM25 检索 → RRF 融合 → 去重
   │
   ▼
[3] 重排       Cross-Encoder 对候选池（默认 20 条）精排，取 Top-K
   │
   ▼
[4] 网搜判定   分数阈值 + 方差判定：本地质量不足时触发联网搜索兜底
   │
   ▼
[5] LLM 生成   拼接【本地资料 + 网络资料】上下文 → SSE 流式输出
   │
   ▼
[6] 链路追踪   全链路步骤（耗时 / 入参 / 结果）写入 JSONL，可 API 查询
```

## 🛠️ 技术栈

| 层 | 技术 |
|---|---|
| 后端 | FastAPI · Uvicorn · LangChain · LangChain-Chroma |
| 向量库 | ChromaDB（本地持久化，cosine 空间） |
| 嵌入 | BGE-M3（ONNX 加速 / INT8 量化）或 OpenAI 兼容嵌入 |
| 重排 | sentence-transformers CrossEncoder（bge-reranker-base） |
| LLM | OpenAI 兼容（DeepSeek/通义/智谱/OpenAI）· Anthropic Claude |
| 网搜 | 豆包搜索 · Tavily · DuckDuckGo |
| 前端 | 原生 HTML / CSS / JS + SSE 流式 |

## 🚀 快速开始

### 1. 安装依赖

推荐用 **uv**：

```bash
uv venv
uv pip install -r requirements.txt
```

或用 venv：

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

> ⚠️ `sentence-transformers` 会拉取 PyTorch（约 2GB），首次安装较慢。若安装 BGE-M3 的 ONNX 后端，模型会自动从 ModelScope / HuggingFace 缓存到本地，之后全程离线加载。

### 2. 配置

```bash
cp .env.example .env
```

编辑 `.env`，**至少填一项** LLM 提供商的 API key：

```env
# OpenAI 兼容接口（默认 DeepSeek）
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat

# 切换 Claude：
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-your-key
# ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

### 3. 放入文档

把文档放进 `data/` 目录（支持子文件夹，支持 PDF / Markdown / DOCX / TXT）：

```
data/
├── 产品手册.pdf
├── 会议纪要.md
├── 规范.docx
└── 笔记.txt
```

### 4. 启动

```bash
bash start.sh          # Linux / macOS / Git Bash
# 或 start.bat          # Windows
```

或手动：

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

浏览器打开 **http://127.0.0.1:8000**

> 首次启动若 `data/` 有文档会自动索引；日志会显示「已索引 N 个文档、M 个片段」。

### 5. 使用

- 在底部输入框提问，Enter 发送
- 答案流式出现，同时显示参考来源卡片（本地知识库 / 网络来源分区展示）
- 新增 / 改动文档后，点左侧「🔄 重建」增量索引（hash 判重，未改动文件自动跳过）
- 「⚙️ 模型设置」可运行时切换模型，改完即生效，无需重启
- 切换 LLM 提供商：改 `.env` 的 `LLM_PROVIDER` 后重启

## ⚙️ 配置项一览

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` / `PORT` | `127.0.0.1` / `8000` | 服务监听地址 |
| **LLM** | | |
| `LLM_PROVIDER` | `openai` | `openai` 或 `anthropic` |
| `OPENAI_API_KEY` | - | OpenAI 兼容接口 key |
| `OPENAI_BASE_URL` | `https://api.deepseek.com` | 接口地址 |
| `OPENAI_MODEL` | `deepseek-chat` | 模型名 |
| `OPENAI_REASONING_MODEL` | *(空)* | 深度思考用的推理模型，留空则回落 `OPENAI_MODEL` |
| `ANTHROPIC_API_KEY` | - | Claude key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Claude 模型 |
| **嵌入** | | |
| `EMBEDDING_PROVIDER` | `local` | `local`（本地 BGE）或 `openai` |
| `EMBEDDING_BACKEND` | `onnx` | `onnx`（CPU 快 5–10 倍）或 `pytorch` |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 本地嵌入模型 |
| `LOCAL_EMBEDDING_INT8` | `false` | INT8 量化（大模型 GPU 加速用） |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI 兼容嵌入模型名 |
| **RAG** | | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | 分块大小 / 重叠 |
| `RETRIEVE_K` | `5` | 最终返回片段数 |
| `ENABLE_QUERY_REWRITE` | `true` | Multi-Query 改写开关 |
| `QUERY_REWRITE_N` | `4` | 改写子查询数量 |
| `ENABLE_HYDE` | `false` | HyDE 假设文档检索 |
| `ENABLE_BM25` | `true` | BM25 关键词检索（混合检索） |
| `ENABLE_RERANKER` | `true` | Cross-Encoder 重排 |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | 重排模型 |
| `RERANK_TOP_K` | `5` | 重排后返回数量 |
| `RERANK_CANDIDATE_K` | `20` | 重排前候选池大小 |
| **网搜兜底** | | |
| `WEB_SEARCH_ENABLED` | `false` | 网搜兜底总开关 |
| `WEB_SEARCH_PROVIDER` | `doubao` | `doubao` / `tavily` / `duckduckgo` |
| `DOUBAO_SEARCH_API_KEY` | - | 豆包搜索 key |
| `TAVILY_API_KEY` | - | Tavily key |
| `WEB_SEARCH_MAX_RESULTS` | `5` | 网搜最大返回条数 |
| `WEB_SEARCH_TIMEOUT` | `15` | 网搜超时秒数 |
| `WEB_SEARCH_PROXY` | *(空)* | 网搜代理（境外服务用） |
| `RELEVANCE_THRESHOLD` | `0.3` | 本地最高分低于此值 → 触发网搜 |
| `WEB_SEARCH_VARIANCE_THRESHOLD` | `0.05` | top-k 分数方差低于此值 → 触发网搜 |
| **链路追踪** | | |
| `TRACE_ENABLED` | `true` | 链路追踪开关 |
| `TRACE_DIR` | `logs/traces` | trace 日志目录 |
| `TRACE_MAX_FIELD_LEN` | `2000` | trace 中长文本字段截断长度 |

## 📁 项目结构

```
.
├── data/                       # 放入知识库文档
├── chroma_db/                  # 向量库持久化（自动生成）
├── logs/traces/                # 链路追踪日志（JSONL 按天滚动）
├── models/                     # 本地嵌入/重排模型缓存
├── app/
│   ├── main.py                 # FastAPI 后端：路由 + SSE 流式 + 静态托管
│   ├── config.py               # pydantic-settings 读 .env
│   ├── config_admin.py         # 运行时模型设置
│   ├── llm/
│   │   ├── factory.py          # LLM + Embeddings 多模型工厂
│   │   └── onnx_embeddings.py  # BGE-M3 ONNX 嵌入后端
│   ├── rag/
│   │   ├── ingest.py           # 文档入库（hash 判重增量索引）
│   │   ├── retriever.py        # 向量检索（带分数）
│   │   ├── hybrid_retriever.py # BM25 + 向量 + RRF 融合
│   │   ├── query_transform.py  # Multi-Query / HyDE 查询改写
│   │   ├── reranker.py         # Cross-Encoder 重排
│   │   └── chain.py            # RAG 主链：改写→检索→重排→网搜→流式
│   ├── tools/
│   │   └── web_search.py       # 网搜模块（豆包/Tavily/DuckDuckGo）
│   ├── tracing/
│   │   ├── tracer.py           # 链路追踪（JSONL 落盘）
│   │   └── reader.py           # trace 查询读取
│   ├── models/schemas.py       # Pydantic 请求/响应模型
│   └── static/                 # 前端（HTML / CSS / JS）
├── docs/                       # 设计文档 + Trace 可视化
├── run_100_queries_v3.py       # 100 题批量评测脚本
├── .env.example
├── requirements.txt
└── start.sh / start.bat
```

## 🔌 API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/health` | 健康检查 + 当前模型 |
| `GET` | `/api/documents` | 已索引文档列表 |
| `POST` | `/api/ingest?clear=false` | 触发索引重建（`clear=true` 清空重来） |
| `POST` | `/api/chat` | SSE 流式聊天（`message` / `history` / `web_search` / `deep_think`） |
| `GET` | `/api/settings` | 读取运行时模型设置 |
| `POST` | `/api/settings` | 保存运行时模型设置（免重启） |
| `POST` | `/api/settings/test` | 测试模型连接 |
| `GET` | `/api/traces?limit=20` | trace 摘要列表（按时间倒序） |
| `GET` | `/api/traces/{trace_id}` | 单条 trace 完整 JSON |
| `POST` | `/api/open-data-folder` | 在文件管理器打开 `data/` |

### SSE 事件序列（`/api/chat`）

```
sources(kb) → [status → sources(web)] → reasoning × N → token × N → done
```

- `sources`：参考来源卡片（`type: kb` 本地 / `type: web` 网络）
- `status`：状态提示（如「正在联网搜索…」）
- `reasoning`：推理模型的思考过程
- `token`：正文增量
- `done`：`{ trace_id, used_web_search }`

## 📊 批量评测

仓库内置 100 题测试集 `run_100_queries_v3.py`：20% 命中本地知识库 + 80% 网搜兜底，覆盖时事实时、科技 AI、生活健康等多类问题，用 `ThreadPoolExecutor` 并发压测，结果输出 JSON / Excel。启动服务后运行：

```bash
python run_100_queries_v3.py
```

## ❓ 常见问题

**Q: 中文 PDF 提取乱码？**
A: 默认用 PyMuPDF，已针对 CJK 优化。若仍有问题请确保 PDF 内嵌字体。

**Q: 答案总是「知识库中未找到相关信息」？**
A: 检查 `.env` 嵌入模型是否正确加载；尝试「🗑️ 清空重建」；确认 `CHUNK_SIZE` 没有过小。

**Q: 想接入国产模型（智谱 / 通义）？**
A: 改 `.env` 的 `OPENAI_BASE_URL` + `OPENAI_MODEL` + key（`LLM_PROVIDER` 保持 `openai` 即可，它们都兼容 OpenAI Chat Completions 格式）。

**Q: 网搜在国内不可用？**
A: 默认供应商为豆包搜索（国内直连）；若用 Tavily / DuckDuckGo，可配置 `WEB_SEARCH_PROXY` 代理。

## 📄 License

MIT
