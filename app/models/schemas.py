"""Pydantic 请求/响应模型。"""
from typing import List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(..., description="user 或 assistant")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="当前用户问题")
    history: List[Message] = Field(default_factory=list, description="对话历史")
    web_search: Optional[bool] = Field(
        default=None,
        description="智能搜索：None=自动判定，True=强制网搜，False=仅本地知识库",
    )
    deep_think: bool = Field(
        default=False,
        description="深度思考：用推理模型生成（需配置 OPENAI_REASONING_MODEL，未配置则回落当前模型）",
    )


class SourceChunk(BaseModel):
    index: int
    source: str
    file_name: str
    chunk_id: int
    snippet: str


# SSE 事件体定义（前端据此解析）
class SourceEvent(BaseModel):
    chunks: List[SourceChunk]


class TokenEvent(BaseModel):
    delta: str


class DoneEvent(BaseModel):
    status: str = "done"


class ErrorEvent(BaseModel):
    message: str
