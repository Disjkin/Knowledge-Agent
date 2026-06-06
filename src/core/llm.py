"""LLM统一接口 - 基于OpenAI兼容协议，支持流式输出"""

import logging
from typing import Generator, List, Optional

logger = logging.getLogger(__name__)


class LLMInterface:
    """LLM统一接口，使用OpenAI兼容协议"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 60,
    ):
        """
        初始化LLM接口

        Args:
            base_url: API调用地址，如 https://api.openai.com/v1
            api_key: API密钥
            model: 模型名称（可选，不填则由服务端决定）
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 请求超时秒数
        """
        if not base_url:
            raise ValueError("请在 .env 中配置 LLM_BASE_URL")
        if not api_key:
            raise ValueError("请在 .env 中配置 LLM_API_KEY")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _build_client(self):
        """创建OpenAI客户端"""
        from openai import OpenAI

        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=1,
        )

    def chat(self, messages: List[dict], **kwargs) -> str:
        """同步对话，返回完整回复"""
        client = self._build_client()

        params = {
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if self.model:
            params["model"] = self.model

        response = client.chat.completions.create(**params)
        return response.choices[0].message.content

    def chat_stream(self, messages: List[dict], **kwargs) -> Generator[str, None, None]:
        """流式对话，逐字返回回复"""
        client = self._build_client()

        params = {
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
        }
        if self.model:
            params["model"] = self.model

        response = client.chat.completions.create(**params)

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
