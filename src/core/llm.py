"""LLM统一接口 - 支持OpenAI、Anthropic、Ollama等多种Provider"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class LLMInterface:
    """LLM统一接口，基于LiteLLM实现多模型支持"""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: List[dict], **kwargs) -> str:
        """对话接口"""
        try:
            import litellm

            response = litellm.completion(
                model=self._get_model_id(),
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                api_key=self.api_key,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise

    def _get_model_id(self) -> str:
        """根据provider返回对应的模型ID"""
        model_map = {
            "openai": self.model,  # gpt-4, gpt-3.5-turbo
            "anthropic": f"anthropic/{self.model}",  # claude-3-opus 等
            "ollama": f"ollama/{self.model}",  # llama2, qwen 等
        }
        return model_map.get(self.provider, self.model)

    @staticmethod
    def list_providers() -> List[str]:
        """支持的Provider列表"""
        return ["openai", "anthropic", "ollama"]

    @staticmethod
    def list_models(provider: str) -> List[str]:
        """某个Provider支持的模型列表"""
        models = {
            "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            "ollama": ["llama2", "qwen", "mistral", "codellama"],
        }
        return models.get(provider, [])
