"""配置管理模块 - 读取config.yaml和.env，提供全局配置"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent

# 默认配置
DEFAULT_CONFIG = {
    "embedding": {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "device": "cpu",
    },
    "vector_store": {
        "persist_directory": "./data/chroma",
        "collection_name": "knowledge",
    },
    "llm": {
        "provider": "openai",
        "model": "gpt-4",
        "api_key": "",
        "temperature": 0.7,
        "max_tokens": 2048,
    },
    "text_splitter": {
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
    "retrieval": {
        "top_k": 5,
    },
    "documents": {
        "directory": "./data/documents",
        "supported_formats": [".pdf", ".docx", ".txt", ".md"],
    },
}


class Config:
    """全局配置管理器"""

    _instance = None
    _config: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        """加载配置文件"""
        # 加载 .env
        env_path = ROOT_DIR / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        # 加载 config.yaml
        config_path = ROOT_DIR / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}
            self._config = self._merge(DEFAULT_CONFIG, file_config)
        else:
            self._config = DEFAULT_CONFIG.copy()

        # 替换环境变量引用 (${VAR_NAME} 格式)
        self._config = self._resolve_env_vars(self._config)

    def _merge(self, base: dict, override: dict) -> dict:
        """深度合并两个字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge(result[key], value)
            else:
                result[key] = value
        return result

    def _resolve_env_vars(self, obj: Any) -> Any:
        """递归替换 ${ENV_VAR} 格式的环境变量"""
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_key = obj[2:-1]
            return os.environ.get(env_key, "")
        elif isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        return obj

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的嵌套key，如 'llm.provider'"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def get_all(self) -> dict:
        """获取全部配置"""
        return self._config.copy()

    def reload(self):
        """重新加载配置"""
        self._load()


# 全局配置实例
config = Config()
