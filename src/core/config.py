"""配置管理模块 - 读取config.yaml和.env，提供全局配置"""

import os
from pathlib import Path
from typing import Any, List

import yaml
from dotenv import load_dotenv

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent

# 默认配置
DEFAULT_CONFIG = {
    "embedding": {
        "base_url": "",   # 嵌入API地址，留空则与LLM共用
        "api_key": "",    # 嵌入API密钥，留空则与LLM共用
        "model": "",      # 嵌入模型名，留空由服务端决定
    },
    "vector_store": {
        "persist_directory": "./data/chroma",
        "collection_name": "knowledge",
    },
    "llm": {
        "base_url": "",
        "api_key": "",
        "model": "",
        "temperature": 0.7,
        "max_tokens": 2048,
        "timeout": 60,
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
        """获取配置值，支持点号分隔的嵌套key，如 'llm.base_url'"""
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

    def validate_llm(self) -> List[str]:
        """验证LLM配置，返回缺失项列表"""
        errors = []
        if not self.get("llm.base_url"):
            errors.append("LLM_BASE_URL (API调用地址)")
        if not self.get("llm.api_key"):
            errors.append("LLM_API_KEY (API密钥)")
        return errors

    def print_status(self):
        """打印当前配置状态"""
        print("\n" + "=" * 50)
        print("📋 当前配置")
        print("=" * 50)

        # LLM
        base_url = self.get("llm.base_url", "")
        api_key = self.get("llm.api_key", "")
        model = self.get("llm.model", "")

        print(f"\n🤖 LLM:")
        print(f"   地址: {base_url or '❌ 未配置'}")
        print(f"   密钥: {'✅ 已配置' if api_key else '❌ 未配置'}")
        print(f"   模型: {model or '(服务端默认)'}")

        # 嵌入模型
        print(f"\n📐 嵌入模型:")
        print(f"   模型: {self.get('embedding.model_name')}")
        print(f"   设备: {self.get('embedding.device')}")

        # 向量库
        print(f"\n💾 向量库:")
        print(f"   路径: {self.get('vector_store.persist_directory')}")
        print(f"   集合: {self.get('vector_store.collection_name')}")

        # 验证
        errors = self.validate_llm()
        if errors:
            print(f"\n⚠️  缺少以下配置，请在 .env 文件中填写:")
            for err in errors:
                print(f"   - {err}")
        else:
            print(f"\n✅ 配置完整，可以启动")

        print("=" * 50 + "\n")


# 全局配置实例
config = Config()
