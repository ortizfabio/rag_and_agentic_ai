from typing import Any

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models import BaseChatModel

EMBEDDING_MODEL = "bge-m3"
DEEPSEEK_R1_32B_MODEL = "hf.co/Bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q4_K_M"
URANUS_BASE_URL = "http://192.168.0.41:11434"
LOCAL_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL_PROVIDER = "ollama"


def get_llm(model: str | None = None,
        model_provider: str | None = None,
        config_prefix: str | None = None,
        **kwargs: Any,
        ) -> BaseChatModel | _ConfigurableModel:
    """This is a wrapper for langchain.chatmodels.init_chat_model to provide a consistent
    interface for initializing chat models with optional parameters."""
    return init_chat_model(model=model, model_provider=model_provider, config_prefix=config_prefix, **kwargs)
