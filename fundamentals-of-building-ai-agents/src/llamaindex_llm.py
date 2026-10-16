import os
# Replace LangChain imports with native LlamaIndex Ollama integrations
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import httpx  # Make sure to import httpx

def simple_llm():
    # Define your Ollama-configured model name exactly as it's registered locally
    # Note: Ensure you have run `ollama run hf.co/Bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q4_K_M` first
    model_id = "llama3.1:8b"

    # 1. Manually create an HTTP client with your custom timeout
    custom_http_client = httpx.Client(
        timeout=httpx.Timeout(1800.0, connect=60.0)  # 30-minute generation limit
    )
    # Configure Ollama parameters
    # (temperature and token adjustments match your original configuration layout)
    ollama_llm = Ollama(
        #base_url="http://localhost:11434",  # Default local Ollama instance path
        model=model_id,
        #base_url=os.environ.get("OllAMA_HOST", "http://127.0.0.1:11434"),
        base_url="http://127.0.0.1:11434",
        custom_http_client=custom_http_client,
        #temperature=0.5,
        #num_predict=256,  # Equivalent to MAX_NEW_TOKENS
    #    additional_kwargs={"keep_alive": -1}  # Keeps the 32B model in VRAM indefinitely
    )
    return ollama_llm


def model_embedding():
    # 1. Manually create an HTTP client with your custom timeout
    custom_http_client = httpx.Client(
        timeout=httpx.Timeout(1800.0, connect=60.0)  # 30-minute generation limit
    )
    # Use the locally pulled Nomic embedding model
    local_emb = OllamaEmbedding(
        model_name="bge-m3",
        base_url="http://192.168.0.41:11434",
        custom_http_client=custom_http_client,
        #model="nomic-embed-text"
    )
    return local_emb
