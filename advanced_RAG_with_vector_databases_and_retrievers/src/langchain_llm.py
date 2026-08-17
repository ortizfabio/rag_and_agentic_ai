from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
import os


def llm():
    # Define your Ollama-configured model name exactly as it's registered locally
    # Note: Ensure you have run `ollama run hf.co/Bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q4_K_M` first
    model_id = "hf.co/Bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q4_K_M"

    # Configure Ollama parameters
    # (temperature and token adjustments match your original configuration layout)
    ollama_llm = Ollama(
        #base_url="http://localhost:11434",  # Default local Ollama instance path
        model=model_id,
        base_url=os.environ.get("Ollama_HOST", "http://192.168.0.41:11434"),
        temperature=0.5,
        num_predict=256,  # Equivalent to MAX_NEW_TOKENS
    )

    return ollama_llm

def model_embedding():
    # Use the locally pulled Nomic embedding model
    local_emb = OllamaEmbeddings(
        base_url=os.environ.get("Ollama_HOST", "http://192.168.0.41:11434"),
        model="bge-m3",
        #model="nomic-embed-text"
    )
    return local_emb