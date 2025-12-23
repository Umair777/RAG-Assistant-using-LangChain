"""
Configuration module for RAG Assistant
Loads and manages configuration from environment variables
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


@dataclass
class Config:
    """Configuration class for RAG Assistant"""
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Vector Store Configuration
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./data/chroma_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "rag_documents")
    
    # Chunking Configuration
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Model Configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "500"))
    
    def validate(self) -> bool:
        """Validate that required configuration is present"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required. Please set it in .env file")
        return True


def get_config() -> Config:
    """Get the configuration instance"""
    config = Config()
    config.validate()
    return config
