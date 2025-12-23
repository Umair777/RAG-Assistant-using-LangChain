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
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Model Configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    temperature: float = 0.7
    max_tokens: int = 500
    
    def __post_init__(self):
        """Post-initialization to handle type conversions with error handling"""
        # Parse chunk_size with error handling
        try:
            chunk_size_str = os.getenv("CHUNK_SIZE", "1000")
            self.chunk_size = int(chunk_size_str)
        except ValueError:
            raise ValueError(f"CHUNK_SIZE must be a valid integer, got: {chunk_size_str}")
        
        # Parse chunk_overlap with error handling
        try:
            chunk_overlap_str = os.getenv("CHUNK_OVERLAP", "200")
            self.chunk_overlap = int(chunk_overlap_str)
        except ValueError:
            raise ValueError(f"CHUNK_OVERLAP must be a valid integer, got: {chunk_overlap_str}")
        
        # Parse temperature with error handling
        try:
            temp_str = os.getenv("TEMPERATURE", "0.7")
            self.temperature = float(temp_str)
        except ValueError:
            raise ValueError(f"TEMPERATURE must be a valid float, got: {temp_str}")
        
        # Parse max_tokens with error handling
        try:
            max_tokens_str = os.getenv("MAX_TOKENS", "500")
            self.max_tokens = int(max_tokens_str)
        except ValueError:
            raise ValueError(f"MAX_TOKENS must be a valid integer, got: {max_tokens_str}")
    
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
