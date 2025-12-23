"""
RAG Assistant Main Module
Orchestrates the complete RAG pipeline
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from .config import Config, get_config
from .document_ingestion import DocumentIngestion
from .chunking import DocumentChunker, ChunkingStrategy
from .vector_store import VectorStoreManager
from .inference import RAGInference


class RAGAssistant:
    """
    Main RAG Assistant class that orchestrates the entire pipeline:
    - Document ingestion
    - Text chunking
    - Vector storage
    - Retrieval and inference
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize RAG Assistant
        
        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()
        
        # Initialize components
        self.document_ingestion = DocumentIngestion()
        self.chunker = DocumentChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            strategy=ChunkingStrategy.RECURSIVE
        )
        self.vector_store_manager = VectorStoreManager(
            persist_directory=self.config.vector_store_path,
            collection_name=self.config.collection_name,
            embedding_model=self.config.embedding_model,
            openai_api_key=self.config.openai_api_key
        )
        self.inference_engine = RAGInference(
            vector_store_manager=self.vector_store_manager,
            llm_model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            openai_api_key=self.config.openai_api_key
        )
        
        self.is_initialized = False
    
    def ingest_documents(
        self,
        source: str,
        is_directory: bool = False
    ) -> int:
        """
        Ingest documents from file or directory
        
        Args:
            source: Path to file or directory
            is_directory: Whether source is a directory
            
        Returns:
            Number of documents ingested
        """
        print(f"📥 Ingesting documents from: {source}")
        
        if is_directory:
            documents = self.document_ingestion.load_directory(source)
        else:
            documents = self.document_ingestion.load_document(source)
        
        print(f"✅ Loaded {len(documents)} documents")
        
        # Chunk documents
        print(f"✂️  Chunking documents (strategy: {self.chunker.strategy.value})")
        chunked_docs = self.chunker.chunk_documents(documents)
        print(f"✅ Created {len(chunked_docs)} chunks")
        
        # Create or update vector store
        if not self.is_initialized:
            print("🔨 Creating vector store...")
            self.vector_store_manager.create_vectorstore(chunked_docs)
            self.is_initialized = True
        else:
            print("➕ Adding to existing vector store...")
            self.vector_store_manager.add_documents(chunked_docs)
        
        print("✅ Vector store updated successfully")
        return len(chunked_docs)
    
    def load_existing_vectorstore(self):
        """Load an existing vector store from disk"""
        print("📂 Loading existing vector store...")
        self.vector_store_manager.load_vectorstore()
        self.is_initialized = True
        info = self.vector_store_manager.get_collection_info()
        print(f"✅ Loaded vector store with {info.get('document_count', 'unknown')} documents")
    
    def query(
        self,
        question: str,
        k: int = 4,
        show_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: Question to ask
            k: Number of documents to retrieve
            show_sources: Whether to display source information
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.is_initialized:
            raise ValueError("RAG Assistant not initialized. Please ingest documents or load existing vector store first.")
        
        print(f"\n🔍 Query: {question}")
        result = self.inference_engine.query_with_retrieval_details(question, k=k)
        
        print(f"\n💡 Answer: {result['answer']}")
        
        if show_sources and result.get('retrieval_details'):
            print(f"\n📚 Sources (top {k}):")
            for i, detail in enumerate(result['retrieval_details'], 1):
                score = detail['relevance_score']
                source = detail['metadata'].get('source', 'Unknown')
                print(f"  {i}. {source} (score: {score:.4f})")
        
        return result
    
    def change_chunking_strategy(
        self,
        strategy: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        Change the chunking strategy
        
        Args:
            strategy: Strategy name ('recursive', 'fixed_size', 'token_based')
            chunk_size: New chunk size (optional)
            chunk_overlap: New chunk overlap (optional)
        """
        if chunk_size:
            self.chunker.chunk_size = chunk_size
        if chunk_overlap:
            self.chunker.chunk_overlap = chunk_overlap
        
        strategy_enum = ChunkingStrategy(strategy)
        self.chunker.change_strategy(strategy_enum)
        print(f"✅ Changed chunking strategy to: {strategy}")
    
    def update_inference_settings(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Update inference parameters
        
        Args:
            temperature: New temperature value
            max_tokens: New max tokens value
        """
        self.inference_engine.update_inference_params(temperature, max_tokens)
        print("✅ Updated inference settings")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG system configuration
        
        Returns:
            Dictionary with system information
        """
        return {
            'initialized': self.is_initialized,
            'chunking': self.chunker.get_strategy_info(),
            'vector_store': self.vector_store_manager.get_collection_info(),
            'inference': self.inference_engine.get_inference_info(),
            'config': {
                'llm_model': self.config.llm_model,
                'embedding_model': self.config.embedding_model,
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap
            }
        }
    
    def reset_vectorstore(self):
        """Delete and reset the vector store"""
        print("🗑️  Resetting vector store...")
        self.vector_store_manager.delete_collection()
        self.is_initialized = False
        print("✅ Vector store reset")
