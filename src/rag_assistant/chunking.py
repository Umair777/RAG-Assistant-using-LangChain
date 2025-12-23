"""
Chunking Strategies Module
Implements various text chunking strategies for document processing
"""

from typing import List, Optional
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain.schema import Document
from enum import Enum


class ChunkingStrategy(Enum):
    """Enumeration of available chunking strategies"""
    RECURSIVE = "recursive"
    FIXED_SIZE = "fixed_size"
    TOKEN_BASED = "token_based"


class DocumentChunker:
    """Document chunking class implementing various chunking strategies"""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    ):
        """
        Initialize document chunker
        
        Args:
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between consecutive chunks
            strategy: Chunking strategy to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self._splitter = self._create_splitter(strategy)
    
    def _create_splitter(self, strategy: ChunkingStrategy):
        """
        Create text splitter based on strategy
        
        Args:
            strategy: Chunking strategy to use
            
        Returns:
            Text splitter instance
        """
        if strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        elif strategy == ChunkingStrategy.FIXED_SIZE:
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator=" "
            )
        elif strategy == ChunkingStrategy.TOKEN_BASED:
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents using the configured strategy
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunked_docs = self._splitter.split_documents(documents)
        
        # Add metadata about chunking
        for i, doc in enumerate(chunked_docs):
            doc.metadata['chunk_id'] = i
            doc.metadata['chunk_strategy'] = self.strategy.value
            doc.metadata['chunk_size'] = self.chunk_size
            doc.metadata['chunk_overlap'] = self.chunk_overlap
        
        return chunked_docs
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk plain text using the configured strategy
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        return self._splitter.split_text(text)
    
    def change_strategy(self, strategy: ChunkingStrategy):
        """
        Change the chunking strategy
        
        Args:
            strategy: New chunking strategy
        """
        self.strategy = strategy
        self._splitter = self._create_splitter(strategy)
    
    def get_strategy_info(self) -> dict:
        """
        Get information about the current chunking configuration
        
        Returns:
            Dictionary with configuration details
        """
        return {
            'strategy': self.strategy.value,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'overlap_percentage': (self.chunk_overlap / self.chunk_size) * 100
        }


def create_chunker_with_strategy(
    strategy_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> DocumentChunker:
    """
    Factory function to create a chunker with a specific strategy
    
    Args:
        strategy_name: Name of the strategy ('recursive', 'fixed_size', 'token_based')
        chunk_size: Target size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        DocumentChunker instance
    """
    try:
        strategy = ChunkingStrategy(strategy_name)
    except ValueError:
        raise ValueError(
            f"Invalid strategy: {strategy_name}. "
            f"Valid options: {[s.value for s in ChunkingStrategy]}"
        )
    
    return DocumentChunker(chunk_size, chunk_overlap, strategy)
