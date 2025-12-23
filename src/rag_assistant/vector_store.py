"""
Vector Store Module
Manages vector embeddings and similarity search using ChromaDB
"""

import os
from typing import List, Optional, Tuple
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


class VectorStoreManager:
    """Vector store manager for embeddings and retrieval"""
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "rag_documents",
        embedding_model: str = "text-embedding-ada-002",
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize vector store manager
        
        Args:
            persist_directory: Directory to persist vector database
            collection_name: Name of the collection
            embedding_model: OpenAI embedding model name
            openai_api_key: OpenAI API key
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key
        )
        
        # Initialize or load vector store
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create a new vector store from documents
        
        Args:
            documents: List of documents to embed
            
        Returns:
            Chroma vectorstore instance
        """
        if not documents:
            raise ValueError("Cannot create vector store from empty document list")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        return self.vectorstore
    
    def load_vectorstore(self) -> Chroma:
        """
        Load existing vector store from disk
        
        Returns:
            Chroma vectorstore instance
        """
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(
                f"Vector store directory not found: {self.persist_directory}"
            )
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        
        return self.vectorstore
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to existing vector store
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore or load_vectorstore first")
        
        ids = self.vectorstore.add_documents(documents)
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of most similar documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore or load_vectorstore first")
        
        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of tuples (document, score)
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore or load_vectorstore first")
        
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
        
        return results
    
    def as_retriever(self, **kwargs):
        """
        Get vector store as a retriever
        
        Args:
            **kwargs: Additional arguments for retriever configuration
            
        Returns:
            Retriever instance
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore or load_vectorstore first")
        
        return self.vectorstore.as_retriever(**kwargs)
    
    def delete_collection(self):
        """Delete the current collection"""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            self.vectorstore = None
    
    def get_collection_info(self) -> dict:
        """
        Get information about the vector store collection
        
        Returns:
            Dictionary with collection information
        """
        if self.vectorstore is None:
            return {
                'status': 'not_initialized',
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
        
        try:
            collection = self.vectorstore._collection
            return {
                'status': 'initialized',
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory,
                'document_count': collection.count(),
                'embedding_model': self.embedding_model
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
