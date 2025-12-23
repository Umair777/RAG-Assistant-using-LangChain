"""
Document Ingestion Module
Handles loading and processing of various document formats
"""

import os
from typing import List, Optional
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    DirectoryLoader
)
from langchain_core.documents import Document


class DocumentIngestion:
    """Document ingestion class for loading various document types"""
    
    SUPPORTED_FORMATS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader,
    }
    
    def __init__(self):
        """Initialize document ingestion"""
        pass
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document based on its file extension
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = path.suffix.lower()
        
        if file_extension not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {list(self.SUPPORTED_FORMATS.keys())}"
            )
        
        loader_class = self.SUPPORTED_FORMATS[file_extension]
        loader = loader_class(str(path))
        documents = loader.load()
        
        return documents
    
    def load_directory(
        self, 
        directory_path: str, 
        glob_pattern: str = "**/*.*",
        recursive: bool = True
    ) -> List[Document]:
        """
        Load all supported documents from a directory
        
        Args:
            directory_path: Path to the directory
            glob_pattern: Glob pattern for file matching
            recursive: Whether to search recursively
            
        Returns:
            List of Document objects
        """
        dir_path = Path(directory_path)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        all_documents = []
        
        # Load documents by extension
        for ext, loader_class in self.SUPPORTED_FORMATS.items():
            try:
                loader = DirectoryLoader(
                    str(dir_path),
                    glob=f"**/*{ext}" if recursive else f"*{ext}",
                    loader_cls=loader_class,
                    show_progress=True
                )
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                print(f"Warning: Error loading {ext} files: {e}")
                continue
        
        return all_documents
    
    def load_multiple_files(self, file_paths: List[str]) -> List[Document]:
        """
        Load multiple document files
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of Document objects
        """
        all_documents = []
        
        for file_path in file_paths:
            try:
                documents = self.load_document(file_path)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Warning: Error loading {file_path}: {e}")
                continue
        
        return all_documents
