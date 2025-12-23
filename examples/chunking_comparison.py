#!/usr/bin/env python3
"""
Advanced example demonstrating different chunking strategies and their trade-offs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_assistant.chunking import DocumentChunker, ChunkingStrategy
from langchain_core.documents import Document


def demonstrate_chunking_strategies():
    """Demonstrate different chunking strategies and their characteristics"""
    
    print("=" * 80)
    print("Chunking Strategies Comparison")
    print("=" * 80)
    
    # Sample text
    sample_text = """
    Retrieval-Augmented Generation (RAG) is a powerful technique for enhancing large 
    language models. It combines the benefits of retrieval-based and generation-based 
    approaches to provide more accurate and contextual responses.
    
    The key components of a RAG system include:
    1. Document Ingestion - Loading and preprocessing documents
    2. Chunking - Splitting documents into manageable pieces
    3. Embedding - Converting text into vector representations
    4. Vector Storage - Storing embeddings in a database
    5. Retrieval - Finding relevant documents based on queries
    6. Generation - Using LLMs to generate responses
    
    Different chunking strategies offer various trade-offs:
    - Fixed-size chunking is simple but may split sentences awkwardly
    - Recursive chunking respects document structure and is more semantic
    - Token-based chunking ensures chunks fit within model token limits
    
    The choice of chunking strategy depends on your specific use case, document type, 
    and the downstream task you're optimizing for.
    """
    
    doc = Document(page_content=sample_text, metadata={"source": "example.txt"})
    
    # Test different strategies
    strategies = [
        (ChunkingStrategy.RECURSIVE, "Recursive Character Splitting"),
        (ChunkingStrategy.FIXED_SIZE, "Fixed-Size Character Splitting"),
    ]
    
    # Note: Token-based splitting requires internet access to download tokenizer
    # Uncomment the line below if you have internet access:
    # strategies.append((ChunkingStrategy.TOKEN_BASED, "Token-Based Splitting"))
    
    chunk_sizes = [200, 300, 500]
    
    for chunk_size in chunk_sizes:
        print(f"\n{'='*80}")
        print(f"Chunk Size: {chunk_size} | Overlap: 50")
        print(f"{'='*80}")
        
        for strategy, name in strategies:
            print(f"\n{name}:")
            print("-" * 80)
            
            chunker = DocumentChunker(
                chunk_size=chunk_size,
                chunk_overlap=50,
                strategy=strategy
            )
            
            chunks = chunker.chunk_documents([doc])
            
            print(f"Number of chunks: {len(chunks)}")
            print(f"\nFirst chunk preview:")
            print(f"{chunks[0].page_content[:150]}...")
            
            info = chunker.get_strategy_info()
            print(f"\nOverlap percentage: {info['overlap_percentage']:.1f}%")
    
    print("\n" + "=" * 80)
    print("Key Observations:")
    print("=" * 80)
    print("""
    1. RECURSIVE CHUNKING:
       - Respects paragraph and sentence boundaries
       - Best for maintaining semantic coherence
       - Recommended for most use cases
       
    2. FIXED-SIZE CHUNKING:
       - Simple and predictable chunk sizes
       - May split sentences awkwardly
       - Good for consistent chunk lengths
       
    3. TOKEN-BASED CHUNKING:
       - Ensures chunks fit within model token limits
       - Important for models with strict token constraints
       - Useful for accurate cost estimation
       
    Trade-offs to consider:
    - Smaller chunks: More precise retrieval, but may lose context
    - Larger chunks: Better context, but less precise matching
    - Higher overlap: Better context continuity, more storage
    - Lower overlap: Less redundancy, faster retrieval
    """)


if __name__ == "__main__":
    demonstrate_chunking_strategies()
