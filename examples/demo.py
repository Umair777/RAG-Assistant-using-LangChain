#!/usr/bin/env python3
"""
Example script demonstrating RAG Assistant usage
This example shows document ingestion, chunking, and querying
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_assistant.rag_assistant import RAGAssistant
from rag_assistant.config import Config


def main():
    """Demonstrate RAG Assistant functionality"""
    
    print("=" * 70)
    print("RAG Assistant Demo - Document Ingestion and Question Answering")
    print("=" * 70)
    
    # Note: This demo requires OPENAI_API_KEY to be set in .env file
    # For testing without API key, you can modify to use local embeddings
    
    try:
        # Initialize RAG Assistant
        print("\n1. Initializing RAG Assistant...")
        assistant = RAGAssistant()
        
        # Display system info
        print("\n2. System Configuration:")
        info = assistant.get_system_info()
        print(f"   - Chunking Strategy: {info['chunking']['strategy']}")
        print(f"   - Chunk Size: {info['chunking']['chunk_size']}")
        print(f"   - Chunk Overlap: {info['chunking']['chunk_overlap']}")
        print(f"   - LLM Model: {info['config']['llm_model']}")
        print(f"   - Embedding Model: {info['config']['embedding_model']}")
        
        # Example 1: Create sample documents for demonstration
        print("\n3. Creating sample documents for demonstration...")
        sample_docs_dir = Path(__file__).parent / "sample_data"
        sample_docs_dir.mkdir(exist_ok=True)
        
        # Create sample text file
        sample_txt = sample_docs_dir / "ai_overview.txt"
        with open(sample_txt, 'w') as f:
            f.write("""Artificial Intelligence Overview
            
Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, 
especially computer systems. These processes include learning, reasoning, and self-correction.

Machine Learning is a subset of AI that enables systems to learn and improve from experience 
without being explicitly programmed. It focuses on the development of computer programs that 
can access data and use it to learn for themselves.

Deep Learning is a subset of machine learning that uses neural networks with multiple layers. 
It has been particularly successful in image recognition, natural language processing, and 
speech recognition tasks.

Natural Language Processing (NLP) is a branch of AI that helps computers understand, 
interpret, and manipulate human language. NLP draws from many disciplines, including 
computer science and computational linguistics.

Retrieval-Augmented Generation (RAG) is a technique that enhances large language models 
by retrieving relevant information from external knowledge sources. This approach combines 
the benefits of retrieval-based and generation-based methods to provide more accurate and 
contextual responses.""")
        
        sample_txt2 = sample_docs_dir / "vector_databases.txt"
        with open(sample_txt2, 'w') as f:
            f.write("""Vector Databases for AI Applications

Vector databases are specialized databases designed to store and query high-dimensional 
vector embeddings. They are essential for modern AI applications, particularly in 
similarity search and recommendation systems.

ChromaDB is an open-source embedding database that makes it easy to build AI applications 
with embeddings. It's designed to be simple, fast, and scalable.

Vector embeddings are numerical representations of data that capture semantic meaning. 
They enable efficient similarity comparisons and are fundamental to many AI tasks.

Similarity search involves finding items that are most similar to a query item based on 
their vector representations. This is commonly used in recommendation systems, image search, 
and semantic search applications.

The quality of embeddings directly impacts the performance of RAG systems. Well-designed 
embeddings capture relevant semantic information and enable accurate retrieval.""")
        
        print(f"   ✓ Created sample documents in {sample_docs_dir}")
        
        # Example 2: Ingest documents
        print("\n4. Ingesting sample documents...")
        chunk_count = assistant.ingest_documents(str(sample_docs_dir), is_directory=True)
        print(f"   ✓ Ingested {chunk_count} chunks")
        
        # Example 3: Query the system
        print("\n5. Querying the RAG system...")
        
        questions = [
            "What is Retrieval-Augmented Generation?",
            "What are vector databases used for?",
            "Explain the difference between machine learning and deep learning"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n   Question {i}: {question}")
            result = assistant.query(question, k=2, show_sources=False)
            print(f"   Answer: {result['answer'][:200]}...")
        
        # Example 4: Demonstrate chunking strategy comparison
        print("\n6. Demonstrating different chunking strategies...")
        
        # Show current strategy info
        current_info = assistant.chunker.get_strategy_info()
        print(f"   Current strategy: {current_info['strategy']}")
        print(f"   Chunk size: {current_info['chunk_size']}")
        print(f"   Overlap: {current_info['chunk_overlap']} ({current_info['overlap_percentage']:.1f}%)")
        
        # Example 5: Show inference settings
        print("\n7. Inference Configuration:")
        inf_info = assistant.inference_engine.get_inference_info()
        print(f"   - Model: {inf_info['llm_model']}")
        print(f"   - Temperature: {inf_info['temperature']}")
        print(f"   - Max Tokens: {inf_info['max_tokens']}")
        
        print("\n" + "=" * 70)
        print("Demo completed successfully!")
        print("=" * 70)
        
        print("\n📝 Key Takeaways:")
        print("   1. Document Ingestion: Load PDFs, TXT, and DOCX files")
        print("   2. Chunking Strategies: Recursive, fixed-size, and token-based")
        print("   3. Vector Retrieval: Efficient similarity search using embeddings")
        print("   4. Inference Trade-offs: Balance between accuracy and speed")
        
        print("\n🔧 Next Steps:")
        print("   - Try the CLI: python src/cli.py --help")
        print("   - Ingest your own documents")
        print("   - Experiment with different chunking strategies")
        print("   - Adjust inference parameters (temperature, max_tokens)")
        
        # Cleanup (optional)
        # import shutil
        # shutil.rmtree(sample_docs_dir)
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease ensure you have:")
        print("   1. Created a .env file (copy from .env.example)")
        print("   2. Set your OPENAI_API_KEY in the .env file")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
