#!/usr/bin/env python3
"""
CLI Interface for RAG Assistant
Command-line tool for interacting with the RAG system
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_assistant.rag_assistant import RAGAssistant
from rag_assistant.config import Config


def print_banner():
    """Print application banner"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        RAG Assistant - LangChain Edition                  ║
║        Retrieval-Augmented Generation System              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def ingest_command(args):
    """Handle document ingestion command"""
    assistant = RAGAssistant()
    
    source = args.source
    is_directory = Path(source).is_dir()
    
    try:
        chunk_count = assistant.ingest_documents(source, is_directory)
        print(f"\n✅ Successfully ingested and indexed {chunk_count} chunks")
    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        sys.exit(1)


def query_command(args):
    """Handle query command"""
    assistant = RAGAssistant()
    
    try:
        # Load existing vector store
        assistant.load_existing_vectorstore()
        
        # Process query
        result = assistant.query(
            question=args.question,
            k=args.top_k,
            show_sources=not args.no_sources
        )
        
        if args.verbose:
            print("\n" + "="*60)
            print("DETAILED RESULTS:")
            print("="*60)
            for i, detail in enumerate(result.get('retrieval_details', []), 1):
                print(f"\nSource {i}:")
                print(f"  Content: {detail['content'][:200]}...")
                print(f"  Score: {detail['relevance_score']:.4f}")
                print(f"  Metadata: {detail['metadata']}")
        
    except FileNotFoundError:
        print("\n❌ No vector store found. Please ingest documents first using 'ingest' command.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during query: {e}")
        sys.exit(1)


def interactive_command(args):
    """Handle interactive mode command"""
    assistant = RAGAssistant()
    
    try:
        # Load existing vector store
        assistant.load_existing_vectorstore()
        
        print("\n🤖 Entering interactive mode. Type 'exit' or 'quit' to end session.")
        print("Type 'help' for available commands.\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['exit', 'quit']:
                    print("👋 Goodbye!")
                    break
                
                if question.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  - Ask any question to get an answer")
                    print("  - 'info' - Show system information")
                    print("  - 'exit' or 'quit' - Exit interactive mode")
                    print()
                    continue
                
                if question.lower() == 'info':
                    info = assistant.get_system_info()
                    print("\nSystem Information:")
                    print(f"  Vector Store: {info['vector_store'].get('document_count', 'N/A')} documents")
                    print(f"  Chunking Strategy: {info['chunking']['strategy']}")
                    print(f"  Chunk Size: {info['chunking']['chunk_size']}")
                    print(f"  LLM Model: {info['config']['llm_model']}")
                    print(f"  Embedding Model: {info['config']['embedding_model']}")
                    print()
                    continue
                
                # Process query
                result = assistant.query(question, k=args.top_k, show_sources=True)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
        
    except FileNotFoundError:
        print("\n❌ No vector store found. Please ingest documents first using 'ingest' command.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def info_command(args):
    """Handle info command"""
    assistant = RAGAssistant()
    
    try:
        assistant.load_existing_vectorstore()
        info = assistant.get_system_info()
        
        print("\n" + "="*60)
        print("RAG ASSISTANT SYSTEM INFORMATION")
        print("="*60)
        
        print("\n📊 Vector Store:")
        vs_info = info['vector_store']
        print(f"  Status: {vs_info.get('status', 'N/A')}")
        print(f"  Documents: {vs_info.get('document_count', 'N/A')}")
        print(f"  Collection: {vs_info.get('collection_name', 'N/A')}")
        print(f"  Path: {vs_info.get('persist_directory', 'N/A')}")
        
        print("\n✂️  Chunking Configuration:")
        chunk_info = info['chunking']
        print(f"  Strategy: {chunk_info['strategy']}")
        print(f"  Chunk Size: {chunk_info['chunk_size']}")
        print(f"  Overlap: {chunk_info['chunk_overlap']} ({chunk_info['overlap_percentage']:.1f}%)")
        
        print("\n🤖 Inference Configuration:")
        inf_info = info['inference']
        print(f"  LLM Model: {inf_info['llm_model']}")
        print(f"  Temperature: {inf_info['temperature']}")
        print(f"  Max Tokens: {inf_info['max_tokens']}")
        
        print("\n🔧 Configuration:")
        cfg_info = info['config']
        print(f"  Embedding Model: {cfg_info['embedding_model']}")
        print(f"  LLM Model: {cfg_info['llm_model']}")
        
        print("\n" + "="*60 + "\n")
        
    except FileNotFoundError:
        print("\n❌ No vector store found.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="RAG Assistant - Retrieval-Augmented Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents into vector store')
    ingest_parser.add_argument('source', help='Path to document file or directory')
    ingest_parser.set_defaults(func=ingest_command)
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('-k', '--top-k', type=int, default=4, help='Number of documents to retrieve (default: 4)')
    query_parser.add_argument('--no-sources', action='store_true', help='Hide source information')
    query_parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed results')
    query_parser.set_defaults(func=query_command)
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive query session')
    interactive_parser.add_argument('-k', '--top-k', type=int, default=4, help='Number of documents to retrieve (default: 4)')
    interactive_parser.set_defaults(func=interactive_command)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    info_parser.set_defaults(func=info_command)
    
    args = parser.parse_args()
    
    if not args.command:
        print_banner()
        parser.print_help()
        sys.exit(0)
    
    print_banner()
    
    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
