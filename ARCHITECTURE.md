# RAG System Architecture and Design Decisions

## System Architecture

### High-Level Overview

The RAG Assistant is built on a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Assistant System                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐      ┌────────────────┐                    │
│  │   CLI Layer    │      │   Python API   │                    │
│  └────────┬───────┘      └────────┬───────┘                    │
│           │                       │                             │
│           └───────────┬───────────┘                             │
│                       │                                         │
│           ┌───────────▼────────────┐                           │
│           │  RAGAssistant (Main)   │                           │
│           └───────────┬────────────┘                           │
│                       │                                         │
│    ┌──────────────────┼──────────────────┐                     │
│    │                  │                  │                     │
│ ┌──▼─────────┐  ┌────▼────────┐  ┌─────▼──────┐              │
│ │Document    │  │  Chunking   │  │Vector Store│              │
│ │Ingestion   │  │  Strategies │  │  Manager   │              │
│ └──┬─────────┘  └────┬────────┘  └─────┬──────┘              │
│    │                  │                  │                     │
│    └──────────────────┼──────────────────┘                     │
│                       │                                         │
│                ┌──────▼───────┐                                │
│                │  Inference   │                                │
│                │   Engine     │                                │
│                └──────────────┘                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

External Dependencies:
┌────────────┐  ┌──────────┐  ┌──────────┐
│  ChromaDB  │  │  OpenAI  │  │LangChain │
└────────────┘  └──────────┘  └──────────┘
```

## Core Components

### 1. Document Ingestion (`document_ingestion.py`)

**Purpose**: Load and preprocess documents from various sources

**Supported Formats**:
- PDF (via PyPDFLoader)
- TXT (via TextLoader)
- DOCX (via Docx2txtLoader)

**Design Decisions**:
- **Extensibility**: Easy to add new loaders via dictionary mapping
- **Error Handling**: Graceful failures for individual files in batch operations
- **Metadata Preservation**: Maintains source information for attribution

**Key Methods**:
```python
load_document(file_path)        # Single file
load_directory(directory_path)  # Batch processing
load_multiple_files(file_paths) # Custom file list
```

### 2. Chunking Strategies (`chunking.py`)

**Purpose**: Split documents into optimal-sized chunks for retrieval

**Available Strategies**:

1. **Recursive Character Splitting** (Default)
   - Uses hierarchical separators: `["\n\n", "\n", " ", ""]`
   - Respects document structure
   - Best semantic preservation
   
2. **Fixed-Size Character Splitting**
   - Simple space-based splitting
   - Predictable chunk sizes
   - May break semantic units
   
3. **Token-Based Splitting**
   - Counts actual tokens
   - Ensures model compatibility
   - Important for cost control

**Design Decisions**:
- **Strategy Pattern**: Easy to switch between strategies
- **Metadata Enrichment**: Tracks chunking parameters for debugging
- **Configuration**: Chunk size and overlap are tunable
- **Factory Function**: Convenient creation via string names

**Chunk Size Recommendations**:
| Use Case | Size | Overlap | Rationale |
|----------|------|---------|-----------|
| Technical Docs | 800-1000 | 150-200 | Code snippets need context |
| Legal Documents | 1200-1500 | 200-300 | Complex sentences, definitions |
| General Text | 600-800 | 100-150 | Balanced approach |
| Chat Logs | 400-600 | 50-100 | Short, focused exchanges |

### 3. Vector Store Manager (`vector_store.py`)

**Purpose**: Manage embeddings and similarity search

**Technology Stack**:
- **Vector DB**: ChromaDB (embedded, persistent)
- **Embeddings**: OpenAI text-embedding-ada-002 (1536 dimensions)

**Design Decisions**:
- **Persistence**: Automatic saving to disk
- **Flexibility**: Can create new or load existing stores
- **Retrieval Modes**: 
  - `similarity_search()`: Basic retrieval
  - `similarity_search_with_score()`: With relevance scores
  - `as_retriever()`: LangChain integration

**Why ChromaDB**:
- ✓ Embedded (no server required)
- ✓ Persistent storage
- ✓ Good performance for small-medium datasets
- ✓ Easy to use
- ✓ Open source

**Alternatives Considered**:
- Pinecone: Cloud-only, paid
- Weaviate: More complex setup
- FAISS: No built-in persistence
- Qdrant: More heavyweight

### 4. Inference Engine (`inference.py`)

**Purpose**: Orchestrate retrieval and generation

**Chain Types Supported**:
- **Stuff**: Concatenate all retrieved docs (default)
- **Map Reduce**: Process docs separately, then combine
- **Refine**: Iteratively refine answer
- **Map Rerank**: Rank and select best answer

**Design Decisions**:
- **Prompt Templates**: Customizable for domain adaptation
- **Source Tracking**: Returns source documents for attribution
- **Batch Processing**: Efficient multi-query handling
- **Parameter Tuning**: Runtime adjustment of temperature/tokens

**Default Prompt**:
```
You are an AI assistant helping users find information from documents.
Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know.

Context: {context}
Question: {question}
Answer:
```

### 5. RAG Assistant (`rag_assistant.py`)

**Purpose**: Main orchestrator tying all components together

**Workflow**:
```
1. Initialize components with config
2. Ingest documents → chunk → embed → store
3. Query: retrieve → augment prompt → generate
4. Return answer + sources
```

**Design Decisions**:
- **Lazy Initialization**: Vector store loaded on demand
- **State Management**: Tracks initialization status
- **Configuration**: Centralized via Config class
- **User Feedback**: Emoji-based progress indicators

## Configuration Management

### Environment Variables (`.env`)

```bash
# Critical: API Access
OPENAI_API_KEY=sk-...

# Storage: Where vectors are persisted
VECTOR_STORE_PATH=./data/chroma_db
COLLECTION_NAME=rag_documents

# Chunking: Controls retrieval granularity
CHUNK_SIZE=1000        # Characters per chunk
CHUNK_OVERLAP=200      # Overlap between chunks (20%)

# Models: Performance vs Cost trade-off
EMBEDDING_MODEL=text-embedding-ada-002  # $0.10 per 1M tokens
LLM_MODEL=gpt-3.5-turbo                 # $0.50 per 1M tokens

# Generation: Quality vs Speed
TEMPERATURE=0.7        # 0.0=deterministic, 1.0=creative
MAX_TOKENS=500         # Response length limit
```

## Data Flow

### Ingestion Pipeline

```
Document Files
    ↓
[DocumentIngestion.load_*()]
    ↓
LangChain Document objects
    ↓
[DocumentChunker.chunk_documents()]
    ↓
Chunked Documents (with metadata)
    ↓
[VectorStoreManager.create_vectorstore()]
    ↓
OpenAI Embeddings API (1536-dim vectors)
    ↓
ChromaDB Storage (persisted to disk)
```

### Query Pipeline

```
User Query (string)
    ↓
[VectorStoreManager.similarity_search()]
    ↓
OpenAI Embeddings API (query → vector)
    ↓
ChromaDB Similarity Search (cosine distance)
    ↓
Top-k Retrieved Documents
    ↓
[RAGInference.query()]
    ↓
Prompt Construction (query + context)
    ↓
OpenAI Chat Completion API
    ↓
Generated Answer + Source Attribution
```

## Trade-offs and Decisions

### 1. Chunking Strategy: Recursive (Default)

**Why**:
- Maintains semantic coherence
- Respects paragraph/sentence boundaries
- Best for most document types

**Trade-off**:
- Slightly variable chunk sizes
- More complex than fixed-size

**Alternative Use Cases**:
- Fixed-size: Strictly limited context windows
- Token-based: Precise cost control needed

### 2. Embedding Model: text-embedding-ada-002

**Why**:
- High quality (1536 dimensions)
- Good semantic understanding
- Widely used, well-tested

**Trade-off**:
- Requires OpenAI API (cost)
- Not customizable for domain

**Alternatives Considered**:
- sentence-transformers: Free, local, but lower quality
- Domain-specific models: Better for specialized content

### 3. LLM: GPT-3.5-turbo (Default)

**Why**:
- Fast inference
- Cost-effective
- Good enough for most use cases

**Trade-off**:
- Lower quality than GPT-4
- Less reasoning capability

**Upgrade Path**:
- GPT-4: Critical applications
- GPT-4-turbo: Balanced upgrade

### 4. Vector DB: ChromaDB

**Why**:
- No infrastructure needed
- Persistent storage
- Good performance for <1M docs

**Trade-off**:
- Not suitable for huge datasets
- Limited distributed capabilities

**Scale-up Path**:
- Pinecone: Cloud-managed, scales better
- Qdrant: Self-hosted, production-ready

### 5. Chunk Size: 1000 characters (Default)

**Why**:
- ~200 tokens (average)
- Good balance of context and precision
- Works well with k=4-6 retrieval

**Trade-off**:
- May be too large for very specific queries
- May be too small for complex topics

**Tuning Guidance**:
- Shorter docs → smaller chunks
- Complex topics → larger chunks
- Precise retrieval → smaller chunks

### 6. Chunk Overlap: 200 characters (20%)

**Why**:
- Maintains context continuity
- Prevents information loss at boundaries
- Reasonable storage overhead

**Trade-off**:
- Some redundancy
- Slightly larger storage

**Alternative Values**:
- 10%: Minimal overlap, less storage
- 30-50%: Maximum continuity, more storage

## Performance Characteristics

### Latency Breakdown

```
Query Latency Components:
┌─────────────────────────┬──────────┬──────────┐
│ Component               │ Typical  │ Variable │
├─────────────────────────┼──────────┼──────────┤
│ Embedding (query)       │ 100-200ms│ Low      │
│ Vector search (ChromaDB)│ 10-50ms  │ Medium   │
│ LLM inference (GPT-3.5) │ 1-3s     │ High     │
│ Total                   │ 1.2-3.5s │          │
└─────────────────────────┴──────────┴──────────┘
```

### Cost Analysis

```
Cost per 1000 Queries (typical):
┌─────────────────────┬─────────┬──────────┐
│ Component           │ Tokens  │ Cost     │
├─────────────────────┼─────────┼──────────┤
│ Query embeddings    │ 10k     │ $0.001   │
│ Retrieved context   │ 2M      │ $1.00    │
│ Generated responses │ 500k    │ $0.25    │
│ Total per 1k queries│         │ ~$1.26   │
└─────────────────────┴─────────┴──────────┘

Note: Assumes avg 10 tokens/query, k=4, 500 chars/chunk, 
500 tokens/response
```

### Scalability Limits

```
ChromaDB (Embedded):
- Documents: Up to ~1M docs
- Storage: Limited by disk space
- Concurrent queries: ~10-50 (single process)
- Throughput: ~100-500 QPS (cached embeddings)

Bottlenecks:
1. OpenAI API rate limits (primary)
2. ChromaDB query performance (secondary)
3. Disk I/O for large collections (tertiary)
```

## Extension Points

### Adding New Document Types

```python
# In document_ingestion.py
from langchain_community.document_loaders import UnstructuredHTMLLoader

SUPPORTED_FORMATS = {
    '.pdf': PyPDFLoader,
    '.txt': TextLoader,
    '.docx': Docx2txtLoader,
    '.html': UnstructuredHTMLLoader,  # Add new loader
}
```

### Custom Chunking Strategy

```python
# In chunking.py
from langchain.text_splitter import NLTKTextSplitter

class ChunkingStrategy(Enum):
    RECURSIVE = "recursive"
    FIXED_SIZE = "fixed_size"
    TOKEN_BASED = "token_based"
    NLTK_SENTENCE = "nltk_sentence"  # Add new strategy

# Implement in _create_splitter()
elif strategy == ChunkingStrategy.NLTK_SENTENCE:
    return NLTKTextSplitter(chunk_size=self.chunk_size)
```

### Alternative Vector Store

```python
# In vector_store.py
from langchain_community.vectorstores import Pinecone

class VectorStoreManager:
    def __init__(self, store_type="chroma", ...):
        if store_type == "chroma":
            self.vectorstore = Chroma(...)
        elif store_type == "pinecone":
            self.vectorstore = Pinecone(...)
```

### Local Embeddings (No API Key)

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

## Testing Strategy

### Unit Tests (Recommended to Add)

```python
# tests/test_chunking.py
def test_recursive_chunking():
    chunker = DocumentChunker(strategy=ChunkingStrategy.RECURSIVE)
    doc = Document(page_content="..." * 1000)
    chunks = chunker.chunk_documents([doc])
    assert all(len(c.page_content) <= chunker.chunk_size * 1.1 for c in chunks)

# tests/test_vector_store.py
def test_similarity_search():
    # Use mock embeddings for testing
    manager = VectorStoreManager()
    docs = [Document(page_content="test")]
    manager.create_vectorstore(docs)
    results = manager.similarity_search("test", k=1)
    assert len(results) == 1
```

### Integration Tests

```python
# tests/test_integration.py
def test_end_to_end_pipeline():
    assistant = RAGAssistant()
    assistant.ingest_documents("test_data/")
    result = assistant.query("What is AI?")
    assert 'answer' in result
    assert len(result['source_documents']) > 0
```

## Monitoring and Observability

### Recommended Metrics

```python
# Production monitoring
metrics = {
    'query_latency_ms': histogram,
    'retrieval_count': counter,
    'llm_tokens_used': counter,
    'cost_per_query': gauge,
    'error_rate': counter,
    'cache_hit_rate': gauge,
}
```

### Logging Strategy

```python
import logging

# Structured logging
logger.info("query_processed", extra={
    'query_length': len(question),
    'retrieved_docs': k,
    'inference_time_ms': elapsed_ms,
    'tokens_used': token_count,
})
```

## Security Considerations

1. **API Key Management**:
   - Never commit `.env` to git
   - Use environment variables in production
   - Rotate keys regularly

2. **Input Validation**:
   - Sanitize file paths
   - Validate document sizes
   - Rate limit queries

3. **Data Privacy**:
   - Sensitive documents → local embeddings only
   - Consider data residency requirements
   - Implement access controls on vector store

## Future Enhancements

1. **Caching Layer**: Redis for query results
2. **Async Processing**: Background document ingestion
3. **Evaluation Suite**: Automated quality metrics
4. **Multi-modal**: Support images, tables
5. **Hybrid Search**: Combine dense + sparse retrieval
6. **Re-ranking**: Two-stage retrieval for better precision
7. **Streaming**: Real-time response generation
8. **Web UI**: Gradio or Streamlit interface

## Conclusion

This RAG system prioritizes:
- ✅ **Simplicity**: Easy to understand and modify
- ✅ **Modularity**: Clean component boundaries
- ✅ **Flexibility**: Configurable for different use cases
- ✅ **Production-Ready**: Error handling, logging, persistence

It's designed as a foundation that can be extended based on specific requirements while maintaining code quality and maintainability.
