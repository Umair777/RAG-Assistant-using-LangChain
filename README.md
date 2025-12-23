# RAG Assistant using LangChain

A comprehensive Retrieval-Augmented Generation (RAG) system built with LangChain, exploring document ingestion, chunking strategies, vector retrieval, and inference trade-offs for real-world AI applications.

## 🎯 Overview

This project demonstrates the design and implementation of a production-ready RAG system, covering:

- **📥 Document Ingestion**: Support for multiple formats (PDF, TXT, DOCX)
- **✂️ Chunking Strategies**: Recursive, fixed-size, and token-based splitting
- **🔍 Vector Retrieval**: Efficient similarity search using ChromaDB
- **🤖 Inference Pipeline**: OpenAI-powered question answering with configurable parameters

## 🏗️ Architecture

```
RAG Pipeline Flow:
┌─────────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────┐
│  Documents  │ -> │ Chunking │ -> │ Embedding│ -> │ Vector Store│
└─────────────┘    └──────────┘    └──────────┘    └─────────────┘
                                                            |
                                                            v
┌─────────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────┐
│   Answer    │ <- │   LLM    │ <- │Retrieval │ <- │    Query    │
└─────────────┘    └──────────┘    └──────────┘    └─────────────┘
```

## 🚀 Features

### Document Ingestion
- Load single files or entire directories
- Support for PDF, TXT, and DOCX formats
- Automatic format detection and processing
- Batch document processing

### Chunking Strategies

1. **Recursive Character Splitting** (Recommended)
   - Respects document structure (paragraphs, sentences)
   - Maintains semantic coherence
   - Best for most use cases

2. **Fixed-Size Splitting**
   - Consistent chunk sizes
   - Simple and predictable
   - Good for uniform processing

3. **Token-Based Splitting**
   - Ensures chunks fit within model token limits
   - Accurate cost estimation
   - Important for models with strict constraints

### Vector Retrieval
- ChromaDB for efficient vector storage
- OpenAI embeddings (text-embedding-ada-002)
- Similarity search with relevance scores
- Configurable retrieval (top-k results)

### Inference Engine
- OpenAI GPT models (gpt-3.5-turbo, gpt-4)
- Configurable temperature and token limits
- Source document tracking
- Batch query processing

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/Umair777/RAG-Assistant-using-LangChain.git
cd RAG-Assistant-using-LangChain
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## 🔧 Configuration

Edit `.env` file to customize settings:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Vector Store Configuration
VECTOR_STORE_PATH=./data/chroma_db
COLLECTION_NAME=rag_documents

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Model Configuration
EMBEDDING_MODEL=text-embedding-ada-002
LLM_MODEL=gpt-3.5-turbo
TEMPERATURE=0.7
MAX_TOKENS=500
```

## 💻 Usage

### Command Line Interface

#### Ingest Documents
```bash
# Ingest a single file
python src/cli.py ingest path/to/document.pdf

# Ingest a directory
python src/cli.py ingest path/to/documents/
```

#### Query the System
```bash
# Ask a question
python src/cli.py query "What is machine learning?"

# Query with custom top-k
python src/cli.py query "Explain RAG systems" -k 5

# Verbose output
python src/cli.py query "What are vector databases?" -v
```

#### Interactive Mode
```bash
python src/cli.py interactive
```

#### System Information
```bash
python src/cli.py info
```

### Python API

```python
from rag_assistant.rag_assistant import RAGAssistant

# Initialize
assistant = RAGAssistant()

# Ingest documents
assistant.ingest_documents("path/to/documents", is_directory=True)

# Query
result = assistant.query("What is retrieval-augmented generation?")
print(result['answer'])

# Show sources
for doc in result['source_documents']:
    print(f"Source: {doc.metadata['source']}")
```

### Example Scripts

#### Basic Demo
```bash
python examples/demo.py
```

#### Chunking Strategy Comparison
```bash
python examples/chunking_comparison.py
```

## 📊 Chunking Strategy Trade-offs

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| **Recursive** | Semantic coherence, respects structure | Slightly more complex | General-purpose RAG |
| **Fixed-Size** | Simple, consistent sizes | May split awkwardly | Uniform processing needs |
| **Token-Based** | Fits model limits, cost control | Requires tokenizer | Token-constrained models |

### Chunk Size Recommendations

- **Small (200-500)**: Precise retrieval, less context
- **Medium (500-1000)**: Balanced approach (recommended)
- **Large (1000-2000)**: More context, less precise

### Overlap Considerations

- **Low (10-20%)**: Less redundancy, faster
- **Medium (20-30%)**: Balanced (recommended)
- **High (30-50%)**: Better continuity, more storage

## 🔍 Inference Trade-offs

### Temperature Settings

- **0.0-0.3**: Deterministic, factual responses
- **0.4-0.7**: Balanced creativity and accuracy
- **0.8-1.0**: More creative, less predictable

### Retrieval Parameters (k)

- **k=2-3**: Fast, focused on most relevant
- **k=4-6**: Balanced (recommended)
- **k=7-10**: Comprehensive, may include noise

### Model Selection

- **gpt-3.5-turbo**: Fast, cost-effective
- **gpt-4**: Higher quality, more expensive
- **gpt-4-turbo**: Balanced performance and cost

## 📁 Project Structure

```
RAG-Assistant-using-LangChain/
├── src/
│   ├── rag_assistant/
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration management
│   │   ├── document_ingestion.py  # Document loading
│   │   ├── chunking.py            # Chunking strategies
│   │   ├── vector_store.py        # Vector database management
│   │   ├── inference.py           # RAG inference engine
│   │   └── rag_assistant.py       # Main orchestrator
│   └── cli.py                      # Command-line interface
├── examples/
│   ├── demo.py                     # Basic demonstration
│   └── chunking_comparison.py     # Strategy comparison
├── requirements.txt                # Python dependencies
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## 🛠️ Development

### Testing Different Configurations

```python
from rag_assistant.rag_assistant import RAGAssistant

assistant = RAGAssistant()

# Change chunking strategy
assistant.change_chunking_strategy(
    strategy="token_based",
    chunk_size=500,
    chunk_overlap=100
)

# Update inference settings
assistant.update_inference_settings(
    temperature=0.5,
    max_tokens=300
)

# Get system info
info = assistant.get_system_info()
print(info)
```

### Custom Prompts

```python
from rag_assistant.inference import RAGInference

inference = RAGInference(vector_store_manager)

custom_template = """Use the context to answer the question.
Context: {context}
Question: {question}
Detailed Answer:"""

inference.set_custom_prompt(custom_template, ["context", "question"])
```

## 📚 Key Concepts

### Retrieval-Augmented Generation (RAG)

RAG enhances LLMs by:
1. Retrieving relevant information from external sources
2. Augmenting the prompt with retrieved context
3. Generating responses based on both the LLM's knowledge and retrieved information

### Why RAG?

- **Accuracy**: Grounds responses in actual documents
- **Up-to-date**: Access to current information
- **Transparency**: Source attribution
- **Specialization**: Domain-specific knowledge without fine-tuning

### Design Decisions

1. **ChromaDB**: Fast, simple, and effective for most use cases
2. **OpenAI Embeddings**: High-quality semantic representations
3. **Configurable Chunking**: Flexibility for different document types
4. **Modular Architecture**: Easy to extend and customize

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- Additional document formats (HTML, CSV, JSON)
- More chunking strategies (semantic chunking, sliding window)
- Alternative vector databases (Pinecone, Weaviate, Qdrant)
- Local embedding models (sentence-transformers)
- Evaluation metrics and benchmarking
- Web interface

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [OpenAI](https://openai.com/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes, demonstrating RAG system design and implementation patterns.