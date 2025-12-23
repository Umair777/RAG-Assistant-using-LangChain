# Quick Start Guide

## Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage Examples

### CLI Mode

#### Ingest Documents
```bash
# Single file
python src/cli.py ingest my_document.pdf

# Whole directory
python src/cli.py ingest ./documents/
```

#### Ask Questions
```bash
python src/cli.py query "What is machine learning?"
```

#### Interactive Session
```bash
python src/cli.py interactive
```

### Python API

```python
from rag_assistant.rag_assistant import RAGAssistant

# Initialize
assistant = RAGAssistant()

# Ingest documents
assistant.ingest_documents("./documents", is_directory=True)

# Query
result = assistant.query("What is RAG?")
print(result['answer'])
```

## Examples

### Run Demo
```bash
python examples/demo.py
```

### Compare Chunking Strategies
```bash
python examples/chunking_comparison.py
```

### Understand Trade-offs
```bash
python examples/inference_tradeoffs.py
```

## Common Tasks

### Change Chunking Strategy
```python
assistant.change_chunking_strategy(
    strategy="token_based",
    chunk_size=500
)
```

### Adjust Inference Settings
```python
assistant.update_inference_settings(
    temperature=0.5,
    max_tokens=300
)
```

### Get System Info
```bash
python src/cli.py info
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| CHUNK_SIZE | 1000 | Characters per chunk |
| CHUNK_OVERLAP | 200 | Overlap between chunks |
| LLM_MODEL | gpt-3.5-turbo | Language model |
| TEMPERATURE | 0.7 | Creativity (0.0-1.0) |
| MAX_TOKENS | 500 | Response length |

## Troubleshooting

### "OPENAI_API_KEY is required"
- Create `.env` file from `.env.example`
- Add your API key: `OPENAI_API_KEY=sk-...`

### "No vector store found"
- Run `ingest` command first to create vector store
- Or check `VECTOR_STORE_PATH` in `.env`

### Out of memory
- Reduce `CHUNK_SIZE`
- Process fewer documents at once
- Use smaller embedding model

## Next Steps

1. ✅ Run the demo: `python examples/demo.py`
2. ✅ Ingest your documents
3. ✅ Experiment with settings
4. ✅ Read ARCHITECTURE.md for deep dive
