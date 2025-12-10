# RAG Lab

A simple question-answering system that reads your documents and answers questions about them.

## Setup

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

You also need Ollama running locally:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## How to use

Put your PDFs or text files in the `data/` folder, then:

```python
from src import RAGPipeline

pipe = RAGPipeline()
pipe.ingest()
print(pipe.ask("your question here"))
```

Or run `python -m src.rag_pipeline` for interactive mode.

## Config

Edit `.env` if you need different settings:

```
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
```
