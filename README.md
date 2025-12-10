# LAB 2: RAG with Ollama

Build a Retrieval-Augmented Generation system using local LLMs.

## Setup

```bash
# create venv
python -m venv venv
.\venv\Scripts\activate

# install deps
pip install -r requirements.txt

# make sure ollama is running with the embedding model
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Project Structure

```
LAB2-RAG/
├── data/               # your documents go here
├── src/                # RAG modules
├── notebooks/          # jupyter lab
├── chroma_db/          # vector store (auto-created)
├── .env                # config
└── requirements.txt
```

## Usage

### Quick Start

```python
from src import RAGPipeline

pipe = RAGPipeline()
pipe.ingest()           # load and embed docs
answer = pipe.ask("What is RAG?")
print(answer)
```

### Interactive Mode

```bash
python -m src.rag_pipeline
```

### Step by Step

```python
from src import load_documents, split_documents, VectorStore, RAGChain

# load your docs
docs = load_documents("data/")

# chunk them
chunks = split_documents(docs)

# embed and store
store = VectorStore()
store.create_from_documents(chunks)

# create the chain
chain = RAGChain(store.as_retriever())

# ask stuff
print(chain.ask("What is machine learning?"))
```

## Config

Edit `.env` to change settings:

```
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

## Notes

- First run takes a bit since it creates embeddings
- Embeddings are cached in `chroma_db/`
- Add your own docs to `data/` folder (txt or pdf)
