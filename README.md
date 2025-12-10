# RAG Lab

Chat with your PDFs using a local LLM.

## What you need

- Python 3.10+
- Ollama installed and running

## Getting started

1. Open a terminal in this folder

2. Create a virtual environment and activate it:

   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Pull the models (do this once):

   ```
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

5. Run the app:

   ```
   streamlit run app.py
   ```

6. Upload a PDF in the sidebar, click Process, then start asking questions.

## Alternative: command line

If you don't want the web interface:

```python
from src import RAGPipeline

pipe = RAGPipeline()
pipe.ingest()
pipe.ask("what is this document about?")
```

## Troubleshooting

- Make sure Ollama is running before you start the app
- If embeddings are slow, just wait - first run takes a minute
- Check that your PDF isn't scanned images (needs actual text)
