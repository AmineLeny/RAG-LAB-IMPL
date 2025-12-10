from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from .config import Config

class VectorStore:
    def __init__(self, persist_dir=None):
        self.persist_dir = persist_dir or Config.PERSIST_DIR
        self.embeddings = OllamaEmbeddings(
            model=Config.EMBED_MODEL,
            base_url=Config.OLLAMA_BASE_URL
        )
        self.vectorstore = None
    
    def create_from_documents(self, documents):
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        return self.vectorstore
    
    def load_existing(self):
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        return self.vectorstore
    
    def search(self, query, k=4):
        if not self.vectorstore:
            self.load_existing()
        return self.vectorstore.similarity_search(query, k=k)
    
    def as_retriever(self, k=4):
        if not self.vectorstore:
            self.load_existing()
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
