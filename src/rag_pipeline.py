from .config import Config
from .document_loader import load_documents
from .text_splitter import split_documents
from .vector_store import VectorStore
from .rag_chain import RAGChain

class RAGPipeline:
    def __init__(self, data_path=None):
        self.data_path = data_path or Config.DATA_DIR
        self.vector_store = VectorStore()
        self.chain = None
    
    def ingest(self):
        print(f"Loading docs from {self.data_path}...")
        docs = load_documents(self.data_path)
        print(f"Found {len(docs)} documents")
        
        chunks = split_documents(docs)
        print(f"Split into {len(chunks)} chunks")
        
        print("Creating embeddings...")
        self.vector_store.create_from_documents(chunks)
        print("Done!")
        return len(chunks)
    
    def setup_chain(self):
        retriever = self.vector_store.as_retriever()
        self.chain = RAGChain(retriever)
        return self.chain
    
    def ask(self, question):
        if not self.chain:
            self.setup_chain()
        return self.chain.ask(question)
    
    def run(self):
        self.ingest()
        self.setup_chain()
        
        print("\nRAG system ready! Type 'quit' to exit.\n")
        while True:
            question = input("You: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if question:
                print(f"\nAssistant: {self.ask(question)}\n")

if __name__ == "__main__":
    RAGPipeline().run()
