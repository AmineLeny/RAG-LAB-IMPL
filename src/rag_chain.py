from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .config import Config

class RAGChain:
    PROMPT_TEMPLATE = """Answer based on this context:

{context}

Question: {question}

Give a clear, helpful answer. If the context doesn't have enough info, say so."""

    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.7
        )
        self.chain = self._build_chain()
    
    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _build_chain(self):
        prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        return (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask(self, question):
        return self.chain.invoke(question)
