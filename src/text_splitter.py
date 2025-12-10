from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import Config

def split_documents(documents, chunk_size=None, chunk_overlap=None):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or Config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)
