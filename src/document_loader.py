import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader

def load_documents(path):
    if os.path.isfile(path):
        if path.endswith('.pdf'):
            return PyPDFLoader(path).load()
        return TextLoader(path, encoding='utf-8').load()
    
    docs = []
    for loader_cls, glob in [(TextLoader, "**/*.txt"), (PyPDFLoader, "**/*.pdf")]:
        try:
            loader = DirectoryLoader(path, glob=glob, loader_cls=loader_cls, 
                                    loader_kwargs={"encoding": "utf-8"} if loader_cls == TextLoader else {})
            docs.extend(loader.load())
        except Exception:
            pass
    return docs
