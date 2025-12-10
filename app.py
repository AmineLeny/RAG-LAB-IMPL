import streamlit as st
import tempfile
import os
from src import load_documents, split_documents, VectorStore, RAGChain

st.set_page_config(page_title="Document Q&A", layout="centered")
st.title("Document Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

with st.sidebar:
    uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    
    if uploaded:
        if st.button("Process"):
            with st.spinner("Reading and embedding..."):
                suffix = os.path.splitext(uploaded.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                
                docs = load_documents(tmp_path)
                chunks = split_documents(docs)
                
                store = VectorStore(persist_dir=tempfile.mkdtemp())
                store.create_from_documents(chunks)
                
                st.session_state.chain = RAGChain(store.as_retriever())
                st.session_state.doc_name = uploaded.name
                st.session_state.messages = []
                os.unlink(tmp_path)
            
            st.success(f"Ready - {len(chunks)} chunks")

if st.session_state.doc_name:
    st.caption(f"Chatting with: {st.session_state.doc_name}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question"):
    if not st.session_state.chain:
        st.warning("Upload a document first")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner(""):
                response = st.session_state.chain.ask(prompt)
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
