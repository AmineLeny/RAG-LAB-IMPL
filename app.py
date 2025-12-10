import streamlit as st
from src import RAGPipeline

st.set_page_config(page_title="Document Q&A", layout="centered")
st.title("Document Q&A")

@st.cache_resource
def get_pipeline():
    pipe = RAGPipeline()
    pipe.ingest()
    return pipe

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Options")
    if st.button("Reload Documents"):
        st.cache_resource.clear()
        st.rerun()

with st.spinner("Loading..."):
    pipe = get_pipeline()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Type your question here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner(""):
            response = pipe.ask(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
