import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import RAGPipeline


st.set_page_config(
    page_title="RAG Confidence Gate",
    page_icon="🔍",
    layout="wide",
)


if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(backend="ollama")
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "history" not in st.session_state:
    st.session_state.history = []

pipeline = st.session_state.pipeline


with st.sidebar:
    st.title("📄 Document Upload")
    st.markdown("Upload one or more PDFs to build your knowledge base.")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Index Documents", type="primary"):
        os.makedirs("data/documents", exist_ok=True)
        progress = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):
            save_path = os.path.join("data", "documents", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Indexing {uploaded_file.name}..."):
                pipeline.ingest(save_path)

            progress.progress((i + 1) / len(uploaded_files))

        st.session_state.indexed = True
        st.success(f"Indexed {len(uploaded_files)} document(s).")

    st.divider()
    st.markdown("**How confidence works:**")
    st.markdown("🟢 **High** — answer generated with citations")
    st.markdown("🟡 **Medium** — answer generated with warning")
    st.markdown("🔴 **Low** — refused, won't hallucinate")
