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
