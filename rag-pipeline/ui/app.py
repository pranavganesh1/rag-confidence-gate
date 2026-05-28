import os
import sys

import streamlit as st


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.pipeline import RAGPipeline


st.set_page_config(
    page_title="RAG Confidence Gate",
    page_icon="search",
    layout="wide",
)


# Session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(backend="ollama")
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "history" not in st.session_state:
    st.session_state.history = []

pipeline = st.session_state.pipeline


st.title("RAG Confidence Gate")
st.caption("Answers grounded in your documents; refuses when confidence is too low.")

if not st.session_state.ingested:
    st.info("Upload and ingest a PDF from the sidebar to get started.")
