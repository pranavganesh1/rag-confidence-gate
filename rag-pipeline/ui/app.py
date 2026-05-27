import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

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
            filename = os.path.basename(uploaded_file.name)
            save_path = os.path.join("data", "documents", filename)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Indexing {filename}..."):
                pipeline.ingest(save_path)

            progress.progress((i + 1) / len(uploaded_files))

        st.session_state.indexed = True
        st.success(f"Indexed {len(uploaded_files)} document(s).")

    st.divider()
    st.markdown("**How confidence works:**")
    st.markdown("🟢 **High** — answer generated with citations")
    st.markdown("🟡 **Medium** — answer generated with warning")
    st.markdown("🔴 **Low** — refused, won't hallucinate")


st.title("🔍 RAG with Confidence Gating")
st.markdown(
    "Ask questions about your uploaded documents. The system refuses to answer "
    "rather than hallucinate."
)

if not st.session_state.indexed:
    st.info("Upload and index documents using the sidebar to get started.")
    st.stop()


query = st.text_input(
    "Ask a question",
    placeholder="What does the document say about...?",
    key="query_input",
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_button = st.button("Ask", type="primary", use_container_width=True)
with col2:
    top_k = st.slider("Chunks to retrieve", min_value=3, max_value=10, value=5)

if ask_button and query.strip():
    with st.spinner("Retrieving and evaluating..."):
        response = pipeline.query(query, top_k=top_k)
        st.session_state.history.insert(0, {"query": query, "response": response})


def render_response(response):
    status = response.get("status")
    confidence = response.get("confidence_score", 0)

    if status == "pass":
        st.success(f"🟢 High Confidence — {confidence:.0%}")
    elif status == "warn":
        st.warning(f"🟡 Medium Confidence — {confidence:.0%} — verify against sources")
    elif status == "refuse":
        st.error(f"🔴 Low Confidence — {confidence:.0%} — refused to answer")

    if status in ("pass", "warn") and response.get("answer"):
        st.markdown("### Answer")
        st.markdown(response["answer"])
    elif status == "refuse":
        st.markdown("### Why was this refused?")
        st.markdown(response.get("refusal_reason", "Confidence too low."))

        suggestions = response.get("suggestions", [])
        if suggestions:
            st.markdown("**Suggestions:**")
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")

    if response.get("sources"):
        st.markdown("### Sources")
        for i, source in enumerate(response["sources"]):
            similarity = source.get("similarity", 0)
            with st.expander(f"Chunk {i + 1} — similarity: {similarity:.3f}"):
                st.markdown(source.get("text_preview", ""))


for item in st.session_state.history:
    with st.container():
        st.markdown(f"**Q: {item['query']}**")
        render_response(item["response"])
        st.divider()
