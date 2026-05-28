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


with st.sidebar:
    st.title("Document Upload")
    st.caption("Upload a PDF to query against.")

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        os.makedirs("data/documents", exist_ok=True)
        filename = os.path.basename(uploaded_file.name)
        save_path = os.path.join("data", "documents", filename)

        with open(save_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        if st.button("Ingest Document", type="primary"):
            with st.spinner("Chunking and embedding document..."):
                pipeline.ingest(save_path)
                st.session_state.ingested = True
            st.success(f"Ingested: {filename}")

    st.divider()

    if st.session_state.ingested:
        st.success("PASS: Document ready")
    else:
        st.warning("No document ingested yet")

    st.divider()
    st.caption("Confidence thresholds")
    st.markdown("**PASS**: score >= 0.75")
    st.markdown("**WARN**: score 0.45-0.74")
    st.markdown("**REFUSE**: score < 0.45")


st.title("RAG Confidence Gate")
st.caption("Answers grounded in your documents; refuses when confidence is too low.")

if not st.session_state.ingested:
    st.info("Upload and ingest a PDF from the sidebar to get started.")
    st.stop()


query = st.text_input(
    "Ask a question about your document",
    placeholder="What is the main topic of this document?",
    key="query_input",
)

col1, col2 = st.columns([1, 5])
with col1:
    ask = st.button("Ask", type="primary", use_container_width=True)
with col2:
    if st.button("Clear history", use_container_width=False):
        st.session_state.history = []
        st.rerun()


if ask and query.strip():
    with st.spinner("Retrieving and evaluating..."):
        response = pipeline.query(query)
        st.session_state.history.append({"query": query, "response": response})


def render_confidence(status, confidence):
    if status == "pass":
        st.success(f"PASS - Confidence: {confidence:.2f}")
    elif status == "warn":
        st.warning(
            f"LOW CONFIDENCE - Score: {confidence:.2f} - Verify against sources"
        )
    else:
        st.error(f"REFUSED - Confidence: {confidence:.2f}")


def render_answer(response):
    status = response.get("status")

    if status in ("pass", "warn") and response.get("answer"):
        st.markdown("**Answer:**")
        st.markdown(response["answer"])
        return

    if status == "refuse":
        reason = response.get("refusal_reason", "Confidence too low to answer reliably.")
        st.markdown(f"**Reason:** {reason}")

        suggestions = response.get("suggestions") or []
        if suggestions:
            st.markdown("**Suggestions:**")
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")


def render_sources(response):
    sources = response.get("sources") or []
    if not sources:
        return

    with st.expander(f"View {len(sources)} source chunks"):
        for index, source in enumerate(sources):
            similarity = float(source.get("similarity", 0))
            color = "green" if similarity > 0.75 else "orange" if similarity > 0.45 else "red"
            chunk_id = source.get("chunk_id", index)

            st.markdown(
                f"**Chunk {int(chunk_id) + 1}** - "
                f"similarity: :{color}[{similarity:.3f}]"
            )
            st.caption(source.get("text_preview", ""))
            st.markdown("---")


for item in reversed(st.session_state.history):
    response = item["response"]
    status = response.get("status")
    confidence = float(response.get("confidence_score", 0))

    st.divider()
    st.markdown(f"**Q: {item['query']}**")
    render_confidence(status, confidence)
    render_answer(response)
    render_sources(response)
