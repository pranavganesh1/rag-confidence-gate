import argparse
from pathlib import Path

from src.embedding.embedder import Embedder
from src.embedding.index import FAISSIndex
from src.ingestion.chunker import Chunker
from src.ingestion.loader import DocumentLoader
from src.retrieval.retriever import Retriever


def main():
    parser = argparse.ArgumentParser(description="Day 2 embedding, indexing, and retrieval test.")
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default="data/documents/sample_day1.pdf",
        help="Path to a PDF file to load, chunk, embed, and search.",
    )
    parser.add_argument(
        "--query",
        default="What is this document about?",
        help="Question to search for in the document chunks.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--index-path",
        default="data/index/faiss.index",
        help="Where to save the FAISS index.",
    )
    parser.add_argument(
        "--chunks-path",
        default="data/index/chunks.pkl",
        help="Where to save the chunk metadata.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}. Put a PDF in data/documents or pass a path."
        )

    loader = DocumentLoader()
    chunker = Chunker()
    text = loader.load_pdf(pdf_path)
    chunks = chunker.chunk(text)
    print(f"Chunks: {len(chunks)}")

    embedder = Embedder()
    index = FAISSIndex()
    embeddings = embedder.embed(chunks)
    print(f"Embeddings: {embeddings.shape}")
    index.add(embeddings, chunks)

    index.save(args.index_path, args.chunks_path)
    print("Index saved.")

    loaded_index = FAISSIndex()
    loaded_index.load(args.index_path, args.chunks_path)

    retriever = Retriever(loaded_index, embedder)
    results = retriever.retrieve(args.query, top_k=args.top_k)

    for i, result in enumerate(results):
        print(f"\n--- Result {i + 1} | Score: {result['score']:.4f} ---")
        print(result["chunk"][:300])


if __name__ == "__main__":
    main()
