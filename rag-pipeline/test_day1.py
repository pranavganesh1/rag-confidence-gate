import argparse
from pathlib import Path

from src.ingestion.chunker import Chunker
from src.ingestion.loader import DocumentLoader


def main():
    parser = argparse.ArgumentParser(description="Day 1 PDF loading and chunking test.")
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default="data/documents/your_file.pdf",
        help="Path to a PDF file to load and chunk.",
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

    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk[:200])


if __name__ == "__main__":
    main()
