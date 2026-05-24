import json

from src.embedding.embedder import Embedder
from src.embedding.index import FAISSIndex
from src.gate.confidence import ConfidenceGate
from src.ingestion.chunker import Chunker
from src.ingestion.loader import DocumentLoader
from src.retrieval.retriever import Retriever


class RAGPipeline:
    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.index = FAISSIndex()
        self.retriever = Retriever(self.index, self.embedder)
        self.gate = ConfidenceGate()
        self.indexed = False

    def ingest(self, path):
        text = self.loader.load_pdf(path)
        chunks = self.chunker.chunk(text)
        embeddings = self.embedder.embed(chunks)
        self.index.add(embeddings, chunks)
        self.index.save("data/index/faiss.index", "data/index/chunks.pkl")
        self.indexed = True
        print(f"Ingested {len(chunks)} chunks.")

    def query(self, question, top_k=5):
        if not self.indexed:
            self.index.load("data/index/faiss.index", "data/index/chunks.pkl")
            self.indexed = True

        results = self.retriever.retrieve(question, top_k=top_k)
        scores = [r["score"] for r in results]

        _, _, response = self.gate.evaluate(question, results, scores)

        print(json.dumps(response, indent=2))
        return response
