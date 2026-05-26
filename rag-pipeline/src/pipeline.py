import json

from src.embedding.embedder import Embedder
from src.embedding.index import FAISSIndex
from src.gate.confidence import ConfidenceGate, GateResult
from src.generation.generator import get_generator
from src.generation.prompt import build_prompt
from src.ingestion.chunker import Chunker
from src.ingestion.loader import DocumentLoader
from src.retrieval.retriever import Retriever


class RAGPipeline:
    def __init__(self, backend="ollama"):
        self.loader = DocumentLoader()
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.index = FAISSIndex()
        self.retriever = Retriever(self.index, self.embedder)
        self.gate = ConfidenceGate()
        self.generator = get_generator(backend)
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

        gate_result, _, response = self.gate.evaluate(question, results, scores)

        if gate_result == GateResult.REFUSE:
            print("\n[GATE] REFUSED - confidence too low to generate.")
            print(json.dumps(response, indent=2))
            return response

        prompt = build_prompt(question, results)
        answer = self.generator.generate(prompt)

        response["answer"] = answer
        if gate_result == GateResult.WARN:
            response["warning"] = "Low confidence - verify this answer against sources."

        print(json.dumps(response, indent=2))
        return response
