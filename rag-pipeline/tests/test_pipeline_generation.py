import sys
import types

from src.gate.confidence import ConfidenceGate

sentence_transformers = types.ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", sentence_transformers)

faiss = types.ModuleType("faiss")
faiss.IndexFlatIP = object
faiss.normalize_L2 = lambda value: value
sys.modules.setdefault("faiss", faiss)

fitz = types.ModuleType("fitz")
sys.modules.setdefault("fitz", fitz)

from src.pipeline import RAGPipeline


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, question, top_k=5):
        return self.results


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return "Grounded answer from Chunk 1."


def make_pipeline(results):
    pipeline = object.__new__(RAGPipeline)
    pipeline.indexed = True
    pipeline.retriever = FakeRetriever(results)
    pipeline.gate = ConfidenceGate()
    pipeline.generator = FakeGenerator()
    return pipeline


def test_refuse_does_not_call_generator():
    results = [
        {"chunk": "this is chunk number 0 with some words", "score": 0.20},
        {"chunk": "this is chunk number 1 with some words", "score": 0.18},
        {"chunk": "this is chunk number 2 with some words", "score": 0.15},
    ]
    pipeline = make_pipeline(results)

    response = pipeline.query("unrelated query")

    assert response["status"] == "refuse"
    assert response["answer"] is None
    assert pipeline.generator.calls == 0


def test_pass_calls_generator_and_attaches_answer():
    results = [
        {"chunk": "alpha answer lives in chunk words", "score": 0.92},
        {"chunk": "alpha supporting words", "score": 0.88},
        {"chunk": "alpha more words", "score": 0.85},
    ]
    pipeline = make_pipeline(results)

    response = pipeline.query("alpha answer chunk words")

    assert response["status"] == "pass"
    assert response["answer"] == "Grounded answer from Chunk 1."
    assert pipeline.generator.calls == 1
