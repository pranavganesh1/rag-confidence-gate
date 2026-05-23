import pickle
from pathlib import Path

import faiss


class FAISSIndex:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []

    def add(self, embeddings, chunks):
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must have the same length")

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def save(self, index_path, chunks_path):
        index_path = Path(index_path)
        chunks_path = Path(chunks_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with chunks_path.open("wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, index_path, chunks_path):
        self.index = faiss.read_index(str(index_path))
        self.dim = self.index.d
        with Path(chunks_path).open("rb") as f:
            self.chunks = pickle.load(f)
