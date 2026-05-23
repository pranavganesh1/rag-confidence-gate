from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return np.array(embeddings).astype("float32")

    def embed_query(self, query):
        return self.model.encode([query]).astype("float32")
