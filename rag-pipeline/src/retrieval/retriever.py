import faiss


class Retriever:
    def __init__(self, index, embedder):
        self.index = index
        self.embedder = embedder

    def retrieve(self, query, top_k=5):
        query_vec = self.embedder.embed_query(query)
        faiss.normalize_L2(query_vec)

        limit = min(top_k, self.index.index.ntotal)
        if limit == 0:
            return []

        scores, indices = self.index.index.search(query_vec, limit)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append(
                    {
                        "chunk": self.index.chunks[idx],
                        "score": float(score),
                        "index": int(idx),
                    }
                )
        return results
