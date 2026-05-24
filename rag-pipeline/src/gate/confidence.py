from enum import Enum

from src.gate.scorer import score_spread, semantic_overlap, top1_score


class GateResult(Enum):
    PASS = "pass"
    WARN = "warn"
    REFUSE = "refuse"


class ConfidenceGate:
    def __init__(self, high_threshold=0.75, low_threshold=0.45):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def evaluate(self, query, chunks, scores):
        """
        query   : raw query string
        chunks  : list of retrieved chunk dicts (from retriever)
        scores  : list of cosine similarity scores (floats)
        """
        if not chunks or not scores:
            confidence = 0.0
            result = GateResult.REFUSE
            return result, confidence, self._build_response(result, confidence, chunks, query)

        s1 = top1_score(scores)
        s2 = score_spread(scores)
        s3 = semantic_overlap(query, chunks[0]["chunk"])

        confidence = (0.5 * s1) + (0.3 * s2) + (0.2 * s3)
        confidence = round(float(confidence), 4)

        if confidence >= self.high_threshold:
            result = GateResult.PASS
        elif confidence >= self.low_threshold:
            result = GateResult.WARN
        else:
            result = GateResult.REFUSE

        return result, confidence, self._build_response(result, confidence, chunks, query)

    def _build_response(self, result, confidence, chunks, query):
        base = {
            "status": result.value,
            "confidence_score": confidence,
            "sources": [
                {
                    "chunk_id": str(i),
                    "text_preview": c["chunk"][:200],
                    "similarity": c["score"],
                }
                for i, c in enumerate(chunks)
            ],
        }

        if result == GateResult.REFUSE:
            base["answer"] = None
            base["refusal_reason"] = (
                f"Retrieval confidence is too low ({confidence:.2f}) "
                f"to answer '{query}' reliably."
            )
            base["suggestions"] = [
                "Try rephrasing the question",
                "Check if this topic exists in your uploaded documents",
                "Upload additional relevant documents",
            ]
        else:
            base["answer"] = None
            base["refusal_reason"] = None
            base["suggestions"] = None

        return base
