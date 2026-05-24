def top1_score(scores):
    """
    Raw cosine similarity of the best matching chunk.
    High value = at least one very relevant chunk exists.
    """
    return float(scores[0])


def score_spread(scores):
    """
    Difference between best and worst retrieved chunk score.
    Tight spread = all chunks relevant.
    Wide spread = retrieval is inconsistent.
    Returns inverted so higher = better.
    """
    spread = float(scores[0]) - float(scores[-1])
    return 1.0 - spread


def semantic_overlap(query, chunk):
    """
    Fraction of query words that appear in the top chunk.
    Sanity check against embedding quirks.
    """
    query_words = set(query.lower().split())
    chunk_words = set(chunk.lower().split())
    if not query_words:
        return 0.0

    overlap = query_words & chunk_words
    return len(overlap) / len(query_words)
