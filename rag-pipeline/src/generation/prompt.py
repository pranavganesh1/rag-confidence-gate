SYSTEM_PROMPT = """You are a precise document question-answering assistant.

STRICT RULES - you must follow these without exception:
1. Answer ONLY using the context chunks provided below.
2. If the answer is not in the context, say exactly: "I cannot find this in the provided documents."
3. Never use your training knowledge to fill gaps.
4. Never guess, infer beyond what is written, or make assumptions.
5. Always cite which chunk number(s) your answer comes from.

Violating these rules defeats the purpose of this system."""


def build_prompt(query, chunks):
    """
    Builds the final prompt sent to the LLM.
    chunks: list of retrieved chunk dicts with 'chunk' and 'score' keys
    """
    context_block = ""
    for i, c in enumerate(chunks):
        context_block += f"\n--- Chunk {i + 1} (similarity: {c['score']:.3f}) ---\n"
        context_block += c["chunk"][:600]
        context_block += "\n"

    prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context_block}

QUESTION: {query}

ANSWER (cite chunk numbers):"""

    return prompt
