from src.gate.confidence import ConfidenceGate, GateResult


gate = ConfidenceGate(high_threshold=0.75, low_threshold=0.45)


def make_chunks(scores):
    return [
        {"chunk": f"this is chunk number {i} with some words", "score": s}
        for i, s in enumerate(scores)
    ]


def test_high_confidence_passes():
    scores = [0.92, 0.88, 0.85, 0.83, 0.80]
    chunks = make_chunks(scores)
    result, confidence, _ = gate.evaluate("what is chunk number one", chunks, scores)
    assert result == GateResult.PASS
    assert confidence >= 0.75


def test_tight_spread_boosts_confidence():
    scores = [0.85, 0.84, 0.83, 0.82, 0.81]
    chunks = make_chunks(scores)
    result, _, _ = gate.evaluate("chunk words", chunks, scores)
    assert result == GateResult.PASS


def test_medium_confidence_warns():
    scores = [0.65, 0.50, 0.40, 0.30, 0.20]
    chunks = make_chunks(scores)
    result, confidence, _ = gate.evaluate("chunk words", chunks, scores)
    assert result == GateResult.WARN
    assert 0.45 <= confidence < 0.75


def test_low_confidence_refuses():
    scores = [0.30, 0.25, 0.20, 0.15, 0.10]
    chunks = make_chunks(scores)
    result, confidence, _ = gate.evaluate("something completely unrelated", chunks, scores)
    assert result == GateResult.REFUSE
    assert confidence < 0.45


def test_refuse_has_reason_and_suggestions():
    scores = [0.20, 0.18, 0.15, 0.12, 0.10]
    chunks = make_chunks(scores)
    _, _, response = gate.evaluate("unrelated query", chunks, scores)
    assert response["refusal_reason"] is not None
    assert len(response["suggestions"]) > 0
    assert response["answer"] is None


def test_wide_spread_lowers_confidence():
    scores = [0.80, 0.50, 0.30, 0.20, 0.10]
    chunks = make_chunks(scores)
    _, confidence, _ = gate.evaluate("some query", chunks, scores)
    assert confidence < 0.75
