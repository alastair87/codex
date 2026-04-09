from reasoning_harness import ReasoningHarness


def test_run_produces_expected_stages():
    harness = ReasoningHarness()
    result = harness.run("How should we launch the feature safely?")

    names = [step.name for step in result.trace]
    assert names == ["interpret", "hypothesize", "challenge", "answer"]
    assert 0.0 <= result.confidence <= 1.0
    assert "Proposed answer:" in result.answer


def test_empty_question_raises():
    harness = ReasoningHarness()

    try:
        harness.run("   ")
        assert False, "Expected ValueError for empty question"
    except ValueError as exc:
        assert "non-empty" in str(exc)
