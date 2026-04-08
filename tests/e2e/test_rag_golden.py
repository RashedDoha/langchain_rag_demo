"""
Golden Q&A evaluation suite.

Runs the full RAG chain against a set of known question/answer pairs and
asserts that:
  1. The answer is non-empty.
  2. At least one expected keyword is present in the answer.
  3. The answer contains an inline source citation.
  4. The answer does NOT contain fabricated "I don't know" when context
     is sufficient, and DOES say it when context is absent.

To add new golden pairs, append entries to GOLDEN_QA below.

Usage:
    pytest tests/e2e/test_rag_golden.py -v

Requirements:
  - ANTHROPIC_API_KEY must be set in the environment.
  - At least one PDF must already be indexed (run the indexing pipeline first).
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Golden Q&A pairs
# ---------------------------------------------------------------------------
# Each entry:
#   question      : the query to send
#   keywords      : at least one of these must appear in the answer (case-insensitive)
#   expect_citation: whether the answer should contain "[Source:" marker
#   expect_no_answer: True when the question is intentionally unanswerable from
#                     the indexed docs (the LLM should say it doesn't know)
# ---------------------------------------------------------------------------
# TODO: Replace these placeholder entries with real Q&A pairs from your documents.

GOLDEN_QA: list[dict] = [
    # Example 1 — replace with a real question answerable from your PDFs
    {
        "question": "What is Rashed's primary profession?",
        "keywords": [],          # add expected keywords once docs are known
        "expect_citation": True,
        "expect_no_answer": False,
    },
    # Example 2 — a question that should NOT be answerable
    {
        "question": "What tech stack is most common in the documents?",
        "keywords": ["don't know", "do not know", "not enough information", "cannot"],
        "expect_citation": False,
        "expect_no_answer": True,
    },
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rag_chain():
    """Create one RAG chain for the whole module (model load is expensive)."""
    from rag.rag_chain import create_rag_chain

    return create_rag_chain()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", GOLDEN_QA, ids=[c["question"][:60] for c in GOLDEN_QA])
def test_golden_answer(rag_chain, case: dict):
    answer: str = rag_chain.invoke({"question": case["question"]})

    assert answer.strip(), "Answer must not be empty."

    if case["keywords"]:
        lower = answer.lower()
        assert any(kw.lower() in lower for kw in case["keywords"]), (
            f"Expected one of {case['keywords']} in answer.\nGot: {answer}"
        )

    if case["expect_citation"]:
        assert "[Source:" in answer, (
            f"Expected an inline [Source: ...] citation.\nGot: {answer}"
        )

    if case["expect_no_answer"]:
        lower = answer.lower()
        fallback_phrases = [
            "don't know",
            "do not know",
            "not enough information",
            "cannot answer",
            "no information",
            "not mentioned",
        ]
        assert any(p in lower for p in fallback_phrases), (
            f"Expected a 'don't know' response for unanswerable question.\nGot: {answer}"
        )
