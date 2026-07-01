"""Unit tests for the semantic chunker using a deterministic MOCK embedder.

The mock maps each window to a vector based on which topic keyword it contains,
so windows about the same topic are near-identical and the cross-topic gap is a
large cosine distance — the chunker must cut exactly there.
"""

from __future__ import annotations

import numpy as np

from ingestion.chunking.semantic_chunker import (
    _post_process,
    semantic_chunk,
    split_sentences,
)

# Two clearly distinct topics.
TOPIC_VECS = {
    "satellite": np.array([1.0, 0.0, 0.0]),
    "enzyme": np.array([0.0, 1.0, 0.0]),
}


def mock_embedder(texts: list[str]) -> np.ndarray:
    out = []
    for t in texts:
        tl = t.lower()
        # A window straddling the boundary mixes both vectors.
        v = np.zeros(3)
        if "satellite" in tl:
            v = v + TOPIC_VECS["satellite"]
        if "enzyme" in tl:
            v = v + TOPIC_VECS["enzyme"]
        if not v.any():
            v = np.array([0.0, 0.0, 1.0])
        out.append(v)
    return np.asarray(out)


def test_split_sentences_protects_abbreviations():
    text = "The course covers M.Tech. topics, e.g. control systems. It is hard."
    sents = split_sentences(text)
    assert len(sents) == 2
    assert "M.Tech." in sents[0] and "e.g." in sents[0]


def test_split_sentences_protects_initials_and_decimals():
    text = "See A. H Shapiro for details. The value is 3.14 exactly. Done."
    sents = split_sentences(text)
    assert len(sents) == 3
    assert "3.14" in sents[1]


def test_chunker_splits_at_topic_boundary():
    text = (
        "The satellite orbits the planet. The satellite uses solar panels. "
        "The satellite transmits telemetry. "
        "The enzyme catalyzes the reaction. The enzyme lowers activation energy. "
        "The enzyme denatures at high heat."
    )
    # min_chars small to isolate split behaviour (not merge behaviour).
    chunks = semantic_chunk(
        text, mock_embedder, buffer_size=0, breakpoint_percentile=90, min_chars=10
    )
    assert len(chunks) == 2, chunks
    assert "satellite" in chunks[0].lower() and "enzyme" not in chunks[0].lower()
    assert "enzyme" in chunks[1].lower() and "satellite" not in chunks[1].lower()


def test_uniform_text_stays_single_chunk():
    text = (
        "The satellite orbits the planet. The satellite uses solar panels. "
        "The satellite transmits telemetry data home."
    )
    chunks = semantic_chunk(text, mock_embedder, buffer_size=0, min_chars=10)
    assert len(chunks) == 1


def test_post_process_merges_small_and_splits_large():
    # Small chunk merges forward.
    merged = _post_process(["tiny", "a" * 300], min_chars=200, max_chars=1200)
    assert len(merged) == 1
    # Oversized chunk splits on sentence boundaries.
    big = " ".join([f"Sentence number {i} about things." for i in range(200)])
    out = _post_process([big], min_chars=0, max_chars=400)
    assert len(out) > 1
    assert all(len(c) <= 400 + 60 for c in out)  # allow a little slack at boundaries


def test_empty_and_single_sentence():
    assert semantic_chunk("", mock_embedder) == []
    assert semantic_chunk("Only one sentence here.", mock_embedder) == [
        "Only one sentence here."
    ]
