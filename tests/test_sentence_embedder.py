import pytest
from sentence_embedder.sentence_embedder import SentenceEmbedder
from typing import List


@pytest.fixture(scope="module")
def embedder() -> SentenceEmbedder:
    return SentenceEmbedder("distilbert-base-uncased")


def test_embed_sentence(embedder: SentenceEmbedder) -> None:
    sentence = "This is a test sentence."
    embedding = embedder.embed_sentence(sentence)
    assert embedding.shape == (1, 768)


def test_embed_sentences(embedder: SentenceEmbedder) -> None:
    sentences: List[str] = [
        "This is the first sentence.",
        "This is the second sentence.",
    ]
    embeddings = embedder.embed_sentences(sentences)
    assert embeddings.shape == (2, 768)
