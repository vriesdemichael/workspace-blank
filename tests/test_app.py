from collections.abc import Generator
import pytest
from fastapi.testclient import TestClient
from sentence_embedder.app import app

@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


def test_embed_sentence(client: TestClient) -> None:
    response = client.post(
        "/embed_sentence", json={"sentence": "This is a test sentence."}
    )
    assert response.status_code == 200
    assert "embedding" in response.json()


def test_embed_sentences(client: TestClient) -> None:
    response = client.post(
        "/embed_sentences",
        json={
            "sentences": ["This is the first sentence.", "This is the second sentence."]
        },
    )
    assert response.status_code == 200
    assert "embeddings" in response.json()
