from fastapi import FastAPI
from pydantic import BaseModel
from sentence_embedder.sentence_embedder import SentenceEmbedder
from typing import List

app = FastAPI()
embedder = SentenceEmbedder(model_name="bert-base-uncased")


class SentenceRequest(BaseModel):
    sentence: str


class SentencesRequest(BaseModel):
    sentences: List[str]


class SentenceEmbeddingResponse(BaseModel):
    embedding: List[float]


class SentencesEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]


@app.post("/embed_sentence", response_model=SentenceEmbeddingResponse)
def embed_sentence(request: SentenceRequest) -> SentenceEmbeddingResponse:
    embedding = embedder.embed_sentence(request.sentence)
    return SentenceEmbeddingResponse(embedding=embedding.tolist())


@app.post("/embed_sentences", response_model=SentencesEmbeddingResponse)
def embed_sentences(request: SentencesRequest) -> SentencesEmbeddingResponse:
    embeddings = embedder.embed_sentences(request.sentences)
    return SentencesEmbeddingResponse(embeddings=embeddings.tolist())


