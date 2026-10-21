"""Deterministic BM25, cosine ranking, and reciprocal rank fusion."""

import math
import re
from collections import Counter

from .data import Document


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def rank(scores: dict[str, float]) -> list[tuple[str, float]]:
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("Ranking scores must be finite")
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class BM25:
    def __init__(self, documents: tuple[Document, ...], k1: float = 1.2, b: float = 0.75):
        if not documents or not math.isfinite(k1) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 corpus or parameters")
        if len({doc.id for doc in documents}) != len(documents):
            raise ValueError("Duplicate document id")
        self.k1, self.b = k1, b
        self.counts = {doc.id: Counter(tokenize(doc.content)) for doc in documents}
        self.lengths = {key: sum(counts.values()) for key, counts in self.counts.items()}
        self.average = sum(self.lengths.values()) / len(documents)
        if self.average == 0:
            raise ValueError("Corpus has no indexable terms")
        self.frequency = Counter(term for counts in self.counts.values() for term in counts)

    def search(self, query: str) -> list[tuple[str, float]]:
        scores = {}
        for identifier, counts in self.counts.items():
            score = 0.0
            for term in set(tokenize(query)):
                frequency = counts[term]
                if not frequency:
                    continue
                df = self.frequency[term]
                idf = math.log1p((len(self.counts) - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * self.lengths[identifier] / self.average
                )
                score += idf * frequency * (self.k1 + 1) / denominator
            scores[identifier] = score
        return rank(scores)


def cosine_rank(
    ids: list[str], vectors: list[list[float]], query: list[float]
) -> list[tuple[str, float]]:
    if not ids or len(ids) != len(vectors) or len(set(ids)) != len(ids) or not query:
        raise ValueError("Invalid embedding matrix")
    if any(len(vector) != len(query) for vector in vectors):
        raise ValueError("Embedding dimensions do not match")
    if any(not math.isfinite(x) for vector in [query, *vectors] for x in vector):
        raise ValueError("Embeddings must be finite")
    norm = math.sqrt(sum(value * value for value in query))
    if norm == 0:
        raise ValueError("Query embedding has zero norm")
    scores = {}
    for identifier, vector in zip(ids, vectors, strict=True):
        length = math.sqrt(sum(value * value for value in vector))
        if length == 0:
            raise ValueError("Document embedding has zero norm")
        scores[identifier] = sum(a * b for a, b in zip(query, vector, strict=True)) / (
            norm * length
        )
    return rank(scores)


def fuse(rankings: list[list[str]], constant: int = 60) -> list[tuple[str, float]]:
    if not rankings or type(constant) is not int or constant < 1:
        raise ValueError("Invalid reciprocal rank fusion inputs")
    scores: dict[str, float] = {}
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("Duplicate document in ranking")
        for position, identifier in enumerate(ranking, 1):
            scores[identifier] = scores.get(identifier, 0.0) + 1 / (constant + position)
    return rank(scores)
