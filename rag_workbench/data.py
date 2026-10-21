"""Strict input contracts and split isolation."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


class ContractError(ValueError):
    pass


def object_fields(value, fields: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ContractError(f"{label}: expected fields {sorted(fields)}")
    return value


def text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label}: expected nonempty text")
    if any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF or c == chr(0x2014) for c in value):
        raise ContractError(f"{label}: unsupported character")
    return value


def identifier(value) -> str:
    value = text(value, "id")
    if not re.fullmatch(r"[a-z][a-z0-9-]*", value):
        raise ContractError("id: expected a lowercase identifier")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"Cannot read {path}: {error}") from error


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str

    @property
    def content(self) -> str:
        return f"{self.title}. {self.text}"


@dataclass(frozen=True)
class Query:
    id: str
    split: str
    group: str
    question: str
    answers: tuple[str, ...]
    relevance: tuple[tuple[str, int], ...]

    @property
    def qrels(self) -> dict[str, int]:
        return dict(self.relevance)


def load_dataset(corpus: Path, queries: Path) -> tuple[tuple[Document, ...], tuple[Query, ...]]:
    raw = object_fields(read_json(corpus), {"schema_version", "license", "documents"}, "corpus")
    labels = object_fields(read_json(queries), {"schema_version", "queries"}, "queries")
    for envelope, field in ((raw, "documents"), (labels, "queries")):
        if type(envelope["schema_version"]) is not int or envelope["schema_version"] != 1:
            raise ContractError("Unsupported schema version")
        if not isinstance(envelope[field], list) or not envelope[field]:
            raise ContractError(f"{field}: expected nonempty list")
    if raw["license"] != "MIT":
        raise ContractError("Corpus license must be MIT")
    documents = []
    for value in raw["documents"]:
        value = object_fields(value, {"id", "title", "text"}, "document")
        documents.append(
            Document(
                identifier(value["id"]), text(value["title"], "title"), text(value["text"], "text")
            )
        )
    ids = {doc.id for doc in documents}
    if len(ids) != len(documents):
        raise ContractError("Duplicate document id")
    parsed = []
    group_splits = {}
    document_splits = {}
    seen_questions = set()
    for value in labels["queries"]:
        value = object_fields(
            value, {"id", "split", "group", "question", "answers", "relevance"}, "query"
        )
        if value["split"] not in ("dev", "eval"):
            raise ContractError("Unknown split")
        if not isinstance(value["answers"], list) or not value["answers"]:
            raise ContractError("Expected accepted answer aliases")
        relevance = value["relevance"]
        if not isinstance(relevance, dict) or any(
            doc not in ids or type(grade) is not int or not 1 <= grade <= 3
            for doc, grade in relevance.items()
        ):
            raise ContractError("Invalid relevance labels")
        query = Query(
            identifier(value["id"]),
            value["split"],
            identifier(value["group"]),
            text(value["question"], "question"),
            tuple(text(a, "answer") for a in value["answers"]),
            tuple(sorted(relevance.items())),
        )
        if not query.relevance and query.answers != ("unknown",):
            raise ContractError("Unanswerable queries must use only the unknown answer")
        normalized = " ".join(query.question.casefold().split())
        if normalized in seen_questions:
            raise ContractError("Duplicate question")
        seen_questions.add(normalized)
        for mapping, keys in ((group_splits, [query.group]), (document_splits, relevance)):
            for key in keys:
                if key in mapping and mapping[key] != query.split:
                    raise ContractError("Development and evaluation overlap")
                mapping[key] = query.split
        parsed.append(query)
    if len({query.id for query in parsed}) != len(parsed):
        raise ContractError("Duplicate query id")
    if {query.split for query in parsed} != {"dev", "eval"}:
        raise ContractError("Both development and evaluation splits are required")
    return tuple(documents), tuple(parsed)
