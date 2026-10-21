"""Replay recorded retrieval and scoring without downloading models."""

import math

from .data import ContractError, Document, Query
from .evaluate import METHODS, summarize
from .metrics import answer_metrics, retrieval_metrics
from .retrieval import BM25, cosine_rank, fuse


def verify_report(
    report: dict, documents: tuple[Document, ...], queries: tuple[Query, ...], config: dict
) -> None:
    selected = [query for query in queries if query.split == report["split"]]
    lookup = {query.id: query for query in selected}
    expected = {(query.id, method) for query in selected for method in METHODS}
    records = report["records"]
    observed = {(row["query_id"], row["method"]) for row in records}
    if (
        not selected
        or observed != expected
        or len(records) != len(expected)
        or report["config"] != config
    ):
        raise ContractError("Report coverage or configuration mismatch")
    ids = [doc.id for doc in documents]
    if set(report["document_embeddings"]) != set(ids):
        raise ContractError("Embedding document coverage mismatch")
    vectors = [report["document_embeddings"][identifier] for identifier in ids]
    bm25 = BM25(documents, config["bm25_k1"], config["bm25_b"])
    for row in records:
        query = lookup[row["query_id"]]
        if (
            row["group"] != query.group
            or row["split"] != query.split
            or row["references"] != list(query.answers)
            or row["relevance"] != query.qrels
            or row["question"] != query.question
            or row["answerable"] != bool(query.relevance)
        ):
            raise ContractError("Recorded query differs from dataset")
        ranking = bm25.search(query.question)
        if row["method"] != "bm25":
            dense = cosine_rank(ids, vectors, row["query_embedding"])
            ranking = (
                dense
                if row["method"] == "dense"
                else fuse(
                    [[key for key, _ in dense], [key for key, _ in ranking]], config["rrf_constant"]
                )
            )
        recorded = row["ranking"]
        if len(recorded) != len(ranking) or any(
            item["id"] != key
            or not math.isclose(item["score"], score, rel_tol=1e-12, abs_tol=1e-12)
            for item, (key, score) in zip(recorded, ranking, strict=True)
        ):
            raise ContractError("Recorded retrieval scores do not replay")
        ranked_ids = [key for key, _ in ranking]
        included = row["context_ids"]
        if included != ranked_ids[: len(included)] or len(included) > config["k"]:
            raise ContractError("Context is not a top-k prefix")
        content = {doc.id: doc.content for doc in documents}
        context = "\n".join(f"[{identifier}] {content[identifier]}" for identifier in included)
        if row["prompt"] != config["prompt"].format(context=context, question=query.question):
            raise ContractError("Recorded prompt differs from context")
        metrics = retrieval_metrics(ranked_ids, query.qrels, config["k"])
        metrics["context_recall"] = retrieval_metrics(included, query.qrels, config["k"])["recall"]
        metrics.update(answer_metrics(row["answer"], query.answers))
        if metrics != row["metrics"]:
            raise ContractError("Recorded metrics do not replay")
        if (
            type(row["input_tokens"]) is not int
            or not 0 < row["input_tokens"] <= config["max_input_tokens"]
            or type(row["output_tokens"]) is not int
            or not 0 <= row["output_tokens"] <= config["max_new_tokens"]
        ):
            raise ContractError("Recorded token counts violate budget")
        if any(
            not math.isfinite(row[key]) or row[key] < 0
            for key in ("retrieval_ms", "generation_ms", "total_ms")
        ):
            raise ContractError("Invalid recorded timing")
        if row["total_ms"] + 1e-6 < row["retrieval_ms"] + row["generation_ms"]:
            raise ContractError("Inconsistent recorded timing")
    if summarize(records, config) != report["summary"]:
        raise ContractError("Summary does not replay")


def error_analysis(records: list[dict]) -> list[dict]:
    errors = []
    for row in records:
        if row["metrics"]["exact_match"] == 1:
            continue
        category = (
            "unanswerable_not_abstained"
            if not row["answerable"]
            else "relevant_context_missing"
            if row["metrics"]["context_recall"] == 0
            else "answer_mismatch_with_relevant_context"
        )
        errors.append(
            {
                "query_id": row["query_id"],
                "method": row["method"],
                "category": category,
                "answer": row["answer"],
                "references": row["references"],
                "context_ids": row["context_ids"],
            }
        )
    return errors
