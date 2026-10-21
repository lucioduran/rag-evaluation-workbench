"""Fair retrieval comparisons with a shared generation contract."""

import time
from collections import defaultdict

from .data import Document, Query
from .local_models import Backend
from .metrics import answer_metrics, interval, paired_interval, percentile, retrieval_metrics
from .retrieval import BM25, cosine_rank, fuse

METHODS = ("bm25", "dense", "hybrid")
METRICS = ("recall", "ndcg", "context_recall", "exact_match", "token_f1")


def make_prompt(
    query: Query, ranking: list[str], documents: dict[str, Document], backend: Backend, config: dict
) -> tuple[str, list[str]]:
    template = config["prompt"]
    prompt = template.format(context="", question=query.question)
    if backend.count_tokens(prompt) > config["max_input_tokens"]:
        raise ValueError("Question and template exceed the input budget")
    included = []
    context = []
    for identifier in ranking[: config["k"]]:
        candidate = context + [f"[{identifier}] {documents[identifier].content}"]
        proposed = template.format(context="\n".join(candidate), question=query.question)
        if backend.count_tokens(proposed) > config["max_input_tokens"]:
            break
        included.append(identifier)
        context = candidate
        prompt = proposed
    return prompt, included


def summarize(records: list[dict], config: dict) -> dict:
    by_method = {
        method: [record for record in records if record["method"] == method] for method in METHODS
    }
    ids = [[row["query_id"] for row in rows] for rows in by_method.values()]
    if (
        not ids[0]
        or any(sequence != ids[0] for sequence in ids[1:])
        or len(set(ids[0])) != len(ids[0])
    ):
        raise ValueError("Methods must have aligned unique query records")
    result = {"methods": {}, "paired_differences": {}}
    for method, rows in by_method.items():
        measured = {}
        for metric in METRICS:
            available = [row for row in rows if row["metrics"][metric] is not None]
            measured[metric] = (
                interval(
                    [row["metrics"][metric] for row in available],
                    [row["group"] for row in available],
                    config["bootstrap_samples"],
                    config["seed"],
                )
                if available
                else None
            )
        for timing in ("retrieval_ms", "generation_ms", "total_ms"):
            values = [row[timing] for row in rows]
            measured[timing] = {"p50": percentile(values, 0.5), "p95": percentile(values, 0.95)}
        measured["input_tokens"] = sum(row["input_tokens"] for row in rows)
        measured["output_tokens"] = sum(row["output_tokens"] for row in rows)
        measured["answerable_exact_match"] = _subset_mean(rows, True)
        measured["unanswerable_exact_match"] = _subset_mean(rows, False)
        measured["paid_api_cost_usd"] = 0
        measured["electricity_cost_usd"] = None
        result["methods"][method] = measured
    for method in ("dense", "hybrid"):
        deltas = {}
        for metric in METRICS:
            pairs = [
                (left, right)
                for left, right in zip(by_method[method], by_method["bm25"], strict=True)
                if left["metrics"][metric] is not None and right["metrics"][metric] is not None
            ]
            deltas[metric] = (
                paired_interval(
                    [left["metrics"][metric] for left, _ in pairs],
                    [right["metrics"][metric] for _, right in pairs],
                    [left["group"] for left, _ in pairs],
                    config["bootstrap_samples"],
                    config["seed"],
                )
                if pairs
                else None
            )
        result["paired_differences"][f"{method}-bm25"] = deltas
    return result


def _subset_mean(rows: list[dict], answerable: bool) -> float | None:
    selected = [row["metrics"]["exact_match"] for row in rows if row["answerable"] == answerable]
    return sum(selected) / len(selected) if selected else None


def run(
    documents: tuple[Document, ...],
    queries: tuple[Query, ...],
    backend: Backend,
    config: dict,
    on_record=None,
) -> dict:
    if not queries:
        raise ValueError("No queries selected")
    start = time.perf_counter()
    bm25 = BM25(documents, config["bm25_k1"], config["bm25_b"])
    bm25_index_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    vectors = backend.embed([document.content for document in documents])
    dense_index_ms = (time.perf_counter() - start) * 1000
    ids = [document.id for document in documents]
    lookup = {document.id: document for document in documents}
    records_by_method = defaultdict(list)
    for number, query in enumerate(queries):
        order = METHODS[number % 3 :] + METHODS[: number % 3]
        for order_index, method in enumerate(order):
            start = time.perf_counter()
            query_vector = None
            if method == "bm25":
                ranking = bm25.search(query.question)
            else:
                query_vector = backend.embed([query.question])[0]
                ranking = cosine_rank(ids, vectors, query_vector)
                if method == "hybrid":
                    ranking = fuse(
                        [
                            [key for key, _ in ranking],
                            [key for key, _ in bm25.search(query.question)],
                        ],
                        config["rrf_constant"],
                    )
            retrieval_ms = (time.perf_counter() - start) * 1000
            ranked_ids = [key for key, _ in ranking]
            prompt, included = make_prompt(query, ranked_ids, lookup, backend, config)
            generation_start = time.perf_counter()
            generated = backend.generate(prompt)
            generation_ms = (time.perf_counter() - generation_start) * 1000
            total_ms = (time.perf_counter() - start) * 1000
            metrics = retrieval_metrics(ranked_ids, query.qrels, config["k"])
            metrics["context_recall"] = retrieval_metrics(included, query.qrels, config["k"])[
                "recall"
            ]
            metrics.update(answer_metrics(generated["answer"], query.answers))
            record = {
                "query_id": query.id,
                "group": query.group,
                "split": query.split,
                "question": query.question,
                "method": method,
                "execution_position": order_index,
                "answerable": bool(query.relevance),
                "references": list(query.answers),
                "relevance": query.qrels,
                "query_embedding": query_vector,
                "ranking": [{"id": identifier, "score": score} for identifier, score in ranking],
                "context_ids": included,
                "prompt": prompt,
                **generated,
                "metrics": metrics,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
            }
            records_by_method[method].append(record)
            if on_record is not None:
                on_record(record)
    records = [record for method in METHODS for record in records_by_method[method]]
    return {
        "document_embeddings": dict(zip(ids, vectors, strict=True)),
        "records": records,
        "summary": summarize(records, config),
        "indexing_ms": {
            "bm25": bm25_index_ms,
            "dense": dense_index_ms,
            "hybrid_shared_index_total": bm25_index_ms + dense_index_ms,
        },
    }
