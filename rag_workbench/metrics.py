"""Retrieval metrics, answer metrics, and grouped paired bootstrap intervals."""

import math
import random
import re
import statistics
import string
from collections import Counter


def retrieval_metrics(ranking: list[str], relevance: dict[str, int], k: int) -> dict:
    if type(k) is not int or k < 1 or len(set(ranking)) != len(ranking):
        raise ValueError("Invalid ranking or cutoff")
    if any(type(grade) is not int or grade < 1 for grade in relevance.values()):
        raise ValueError("Relevance grades must be positive integers")
    if not relevance:
        return {"recall": None, "ndcg": None}
    selected = ranking[:k]
    recall = sum(identifier in relevance for identifier in selected) / len(relevance)
    dcg = sum(
        (2 ** relevance.get(identifier, 0) - 1) / math.log2(position + 2)
        for position, identifier in enumerate(selected)
    )
    ideal = sum(
        (2**grade - 1) / math.log2(position + 2)
        for position, grade in enumerate(sorted(relevance.values(), reverse=True)[:k])
    )
    return {"recall": recall, "ndcg": dcg / ideal}


def normalize(answer: str) -> str:
    answer = answer.casefold().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", answer).split())


def answer_metrics(prediction: str, aliases: tuple[str, ...]) -> dict:
    if not aliases:
        raise ValueError("At least one reference answer is required")
    prediction = normalize(prediction)
    scores = []
    for alias in aliases:
        reference = normalize(alias)
        predicted, expected = prediction.split(), reference.split()
        overlap = sum((Counter(predicted) & Counter(expected)).values())
        f1 = 2 * overlap / (len(predicted) + len(expected)) if predicted or expected else 1.0
        scores.append((float(prediction == reference), f1))
    return {
        "exact_match": max(score[0] for score in scores),
        "token_f1": max(score[1] for score in scores),
    }


def percentile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1 or any(not math.isfinite(v) for v in values):
        raise ValueError("Invalid percentile inputs")
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def interval(values: list[float], groups: list[str], samples: int = 2000, seed: int = 1729) -> dict:
    if not values or len(values) != len(groups) or type(samples) is not int or samples < 1:
        raise ValueError("Invalid bootstrap inputs")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Bootstrap values must be finite")
    buckets: dict[str, list[float]] = {}
    for group, value in zip(groups, values, strict=True):
        buckets.setdefault(group, []).append(value)
    keys = sorted(buckets)
    rng = random.Random(seed)
    draws = [
        statistics.mean(value for key in rng.choices(keys, k=len(keys)) for value in buckets[key])
        for _ in range(samples)
    ]
    return {
        "mean": statistics.mean(values),
        "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "queries": len(values),
        "groups": len(keys),
    }


def paired_interval(
    left: list[float], right: list[float], groups: list[str], samples: int = 2000, seed: int = 1729
) -> dict:
    if len(left) != len(right):
        raise ValueError("Paired observations must align")
    return interval([a - b for a, b in zip(left, right, strict=True)], groups, samples, seed)
