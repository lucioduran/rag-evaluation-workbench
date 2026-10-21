"""Validate the frozen experiment settings."""

import math
import re
from pathlib import Path
from string import Formatter

from .data import ContractError, object_fields, read_json

FIELDS = {
    "schema_version",
    "embedding_model",
    "answer_model",
    "k",
    "bm25_k1",
    "bm25_b",
    "rrf_constant",
    "max_input_tokens",
    "max_new_tokens",
    "embedding_max_tokens",
    "threads",
    "seed",
    "bootstrap_samples",
    "prompt",
}


def load_config(path: Path) -> dict:
    config = object_fields(read_json(path), FIELDS, "config")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ContractError("Unsupported config version")
    for field in ("embedding_model", "answer_model"):
        model = object_fields(config[field], {"id", "revision"}, field)
        if not isinstance(model["id"], str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", model["id"]):
            raise ContractError("Invalid model id")
        if not isinstance(model["revision"], str) or not re.fullmatch(
            "[0-9a-f]{40}", model["revision"]
        ):
            raise ContractError("Models must use full commit revisions")
    for field in (
        "k",
        "rrf_constant",
        "max_input_tokens",
        "max_new_tokens",
        "embedding_max_tokens",
        "threads",
        "bootstrap_samples",
    ):
        if type(config[field]) is not int or config[field] < 1:
            raise ContractError(f"{field}: expected positive integer")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ContractError("Invalid seed")
    for field in ("bm25_k1", "bm25_b"):
        if type(config[field]) not in (int, float) or not math.isfinite(config[field]):
            raise ContractError("Invalid BM25 parameter")
    if config["bm25_k1"] <= 0 or not 0 <= config["bm25_b"] <= 1:
        raise ContractError("Invalid BM25 parameter range")
    prompt = config["prompt"]
    if not isinstance(prompt, str) or chr(0x2014) in prompt:
        raise ContractError("Invalid prompt")
    try:
        fields = [
            field for _, field, fmt, conversion in Formatter().parse(prompt) if field is not None
        ]
        if sorted(fields) != ["context", "question"]:
            raise ValueError
        prompt.format(context="context", question="question")
    except (ValueError, KeyError, IndexError, AttributeError) as error:
        raise ContractError(
            "Prompt must contain exactly context and question placeholders"
        ) from error
    return config
