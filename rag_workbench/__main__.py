"""Download free models, run offline evaluation, or verify recorded evidence."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .config import load_config
from .data import ContractError, load_dataset, read_json, sha256
from .evaluate import run
from .local_models import LocalModels, download_models, model_path
from .report import error_analysis, verify_report


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def provenance(config: dict, cache: Path, inputs: dict[str, Path]) -> dict:
    files = {}
    for field in ("embedding_model", "answer_model"):
        path = model_path(cache, config[field]["id"], config[field]["revision"])
        files[field] = {p.name: sha256(p) for p in sorted(path.iterdir()) if p.is_file()}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    )
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "numpy",
            "tokenizers",
            "huggingface-hub",
            "safetensors",
        )
    }
    processor = platform.processor()
    if sys.platform == "darwin":
        processor = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return {
        "revision": revision,
        "tracked_worktree_dirty": dirty,
        "recorded_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": processor,
        "logical_cpus": os.cpu_count(),
        "device": "cpu",
        "threads": config["threads"],
        "packages": packages,
        "model_files_sha256": files,
        "inputs_sha256": {name: sha256(path) for name, path in inputs.items()},
        "external_inference_calls": 0,
        "paid_api_cost_usd": 0,
        "electricity_cost_usd": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local RAG evaluation without paid APIs.")
    parser.add_argument("command", choices=("download", "run", "verify"))
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus.json"))
    parser.add_argument("--queries", type=Path, default=Path("data/queries.json"))
    parser.add_argument("--config", type=Path, default=Path("config/experiment.json"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/models"))
    parser.add_argument("--split", choices=("dev", "eval"), default="dev")
    parser.add_argument("--output", type=Path, default=Path("build/run"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "download":
            download_models(config, args.cache)
            print("Pinned model files downloaded. Evaluation runs offline.")
            return 0
        documents, queries = load_dataset(args.corpus, args.queries)
        inputs = {"corpus": args.corpus, "queries": args.queries, "config": args.config}
        if args.command == "verify":
            if args.report is None:
                raise ContractError("verify requires --report")
            report = read_json(args.report)
            if report["provenance"]["inputs_sha256"] != {
                name: sha256(path) for name, path in inputs.items()
            }:
                raise ContractError("Input hashes differ from recorded evidence")
            verify_report(report, documents, queries, config)
            print("Recorded rankings, prompts, metrics, and intervals verified.")
            return 0
        if args.output.exists():
            raise ContractError("Output exists; choose a new run directory")
        args.output.mkdir(parents=True)
        initialization_start = time.perf_counter()
        backend = LocalModels(config, args.cache)
        initialization_seconds = time.perf_counter() - initialization_start
        backend.embed(["Warmup passage."])
        backend.generate(
            config["prompt"].format(
                context="The warmup answer is ready.", question="What is the warmup answer?"
            )
        )
        selected = tuple(query for query in queries if query.split == args.split)
        with (args.output / "trace.jsonl").open("w", encoding="utf-8") as trace:

            def record(row):
                trace.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                trace.flush()
                print(
                    f"{row['query_id']} {row['method']}: EM={row['metrics']['exact_match']}",
                    flush=True,
                )

            report = run(documents, selected, backend, config, on_record=record)
        report.update(
            {
                "schema_version": 1,
                "split": args.split,
                "config": config,
                "provenance": provenance(config, args.cache, inputs),
                "initialization_seconds": initialization_seconds,
            }
        )
        verify_report(report, documents, queries, config)
        write_json(args.output / "report.json", report)
        write_json(args.output / "errors.json", error_analysis(report["records"]))
        write_json(args.output / "summary.json", report["summary"])
        print(f"Completed {len(selected)} queries across three methods.")
        return 0
    except (ContractError, ValueError, OSError, KeyError, TypeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
