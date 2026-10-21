# RAG Evaluation Workbench

[![CI](https://github.com/lucioduran/rag-evaluation-workbench/actions/workflows/ci.yml/badge.svg)](https://github.com/lucioduran/rag-evaluation-workbench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Compare BM25, dense, and hybrid retrieval with one fixed local answer model. Inspect every ranking, prompt, prediction, metric, and confidence interval.

## Problem

Finding the right document does not guarantee the right answer. Retrieval and generation need separate measurements under equivalent conditions. This workbench makes that distinction visible and preserves the evidence needed to check it.

## Scope

Project 02 of the engineering evidence series. The included experiment uses 24 original fictional station-operation documents, 12 development questions, and 26 evaluation questions. Development and evaluation have disjoint source-document groups. Two evaluation questions are intentionally unanswerable.

The real experiment runs on a local CPU with pinned MiniLM embeddings and FLAN-T5-small answers. No paid API, hosted inference, rented GPU, or account is required. [Dataset provenance](data/README.md) and [model notices](docs/model-notices.md) describe licenses and boundaries.

## Quick start

With Python 3.11.13, inspect the recorded experiment without installing model libraries:

```sh
git clone https://github.com/lucioduran/rag-evaluation-workbench.git
cd rag-evaluation-workbench
python -m unittest discover -v
python -m rag_workbench verify --report results/eval/report.json
```

Verification reconstructs all rankings from saved embeddings, rebuilds prompts from recorded context IDs, and recomputes metrics and intervals. It does not rerun neural inference. See Reproduction for the full local experiment.

## Usage

| Command | Behavior |
| --- | --- |
| `download` | Explicitly downloads the two pinned public models |
| `run --split dev` | Runs the development questions with local model files |
| `run --split eval` | Runs the reserved evaluation questions |
| `verify --report PATH` | Replays retrieval and scoring using only the standard library |

Every run requires a new output directory and writes `trace.jsonl`, `report.json`, `summary.json`, and `errors.json`. The trace is flushed after each inference. A failed run can leave a partial trace, but no successful report is claimed. Use a new directory for retries.

```sh
python -m rag_workbench run --split dev --output build/my-dev-run
python -m rag_workbench run --split eval --output build/my-eval-run
python -m rag_workbench verify --report build/my-eval-run/report.json
```

`--corpus`, `--queries`, `--config`, and `--cache` accept alternate paths. Input contracts reject unknown fields, duplicate IDs, invalid relevance grades, and overlap between split groups. Exit code `0` means success; `2` means input, environment, or verification failure. Evaluation loads model files locally and never requests hosted inference.

## Architecture

```text
Licensed corpus + labeled split + frozen configuration
                         |
               validated document model
                         |
          +--------------+--------------+
          |              |              |
         BM25       MiniLM cosine    reciprocal rank fusion
          +--------------+--------------+
                         |
            whole-document context budget
                         |
              fixed local FLAN-T5-small
                         |
          retrieval metrics + answer metrics
                         |
          grouped bootstrap + paired deltas
                         |
          raw report + errors + offline replay
```

Every method uses the same corpus, top-three cutoff, answer model, prompt template, input budget, output budget, and decoding settings. Method execution order rotates across questions. No labels or reference answers enter the prompt.

See [methodology](docs/methodology.md) for formulas, timing boundaries, context packing, and statistical interpretation.

## Reproduction

The measured reference environment is Intel macOS, Python 3.11.13, four CPU threads, and the packages in [requirements-local.lock](requirements-local.lock). It is a reference-environment lock, not a universal platform lock.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-local.lock
python -m rag_workbench download
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false python -m rag_workbench run --split dev --output build/reproduced-dev
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false python -m rag_workbench run --split eval --output build/reproduced-eval
python -m rag_workbench verify --report build/reproduced-eval/report.json
```

Downloads require internet access and several hundred megabytes of local storage for weights. Inference runs offline. Other platforms may need compatible PyTorch platform packages; their runtime versions and outcomes must be recorded separately. The standard-library tests and report replay are verified by CI on Linux, macOS, and Windows.

Frozen configuration: seed 1729, BM25 `k1=1.2` and `b=0.75`, RRF constant 60, `k=3`, 256 embedding input tokens, 384 generation input tokens, 32 output tokens, greedy decoding, and 2,000 grouped bootstrap resamples. No tuning followed the evaluation run.

Each raw report records input hashes, model file hashes, exact model revisions, package versions, CPU identity, thread count, OS, timestamp, and implementation revision. Host timings can vary. Zero paid API cost does not mean zero electricity or hardware cost; those costs are unmeasured.

## Results

Real CPU evaluation on an Intel Xeon W-3223. Retrieval metrics use 24 answerable queries; answer metrics use all 26 queries. Intervals are 95% grouped percentile bootstrap intervals, not guarantees about production performance.

| Method | Recall@3 | NDCG@3 | Exact match | Token F1 | Request P50 / P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 1.000 | 0.969 | 0.731 | 0.756 | 93.7 / 133.4 ms |
| Dense | 1.000 | 1.000 | 0.769 | 0.795 | 105.9 / 138.8 ms |
| Hybrid | 1.000 | 1.000 | 0.731 | 0.767 | 105.9 / 144.4 ms |

| Exact-match comparison | Mean | 95% interval |
| --- | ---: | --- |
| BM25 | 0.731 | [0.560, 0.885] |
| Dense | 0.769 | [0.600, 0.893] |
| Hybrid | 0.731 | [0.560, 0.885] |
| Dense minus BM25, paired | +0.038 | [0.000, 0.120] |
| Hybrid minus BM25, paired | 0.000 | [0.000, 0.000] |

All methods retrieved relevant context for every answerable query, yet generated answer errors remained. All methods failed both abstention questions. Dense has one more exact-match success than BM25 here, but its paired interval includes zero. This does not establish a reliable or general winner.

Some exact-match failures are scoring limitations: `90 requests per minute` is a correct response excluded by the frozen short aliases. Aliases were not amended after observing outputs. Other failures are substantive, including confusing the network enrollment code with the door access code.

- [Evaluation raw report](results/eval/report.json), [summary](results/eval/summary.json), and [all errors](results/eval/errors.json)
- [Development raw report](results/dev/report.json), [summary](results/dev/summary.json), and [all errors](results/dev/errors.json)
- [Error analysis](docs/error-analysis.md)

## Testing

```sh
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m coverage run -m unittest discover -v
python -m coverage report
python -m rag_workbench verify --report results/dev/report.json
python -m rag_workbench verify --report results/eval/report.json
```

Tests cover hand-calculated BM25, cosine similarity, rank fusion, graded NDCG, recall, answer aliases, token F1, grouped paired intervals, invalid inputs, split isolation, context budgets, offline model-loading contracts, CLI behavior, and tampered evidence. Repository tests enforce the shared README structure and absence of em dashes.

CI requires at least 90% combined statement/branch coverage and repeats report verification on three operating systems. It installs no model libraries. Adapter tests use mocks; real neural execution is demonstrated separately by the recorded local runs.

## Limitations

- This small synthetic corpus is easy for retrieval and cannot establish production generalization. The evaluation split is author-known, not externally blind.
- Exact match and token F1 do not measure factuality. Fixed aliases miss some valid paraphrases. Manual analysis does not alter the frozen scores.
- The small answer model is poor at abstention here. Successful retrieval must not be reported as successful answering.
- Bootstrap intervals have only 12 answerable source groups and two unanswerable groups. Degenerate intervals do not prove certainty outside this dataset.
- One timing observation per query and method is not a load test. Model loading and index construction are reported separately from request latency. Hardware and backend changes can affect predictions as well as timing.
- Replay validates internal consistency, not authenticity of measurements or neural inference. Token counts are checked against limits, not independently recounted without the tokenizer.
- Inputs and outputs are fully loaded into memory. This research CLI expects trusted local files and is not an unrestricted upload service.
- The native model stack is pinned to the measured Intel macOS environment. Cross-platform CI validates the portable evaluator and evidence replay, not identical neural inference on every OS.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Keep changes focused, add regression tests, and version changed experiments separately. Never tune on the evaluation split or overwrite inconvenient outcomes. Report vulnerabilities through [SECURITY.md](SECURITY.md).

## License

Code, original fictional corpus, labels, documentation, and recorded results use the [MIT license](LICENSE). Downloaded models are Apache-2.0 and are not redistributed here. Dependencies retain their upstream licenses; see [model notices](docs/model-notices.md).
