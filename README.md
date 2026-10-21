# RAG Evaluation Workbench

Reproducible local comparisons of BM25, dense, and hybrid retrieval with a fixed answer model.

## Problem

Retrieval quality and answer quality measure different failure modes. A useful evaluation must preserve that distinction and hold generation settings constant.

## Scope

Project 02 of the engineering evidence series. An original fictional operations corpus, labeled queries, document-disjoint development and evaluation groups, and a CPU-only experiment with no paid APIs.

## Quick start

The implementation and verified commands will be added in focused commits.

## Usage

The workbench evaluates a selected split and writes a complete run directory.

## Architecture

Validated data flows through three retrievers, one fixed generator, and independent retrieval and answer metrics.

## Reproduction

The experiment uses pinned local models, greedy generation, a fixed prompt and token budget, recorded seeds, and paired comparisons.

## Results

No improvement is assumed in advance. Raw model outputs and failures will be published after measurement.

## Testing

CI checks deterministic contracts without model downloads. Local integration measurements exercise the real models.

## Limitations

Small synthetic data cannot establish performance on production RAG tasks.

## Contributing

Use regression tests and focused changes. Do not tune on the evaluation split.

## License

Code and original data use the [MIT license](LICENSE). Model weights retain their upstream licenses.
