# Evaluation methodology

## Frozen experiment

The dataset, prompt, model commit IDs, and configuration were committed before
running any model evaluations. Development and evaluation share a retrieval
corpus but use disjoint relevant-document groups. The development run checks
execution. No parameters or aliases are selected from evaluation results.

This is an authored synthetic benchmark, not an externally blind evaluation.
The test split is reserved against parameter tuning, not against author knowledge.

## Retrieval

Each document is one retrieval unit. BM25 indexes title plus text with lowercase
ASCII alphanumeric tokenization, no stemming, and no stopword removal. Query
terms are deduplicated. Its Robertson-style IDF uses a positive log-one-plus
form. Dense retrieval uses attention-mask-aware mean pooling of MiniLM hidden
states followed by L2 normalization and cosine similarity. All input tokens must
fit the 256-token embedding budget; no silent embedding truncation is allowed.

Hybrid retrieval uses reciprocal rank fusion across the complete BM25 and dense
rankings with constant 60. Ties break lexicographically by document ID. All
methods retrieve three documents. Corpus embeddings are built once; dense and
hybrid share that immutable index, with construction cost reported separately.
Query embeddings are computed separately for each method's measured request.

## Generation

Every method uses the same pinned FLAN-T5-small weights, prompt, greedy decoding,
32-token output cap, 384-token input cap, CPU device, seed 1729, and four threads.
Documents are packed whole in ranking order until adding the next would exceed
the token budget. A too-large question fails rather than truncates silently.
The report records both top-k retrieval recall and actual included-context recall.

No reference answers or relevance labels are included in prompts. They are used
only after inference for scoring. Model downloads are explicit; evaluation loads
local safetensors with remote code disabled and local-only file resolution.

## Metrics

Recall@3 divides retrieved relevant documents by all relevant documents.
NDCG@3 uses gain `2^grade - 1` and logarithmic rank discount. The included corpus
has binary labels, although metric tests cover graded relevance. Unanswerable
questions have no defined retrieval score and are excluded from its denominator.

Answer exact match compares normalized output against fixed aliases. Normalization
lowercases, removes ASCII punctuation and English articles, and collapses spaces.
Token F1 uses multiset overlap and the best alias. These are surface-form metrics,
not factuality or entailment judgments. Correct paraphrases can be marked wrong.
Answerable and unanswerable exact match are also reported separately.

95% intervals use 2,000 percentile bootstrap resamples of source groups. Both
questions about a document stay together. Method differences use aligned paired
observations with the same group resampling. The intervals describe variability
within this small synthetic sample, not general deployment uncertainty.

## Timing and cost

Models receive a warmup before measurement. Per-query method order rotates to
reduce systematic first-method bias. Retrieval time includes query embedding
and ranking when applicable. Generation time includes tokenization and decoding.
Total request time also includes context packing. Model loading, warmup, and
index construction are outside per-query latency and documented separately.

There is one measured observation per query and method, not a controlled load
test. P50 and P95 are descriptive interpolated percentiles. Hardware contention,
OS scheduling, and CPU libraries affect results. No GPU or paid inference is used.
Paid API cost is zero; electricity cost and embodied hardware cost are unmeasured,
not zero.

## Evidence verification

The report stores document and query embeddings, complete rankings, actual prompts,
raw predictions, aliases, timings, and metrics. Offline verification replays
ranking, prompt construction from included IDs, scoring, and bootstrap intervals.
It checks source hashes and query coverage. It does not rerun the model, authenticate
measurements, independently recount tokenizer tokens, or prove maximal packing.
A full local run is required to reproduce neural inference and timing.
