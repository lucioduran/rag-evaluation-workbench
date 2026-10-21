# Evaluation error analysis

The fixed evaluation has 26 questions and three methods, giving 78 model answers.
Twenty answers fail normalized exact match. Fourteen failures have the labeled
relevant document in the actual prompt; six concern unanswerable questions.
There are no answerable evaluation cases with missing relevant context.

These categories are mechanical diagnostics, not causal proofs. The raw prompts,
outputs, references, and rankings remain available in results/eval/report.json.

## Confusing related facts

For `eval-iris-1`, BM25 and dense return `7319`, the network enrollment code,
instead of the offline door code `5826`. Both prompts contain the correct source
and a related distractor. Dense ranks the correct document first but still fails,
showing why retrieval rank cannot substitute for answer evaluation.

For `eval-nova-1`, the model returns `observation team` rather than the on-call
navigator. Both roles occur in the relevant document, but belong to different
alert severities. More accurate retrieval alone would not remove this ambiguity.

## Surface-form scoring errors

For `eval-quartz-1`, BM25 returns `90 requests per minute`. This is semantically
correct, but the frozen aliases include only `90` and `90 requests`. Its exact
match is therefore zero, with partial token F1. This is an annotation limitation.

For `eval-orbit-2`, dense returns `revokes`, which captures the action but differs
from the frozen aliases. Whether that short fragment is sufficient is a separate
human judgment. It remains zero in the deterministic exact-match report.

The aliases were not expanded after seeing evaluation predictions. A future
benchmark version could improve annotation before a new reserved evaluation.

## Abstention failures

Neither unanswerable question produces `unknown` with any retriever. Weather
answers include unsupported temperatures. The fictional captain-phone question
produces unrelated numbers, including synthetic codes from other documents.
No real phone number or private information is present in this corpus.

Abstention exact match is 0/2 for every method. This small sample demonstrates
specific failures; it does not estimate a general hallucination rate.

## What the comparison supports

Dense gets one additional exact-match success compared with BM25, while hybrid
has the same per-query exact-match outcomes as BM25. The paired dense-minus-BM25
95% interval includes zero. The experiment supports separate retrieval and answer
diagnostics, not a broad claim that dense or hybrid retrieval is superior.
