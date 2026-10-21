# Dataset provenance and split policy

These original fictional station-operation documents and questions are MIT
licensed. Names, codes, policies, and numbers are synthetic. They describe no
real organization or operational security system.

There are 24 documents: six development source groups, twelve evaluation source
groups, and six distractors. Development has 12 questions. Evaluation has 24
answerable questions plus two explicitly unanswerable questions. All documents
are available to every retriever, as in a fixed-corpus retrieval experiment.

Questions in different splits never share a relevant document or group. The
split is reserved for avoiding parameter selection on evaluation results. It is
not an externally blinded holdout: the same project created both splits. No
learned model is trained or fine-tuned on this corpus.

Each question records accepted short-answer aliases and binary relevance labels.
Aliases are fixed before evaluation. Exact match ignores case, punctuation,
articles, and repeated whitespace. Correct paraphrases can still score zero.
Unanswerable questions are excluded from retrieval metrics and scored against
`unknown` for answer metrics. Bootstrap sampling uses source groups, preserving
correlation between the two questions about a document.
