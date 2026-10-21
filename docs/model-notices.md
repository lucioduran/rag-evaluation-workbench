# Model and dependency notices

Code, original data, and reports are MIT licensed. Downloaded weights retain
upstream terms and are not included in this repository or its release archive.

| Model | Purpose | License | Pinned revision |
| --- | --- | --- | --- |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384-dimensional embeddings | Apache-2.0 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| [FLAN-T5-small](https://huggingface.co/google/flan-t5-small) | Short-answer generation | Apache-2.0 | `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` |

The reference environment uses Python 3.11.13, PyTorch 2.2.2, Transformers 4.46.3,
and NumPy 1.26.4 on Intel macOS. Exact installed packages are recorded in
requirements-local.lock. This is a reference-environment lock, not a universal
platform lock. Other platforms may require additional PyTorch platform packages.

The core evaluator, statistical functions, and evidence replay require only the
Python standard library. CI installs only the pinned coverage and formatting
tools. Adapter contract tests use mocks; actual neural execution is exercised by
the published local runs. CI does not claim to repeat neural inference.

Model cards describe training, intended uses, and limitations. Model selection
here targets a small free local experiment, not a recommendation for production.
