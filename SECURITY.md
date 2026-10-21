# Security

Report vulnerabilities privately to email@lucioduran.com with a minimal synthetic
reproduction, affected revision, and expected behavior. Do not include secrets.

Version 0.1.x is supported. Run this research tool with trusted local inputs.
It is not a multi-tenant upload service. Model weights are downloaded only through
an explicit command, pinned by commit, and loaded from safetensors with remote
code disabled. Do not place private data in public experiment outputs: prompts
and answers are deliberately recorded in full.
