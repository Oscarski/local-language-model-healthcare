# MedMCQA Validation Overlap Audit

CPU-only post-run check using public dataset text and committed split indices.
It is not model evaluation and does not alter recorded experiment outputs.
Pinned source: `openlifescienceai/medmcqa` at revision `91c6572c454088bf71b679ad90aa8dffcd0d5868`.

| Comparison | Exact normalized matches |
|---|---:|
| train_ft vs validation | 1 |
| probe_set vs validation | 0 |

TF-IDF nearest-neighbour entries in the JSON, when present, are candidates
for manual review only; they are not evidence of contamination by themselves.
