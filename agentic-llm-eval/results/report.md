# LLM evaluation report

Rates show `value [95% interval] (n)`. Wilson intervals for rates, bootstrap for means. Denominators include ALL scenarios (missing/unparseable outputs count as failures).

## Sentiment (GLUE SST-2 validation subset)

| model | prune | accuracy (all) | parse-fail | acc on parsed only | lat p50/p95 s | tok/s |
|---|---|---|---|---|---|---|
| baseline-rules | none | 69.0% [59-77] (n=100) | 0.0% | 69.0% [59-77] (n=100) | 0.00 / 0.00 | - |
| llama3.2:3b | none | 91.0% [84-95] (n=100) | 1.0% | 91.9% [85-96] (n=99) | 2.87 / 3.26 | 68.8 |
| qwen2.5-coder:3b | none | 95.0% [89-98] (n=100) | 0.0% | 95.0% [89-98] (n=100) | 2.91 / 3.26 | 65.1 |
| qwen2.5:3b | none | 96.0% [90-98] (n=100) | 1.0% | 97.0% [91-99] (n=99) | 3.04 / 3.38 | 66.6 |

## Question answering (SQuAD 2.0 validation subset)

| model | prune | answerable EM | answerable F1 | abstains correctly | fabrication | false abstain | answer retained | ctx reduction | lat p50/p95 s |
|---|---|---|---|---|---|---|---|---|---|
| baseline-rules | none | 0.0% [0-0] (n=100) | 0.0% [0-0] (n=100) | 100.0% [96-100] (n=100) | 0.0% [0-4] (n=100) | 100.0% [96-100] (n=100) | 100.0% [96-100] (n=100) | 0% | 0.00 / 0.00 |
| llama3.2:3b | none | 63.0% [54-72] (n=100) | 70.7% [62-79] (n=100) | 43.0% [34-53] (n=100) | 57.0% [47-66] (n=100) | 14.0% [9-22] (n=100) | 100.0% [96-100] (n=100) | 0% | 2.87 / 5.70 |
| llama3.2:3b | topk:2 | 54.0% [44-64] (n=100) | 66.2% [57-74] (n=100) | 40.0% [31-50] (n=100) | 60.0% [50-69] (n=100) | 10.0% [6-17] (n=100) | 93.0% [86-97] (n=100) | 60% | 2.77 / 5.41 |
| qwen2.5-coder:3b | none | 78.0% [70-86] (n=100) | 84.6% [78-90] (n=100) | 35.0% [26-45] (n=100) | 65.0% [55-74] (n=100) | 2.0% [1-7] (n=100) | 100.0% [96-100] (n=100) | 0% | 2.94 / 3.31 |
| qwen2.5-coder:3b | topk:2 | 67.0% [58-76] (n=100) | 77.4% [70-85] (n=100) | 37.0% [28-47] (n=100) | 63.0% [53-72] (n=100) | 4.0% [2-10] (n=100) | 93.0% [86-97] (n=100) | 60% | 2.86 / 3.21 |
| qwen2.5:3b | none | 74.0% [66-82] (n=100) | 82.1% [75-88] (n=100) | 54.0% [44-63] (n=100) | 46.0% [37-56] (n=100) | 9.0% [5-16] (n=100) | 100.0% [96-100] (n=100) | 0% | 3.25 / 7.06 |
| qwen2.5:3b | topk:2 | 64.0% [54-73] (n=100) | 77.4% [70-83] (n=100) | 51.0% [41-61] (n=100) | 49.0% [39-59] (n=100) | 9.0% [5-16] (n=100) | 93.0% [86-97] (n=100) | 60% | 3.21 / 10.31 |

## Tool use (synthetic labelled set)

| model | prune | accuracy (all) | acc when tool required | tool used when required | unnecessary calls | lat p50/p95 s |
|---|---|---|---|---|---|---|
| baseline-rules | none | 0.0% [0-9] (n=40) | 0.0% [0-16] (n=20) | 0.0% [0-16] (n=20) | 0.0% [0-16] (n=20) | 0.00 / 0.00 |
| llama3.2:3b | none | 82.5% [68-91] (n=40) | 70.0% [48-85] (n=20) | 100.0% [84-100] (n=20) | 0.0% [0-16] (n=20) | 3.97 / 5.77 |
| qwen2.5-coder:3b | none | 82.5% [68-91] (n=40) | 65.0% [43-82] (n=20) | 65.0% [43-82] (n=20) | 0.0% [0-16] (n=20) | 2.86 / 6.05 |
| qwen2.5:3b | none | 85.0% [71-93] (n=40) | 70.0% [48-85] (n=20) | 100.0% [84-100] (n=20) | 10.0% [3-30] (n=20) | 5.68 / 12.44 |

## Context pruning (paired, same scenarios)

| model | config | n | success none | success pruned | only none right | only pruned right | McNemar p | F1 diff [95%] |
|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | topk:2 | 200 | 53.0% | 47.0% | 27 | 15 | 0.088 | +0.038 [-0.025, +0.100] |
| qwen2.5-coder:3b | topk:2 | 200 | 56.5% | 52.0% | 22 | 13 | 0.175 | +0.026 [-0.024, +0.077] |
| qwen2.5:3b | topk:2 | 200 | 64.0% | 57.5% | 32 | 19 | 0.092 | +0.039 [-0.023, +0.105] |

## Notes
- `fabrication` = answered an unanswerable question (the primary hallucination measure here).
- `answer retained` = gold answer still present in the (pruned) context; low values mean pruning removed the evidence.
- Latency excludes model load (a warm-up call precedes each model). p95 over few scenarios is noisy.
