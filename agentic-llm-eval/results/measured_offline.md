# Measured offline on the REAL datasets (no LLM involved)

These numbers need no Ollama, so they were computed in the dev sandbox. LLM results are not here yet.

Data: manifest hashes sst2=37f20a329eadc12c, squad=f784d0842a76e598, tools=cf3ac9e9b0f197e0; seed 42.

## Non-LLM baselines on the benchmark subsets

| baseline | task | result |
|---|---|---|
| VADER lexicon | SST-2 (n=100) accuracy | 69.0% [59-77] (n=100) |
| always-abstain | SQuAD 2.0 (n=200) overall exact | 50.0% [43-57] (n=200) |
| always-abstain | unanswerable correct (n=100) | 100.0% [96-100] (n=100) |
| always-abstain | answerable EM (n=100) | 0.0% [0-0] (n=100) |

An LLM must beat 50% overall on this SQuAD 2.0 mix (and ~69% on this SST-2 subset) to show it adds anything over trivial rules.

## Context pruning: how often the gold answer survives (model-independent)

Top-k TF-IDF sentence selection. Subset = the 100 answerable benchmark questions; full = all 5928 answerable SQuAD 2.0 dev questions.

| k | retained (subset) | retained (full dev) [95%] | context removed (full dev) |
|---|---|---|---|
| 1 | 86.0% | 81.0% [80-82] | 75% |
| 2 | 93.0% | 92.3% [92-93] | 54% |
| 3 | 96.0% | 96.4% [96-97] | 35% |
| 4 | 98.0% | 98.3% [98-99] | 20% |

Retention is an upper bound on answerable-question accuracy under pruning: if the answer is removed, no model can extract it.
