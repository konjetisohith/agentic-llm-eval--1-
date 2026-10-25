# Interview notes (fill in numbers only from YOUR real runs)

**30 seconds:** I built a harness that runs local LLMs through the same tool-using agent on sentiment, QA with
unanswerable questions, and tool use, and reports accuracy with confidence intervals and non-LLM baselines.
I rebuilt the metrics after finding that my first version counted execution as success.

**Questions you should be able to answer**
1. *What does "success" mean?* Correct answer; completion is reported separately. Failed outputs stay in the denominator.
2. *Why a baseline?* 76% on SST-2 means little if VADER or chance gets close.
3. *How do you score unanswerable questions?* Correct only if the model abstains; fabrication rate = answers given.
4. *Why exact match is low for LLMs:* verbose spans, normalization, abstaining wrongly. I report F1 and false-abstain separately.
5. *Is the hallucination metric valid?* It is a proxy: fabrication on unanswerable + extractive groundedness. Not general.
6. *Is pruning fairly compared?* Same scenarios, temperature 0, seed, paired McNemar test, plus answer-retention.
7. *Why is latency comparable?* Warm-up first, load time excluded, same machine, p50/p95, tokens/s. Still depends on model size.
8. *Is it agentic?* Small ReAct loop: tool choice, observation, bounded repair, termination. Not planning/multi-agent.
9. *How do you know the metrics are right?* Known-answer tests, official SQuAD normalization, and a mutation check on the denominator bug.
10. *Limits:* small n, synthetic tool set, one tool, quantized 3B models.

**Hard follow-ups:** What if your 3B model is below the always-abstain baseline on overall SQuAD 2.0? (Then say so; explain the tradeoff
between abstention and answering.) Did pruning drop answers from context? (Check answer retention.) Why trust one seed? (You shouldn't; rerun with seeds.)

**Resume line (use only after you have real results):**
"Built a reproducible local-LLM evaluation platform (Ollama; Qwen2.5-3B, Llama3.2-3B, Qwen2.5-Coder-3B) over SST-2, SQuAD 2.0 and a labelled tool-use set,
with confidence intervals, non-LLM baselines, paired context-pruning tests and a Streamlit dashboard; 32 automated tests."
