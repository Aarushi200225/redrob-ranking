# Build Journey: Redrob Ranking System

## India Runs Hackathon, Track 1: Intelligent Candidate Discovery

---

## Where It Started

The problem statement was deceptively simple: rank 100 candidates from 100,000 for a Senior AI Engineer role. Most teams would reach for TF-IDF or BM25, score by keyword overlap, call it done. That approach works. It would produce a ranking. But it would also fail at every interesting edge case the dataset was probably designed to test.

Before writing any code, I spent time reading the job description carefully. Not scanning it for keywords. Actually reading it. A few things stood out:

The JD explicitly says it does not want consulting-only careers. It says "not looking for IT services / consulting profiles." Most keyword-based systems would rank a TCS engineer with 8 years of FAISS experience highly. This system needed to know that was a disqualifier, not a positive signal.

The JD talks about "scrappy product-engineering attitude," "ships over researches," and wanting someone who has "learned from real users." These are culture signals, not skill requirements. A system that only matches technical keywords misses half the picture.

The JD mentions honeypots. The problem spec says the dataset contains approximately 80 logically impossible profiles designed to fool naive systems. Getting caught with honeypots in the top 100 means disqualification.

These three observations drove the entire architecture. This needed to be a system that could model role fit, culture fit, and data integrity simultaneously, not just count keyword matches.

---

## The First Architecture Decision: Hybrid Retrieval

The first real technical decision was retrieval strategy.

**Pure keyword search (BM25):** Catches exact technical matches but misses semantic similarity. A candidate who writes "dense retrieval systems" might not say "embeddings" explicitly even though they mean the same thing.

**Pure semantic search (dense embedding):** Catches semantic similarity but misses exact requirements. If the JD specifically says FAISS, a candidate with FAISS experience should rank above one who has a different vector database, and a simple embedding similarity score might not capture this.

**Hybrid with RRF:** Run both independently, combine the ranked lists using Reciprocal Rank Fusion. Candidates who appear high in both lists score highest. This is the pattern used in production search systems. It requires no training data to calibrate the fusion, which was important since I had no labeled "good candidate" examples.

The RRF formula is: for each candidate, sum 1/(60 + rank) across all ranking streams. The constant 60 is empirically validated across IR benchmarks and works well without tuning.

---

## Building the Six-Stream RRF

I realised early that "semantic similarity to the JD" is not one thing. Technical skill match, experience quality match, and culture/vibe match are genuinely different retrieval dimensions. A candidate with very high technical similarity might have low experience similarity (junior engineer who knows all the tools). Running separate query vectors for each dimension lets each one surface its best candidates independently.

The six streams ended up as:
- Q1: Primary query, blending JD text (40%) with a HyDE-generated ideal candidate profile (60%)
- Q2: Technical requirements query
- Q3: Experience signals query
- Q4: Culture and vibe signals query
- BM25_A: Raw JD terms
- BM25_B: Expanded taxonomy terms (~102 terms from skill taxonomy)

**The HyDE decision** was the most interesting. HyDE (Hypothetical Document Embedding) generates a fake "ideal candidate" profile written in first-person candidate vocabulary. The JD says "we need someone who has shipped retrieval systems." A strong candidate's profile says "I led development of a FAISS-based search system serving 2M daily users." These describe the same thing but share almost no words. HyDE bridges this by generating a synthetic candidate profile that uses the same vocabulary as real candidates, then blending it into the primary query vector.

---

## The RRF Bug That Stayed Hidden

Early in building Stage 3, there was a bug in the RRF implementation. Instead of accumulating independent stream contributions, the code was overwriting a single variable on each stream iteration. So only the last stream was contributing to the final score. The other five streams were being silently discarded.

The bug was not immediately obvious because the system still produced a ranking. It just was not doing 6-stream fusion. It was doing 1-stream fusion with the last processed stream.

I caught it when examining why the retrieval pool was less diverse than expected. Candidates surfaced by Q2 (technical requirements) were not appearing in the final pool even when they ranked highly in that stream. Tracing back through the code revealed the overwrite bug.

After fixing it to properly accumulate across all 6 independent streams, the pool quality improved and candidates with strong technical keyword matches started appearing alongside the semantic similarity matches.

---

## The Honeypot Gate: More Work Than Expected

My first F8 implementation used three signals:
- Expert skill claimed with zero months of usage duration
- Multiple expert skills with zero duration
- Timeline inconsistency between claimed YoE and sum of role durations

The logic was sound but the thresholds were wrong. The initial timeline check flagged any candidate where the total role history duration did not closely match the claimed years of experience. This sounds reasonable. In practice it was catching legitimate senior professionals who simply did not list their full career history.

I ran a diagnostic that printed all eliminated AI/ML titled candidates with their data. Eight false positives showed up. Investigating each one:

- CAND_0013536 (Applied ML Engineer, 14.1yr claimed, 56mo of roles): Valid. Senior professionals commonly do not list roles older than 5-7 years on their profile.
- CAND_0019480 (NLP Engineer, 2.8yr claimed, 87mo of roles): Invalid. Cannot have 87 months of role history while claiming only 2.8 years of experience. This is a genuine honeypot.
- CAND_0095619 (NLP Engineer, 15.6yr claimed, 50mo in 1 role at Nykaa): Valid. One role listed does not mean only one job worked. Expert skills show legitimate durations (41-82 months each).

The fix was to redesign H3 completely. The new logic only flags direction A: candidates where total role months exceed claimed YoE by more than 36 months (worked more time than claimed, genuinely impossible). Direction B (claimed YoE exceeds listed role months) was removed entirely. Legitimate candidates stop listing old jobs. Honeypots cannot retroactively add more role history than their claimed experience.

After the redesign: 43 eliminations, 0 false positives. Verified manually by checking each eliminated candidate's data.

The two confirmed honeypot patterns in the real dataset:
- 21 candidates with expert skills claimed at zero months usage (H1/H2 signals)
- 22 candidates with role history months exceeding claimed YoE by more than 36 months (H3 signal)

---

## The MinMax Normalisation Bug

This was the most impactful bug in the entire system. It would have severely hurt ranking quality if I had not caught it.

The composite score formula was:
```
score = 0.25 * ce + 0.20 * rrf + 0.15 * f2 + 0.18 * f4 + ...
```

The problem: cross-encoder (ms-marco-MiniLM-L-6-v2) outputs raw logits, not probabilities. On this dataset they range from approximately -9.26 to -1.89. Every other component was already in [0, 1]. Without normalisation, the CE term was contributing on a completely different scale. A difference of 7 units in CE (best vs worst) dwarfed any difference in other components where the maximum difference was 1 unit.

I caught this by printing score distributions after Stage 4 completed. The mean composite score was tracking the cross-encoder score almost perfectly. The other six features were making negligible contributions to the final ranking regardless of their assigned weights.

The fix was MinMax normalisation of all additive components over the reranking batch before the weighted sum:

```python
ce_norm = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min() + 1e-8)
```

Applied to: rrf, ce, f2, f3, f4, f5, f6. Then the weighted sum. Then multiply by f7, f9, recency_multiplier.

After the fix: min=0.034, max=0.710, mean=0.272. A real distribution with genuine spread. The top candidate (CAND_0002025, Senior AI Engineer) scoring 0.71 with the bottom candidates around 0.03 confirmed the system was now actually differentiating.

---

## Stage 1 JD Parsing: The Qwen 0.5B Problem

The original plan was to use Qwen for structured JSON extraction of all JD fields. I wrote a chain-of-thought extraction prompt that walked the model through each field step by step, then requested JSON output.

Qwen returned 2 hard requirements instead of 14. It was filling the JSON template with generic ML terms from its training data instead of extracting from the actual JD text.

I tried multiple approaches in sequence:

1. **Chain-of-thought prompt with step decomposition:** Still produced 2 requirements. The model under-extracted.
2. **JSON prefix completion (feed `{"reasoning": "` and let model complete):** Model ignored the prefix structure and generated a new JSON object from scratch, echoing placeholder text verbatim ("one sentence here").
3. **Grounded extraction with JD text inline and explicit instruction:** Produced the same 2 requirements. Model was drawing from training data rather than the provided text.
4. **temperature=0.0:** Made the output deterministic but did not fix the under-extraction.

After these attempts, the decision was to split the extraction problem into what Qwen can do and what deterministic code can do better:

- **Qwen:** Generates the role_intent summary (one natural language sentence, unstructured). This is within a 0.5B model's reliable capability range.
- **Keyword scan:** Deterministic scan of 16 technical requirement patterns against the actual JD text. Produces 14 verified requirements every time. Verifiably accurate.
- **Defaults:** Seven fields populated from `_minimal_jd_fallback()` which was carefully hand-tuned by reading the JD closely. These defaults are accurate for this specific JD.

This is documented honestly in the code and README. For a production system serving multiple different JDs, upgrading to Phi-3-mini-4k-instruct (3.8B) would solve this reliably.

---

## The Windows Environment Challenges

Development started on Windows with Anaconda. This produced a series of environment conflicts that required individual investigation and resolution:

**numpy binary incompatibility:** The venv had numpy 2.2.6 but scipy in the Anaconda base environment was compiled against numpy 1.x. Importing sentence-transformers triggered scipy through Anaconda's site-packages rather than the venv. Fixed by reinstalling torch and sentence-transformers with compatible versions.

**ProcessPoolExecutor MemoryError:** BM25 tokenisation used multiprocessing to parallelize across 100K candidates. On Windows, multiprocessing uses spawn (not fork), which means each worker process re-loaded the entire Python environment plus serialized and sent the full candidate chunk across process boundaries. With 100K candidate dicts, each chunk was large enough to cause MemoryError even with 2 workers. Fixed by removing multiprocessing entirely and using sequential tokenisation (~60-90s, reliable on all platforms).

**run.py using system Python:** The script called `subprocess.run(["python", ...])` which picked up the Anaconda system Python instead of the venv Python. Fixed by replacing "python" with `sys.executable` which always refers to the currently running interpreter.

**faiss-cpu binary incompatibility:** faiss-cpu 1.8.0 was compiled against numpy 1.x and failed with numpy 2.x. Fixed by upgrading faiss-cpu.

**Disk space:** By this point the C: drive had approximately 133MB free after multiple rounds of pip installs, uninstalls, and Anaconda base packages. Development moved to Google Colab at this point.

---

## Moving to Google Colab

Colab provided a clean Linux environment without Anaconda conflicts, adequate disk space, and a closer approximation to the hackathon's Linux evaluation sandbox. The tradeoff was Colab free tier's 2 CPU cores and a single PyTorch thread.

The two-track configuration (`REDROB_ENV`) was built specifically to handle this transition. Dev profile allowed fast local testing on small candidate samples. Submission profile ran full parameters. This meant iterative development could continue on constrained hardware without the full 33-minute pipeline run each time.

Moving to Colab also made git operations cleaner. Rather than fighting Windows path issues and venv activation, the Colab workflow was:
1. Mount Drive
2. `git clone` or `git pull`
3. Copy candidates.jsonl from Drive
4. Install dependencies
5. Run and test
6. `git push` with token authentication

---

## Stage 5 Reasoning: The Final Struggle

The original plan for Stage 5 was Qwen generating JSON-formatted reasoning strings for the top 40 candidates.

Qwen produced output like:
```
{"reasoning": "One sentence max 20 words": {"candidate": "Candidate: Backend Engineer, 6.9y..."}}
```

It was treating the instruction text "One sentence max 20 words" as the reasoning content and echoing the candidate data back as a nested object.

Approaches tried:

1. **JSON prefix completion:** Feed `{"reasoning": "` and let Qwen complete. Qwen ignored the prefix and generated a new structure. Produced `{"reasoning": "Here is a one-sentence recruiter summary for the given JSON:"}`.

2. **Plain text extraction with preamble cleaning:** Ask for plain text, clean known preamble patterns. Produced sentences truncated mid-word: "The candidate's strong experience in NLP and their 12."

3. **JSON-prefix completion with brace closing:** Attempt to close truncated JSON by counting braces. Sometimes worked, often produced partial sentences.

4. **Partial reasoning recovery via regex:** Extract whatever text appeared after `"reasoning":` in the output. Produced grounded partial sentences but cut off mid-word.

After this iteration, the decision was: Qwen with plain text output and preamble cleaning for ranks 1-40, with per-candidate fallback to `build_structured_reasoning()` when Qwen output is too short or malformed. The structured assembly produces specific, grounded, recruiter-readable strings from the actual score breakdown data.

The structured assembly output for comparison:
```
Senior AI Engineer at PhonePe with 5.9y experience; top skills: FAISS, TensorFlow, scikit-learn; available immediately; actively looking.
```

This is specific, grounded, and correct. It is honestly better output than what Qwen 0.5B consistently produces for this task. This is documented as a known limitation with a clear upgrade path (Phi-3-mini-4k-instruct at 3.8B).

---

## The Full Pipeline Run

The end-to-end run on 100K candidates completed correctly on Colab free tier. Key results:

- Stage 1: 14 hard requirements, 4 query vectors, HyDE profile generated
- Stage 2: 43 honeypots eliminated, 7,683 BM25 pool
- Stage 3: 6-stream RRF fusion, 2,000 retrieval pool. Top FAISS scores: Q1=0.850, Q2=0.787, Q3=0.857, Q4=0.680
- Stage 4: 90 consulting-only eliminated, 1,910 cross-encoder candidates. Score distribution: min=0.034, max=0.710, mean=0.272
- Stage 5: Reasoning generated for all 100 candidates

Top ranked candidate: CAND_0002025, Senior AI Engineer, score=0.710.

Total wall-clock: approximately 33 minutes on 2-core Colab free tier.

---

## What Works Well

**Honeypot detection:** 43 genuine eliminations, 0 false positives after careful iteration on the threshold logic. The H1/H2 signals (expert skills with zero usage duration) are clean and definitively catch fabricated profiles. The H3 signal (worked more time than claimed) catches timeline manipulation.

**Consulting gate:** F1 correctly eliminated 90 consulting-only profiles from the 2,000 retrieval pool. This is one of the JD's most explicit signals and the system handles it precisely.

**Score distributions:** The composite scoring after MinMax normalisation produces a real, informative distribution. The spread from 0.034 to 0.710 means the system is genuinely differentiating candidates, not producing near-identical scores.

**6-stream RRF:** The hybrid fusion correctly surfaces candidates who would be missed by either BM25 or dense retrieval alone. Technical keyword specialists appear through the BM25 streams. Semantically strong but keyword-sparse candidates appear through the dense streams.

**Memory management:** The pipeline runs well within 16GB at peak approximately 1.8GB RSS. Explicit GC gates between every stage, BM25 corpus freed before embedding starts, models released after each use.

**Per-candidate reasoning fallback:** One Qwen failure affects exactly one candidate and no others. The pipeline never crashes on reasoning generation.

---

## What is Weak

**JD parsing completeness:** Seven of nine JD fields come from hand-tuned defaults. These are accurate for this specific JD. They are not generalisable. Any change to the JD (different role, different company, different location preferences) would require manually updating the defaults.

**Qwen reasoning reliability:** Qwen 0.5B cannot consistently follow structured output instructions. The per-candidate fallback works, but it means the system is doing less LLM-generated reasoning than intended. The reasoning quality from structured assembly is acceptable but it is not the same as genuinely LLM-generated candidate-specific text.

**Timing validation gap:** The full pipeline was not validated within the 5-minute budget. The Colab free tier machine is significantly slower than what the hackathon's 5-minute budget implies. The architecture is designed for that budget based on theoretical throughput estimates, but direct timing validation on appropriate hardware did not happen before submission.

**F4 vibe scoring approximation:** The vibe score uses cosine similarity between candidate embeddings and pre-computed culture signal phrase clusters. This is a reasonable approximation but not as precise as, for example, a classifier trained to detect "ships over researches" versus "researches over ships" from career description text.

---

## Technical Decisions Summary

| Decision | Alternatives Considered | Reason Chosen |
|----------|------------------------|---------------|
| Hybrid BM25 + dense retrieval | BM25 only, dense only, learned fusion | No training labels; RRF robust without tuning |
| 6-stream RRF | 2-stream (BM25 + dense), single query | Different retrieval dimensions benefit from independent streams |
| bge-small-en-v1.5 | bge-large, e5-small, all-MiniLM | Speed vs quality tradeoff; 384d, CPU-feasible, strong performance |
| FAISS IndexFlatIP | HNSW, IVF | At 8-15K vectors, exact search is fast enough; no index-build overhead |
| Sequential BM25 tokenisation | ProcessPoolExecutor | Windows spawn MemoryError; sequential is reliable cross-platform |
| MinMax normalisation per batch | Global normalisation, sigmoid | Per-batch normalisation matches what we can compute at scoring time |
| Consulting hard gate (F1) | Soft penalty | JD explicitly disqualifies; zero ambiguity |
| Qwen 0.5B for LLM reasoning | Phi-3-mini, Ollama, API calls | Local only (no network during ranking); Phi-3-mini too large for test environment |
| Structured assembly fallback | Template strings, fixed format | Signal-driven, candidate-specific, no two identical strings |
| Two-track REDROB_ENV config | Single config, separate repos | Separates development constraints from submission parameters cleanly |
