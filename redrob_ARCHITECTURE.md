# System Architecture

## What This System Is

This is a five-stage hybrid retrieval and reranking pipeline for candidate ranking. It is not a RAG system and not an agentic system. It follows the two-stage retrieval pattern used in production hiring systems: fast approximate retrieval to narrow a large pool, followed by expensive but precise reranking on the shortlisted candidates.

The pattern is identical to what LinkedIn Recruiter, Indeed, and Redrob itself use internally. The difference from most hackathon submissions is that this system does not stop at keyword matching or a single embedding similarity score. It models nine distinct candidate quality signals, fuses them with calibrated weights, and generates recruiter-readable reasoning grounded in actual candidate data.

---

## Pipeline Overview

```
candidates.jsonl [100K]          job_description.txt
        |                               |
        +---------------+---------------+
                        |
              Stage 1: JD Intelligence
              Qwen2.5-0.5B + bge-small-en-v1.5
              Output: JD object + 4 query vectors + vibe embeddings
                        |
              Stage 2: Honeypot Gate + Dual-Chamber BM25
              F8 vectorised gate (all 100K) + BM25 retrieval
              100K -> ~8K candidates
                        |
              Stage 3: Semantic Embedding + 6-Stream RRF
              bge-small batch encoding + FAISS + RRF fusion
              8K -> 2,000 candidates
                        |
              Stage 4: Multi-Signal Reranking
              F1-F9 features + cross-encoder + MinMax composite scoring
              2,000 -> top 100 candidates
                        |
              Stage 5: Reasoning + Output
              Qwen LLM (ranks 1-40) + structured assembly (41-100)
              Output: ranked_output.csv
```

---

## Stage 1: JD Intelligence

**Input:** job_description.txt (9,572 chars)
**Output:** Structured JD object with query vectors and vibe cluster embeddings
**Runtime:** ~25s on submission hardware | ~68s on Colab free tier

The JD is processed in three sequential passes.

### Pass 1: Qwen2.5-0.5B (n_ctx=2048, n_batch=512)

Qwen generates two outputs:

- **role_intent:** One sentence describing the ideal candidate in plain language. Example output: "The ideal candidate for the Senior AI Engineer position at Redrob AI is someone with a deep technical background in modern ML systems and a scrappy product-engineering attitude, ready to learn from real users and optimize for the underlying ML."

- **HyDE profile:** A hypothetical "ideal candidate" LinkedIn summary written in first-person candidate vocabulary. This bridges the vocabulary gap between JD language ("seeking candidates who have", "must demonstrate") and candidate profile language ("I built", "I shipped", "I led"). Without HyDE, semantic search on raw JD text produces lower recall for candidates who describe the same skills with different words.

### Pass 2: Deterministic Keyword Scan

Qwen 0.5B cannot reliably produce multi-field structured JSON extraction. After testing chain-of-thought prompting, JSON prefix completion, grammar-constrained generation, and grounded extraction with JD text inline, the model consistently under-extracts (2 requirements instead of 14) or echoes placeholder text verbatim.

Decision: deterministic keyword scan for hard requirements extraction. The scan checks 16 technical requirement patterns against the JD text and produces 14 verified hard requirements. These are then expanded via the skill taxonomy to approximately 102 related terms for BM25 Chamber B.

Seven of nine JD object fields use hand-tuned defaults from `_minimal_jd_fallback()`. These defaults were carefully derived from close reading of the actual JD text and are accurate for this specific role. They are not generalisable to other JDs without a model upgrade to Phi-3-mini-4k-instruct (3.8B), which handles multi-field structured extraction reliably.

**Fields from keyword scan:** hard_requirements (14 items)
**Fields from Qwen:** role_intent (1 sentence)
**Fields from defaults:** soft_penalties, soft_positives, experience_band, location_preferences, notice_preference, culture_signals, company_type

### Pass 3: bge-small-en-v1.5 Embedding

Four query vectors generated:

| Vector | Construction | Purpose |
|--------|-------------|---------|
| Q1 | 0.4 * JD_vec + 0.6 * HyDE_vec | Primary retrieval (HyDE-weighted) |
| Q2 | encode(technical requirements) | Technical skills retrieval |
| Q3 | encode(experience signals) | Experience quality retrieval |
| Q4 | encode(vibe phrase clusters) | Culture fit retrieval |

Four vibe cluster embeddings also generated (ships_over_researches, async_writer, startup_tolerance, responsive_communicator) for use in F4 behavioral scoring in Stage 4.

---

## Stage 2: Honeypot Gate + Dual-Chamber BM25

**Input:** 100,000 candidates
**Output:** ~8K candidate pool + top_a, top_b chamber results for RRF
**Runtime:** ~90s submission | ~102s Colab free tier

### F8 Honeypot Gate (vectorised numpy, all 100K)

Three detection signals, each contributing to a flag score:

| Signal | Condition | Weight |
|--------|-----------|--------|
| H1 | Any skill with proficiency="expert" AND duration_months=0 | +0.50 |
| H2 | Two or more expert+zero-duration skills | +0.30 additional |
| H3 | sum(role_months) > YoE*12 + 36 (worked more than claimed) | +0.20 to +0.50 (scaled) |

Score = 1.0 - flag_sum. Gate threshold: 0.55.
Candidates below threshold are hard-eliminated before any retrieval.

**Results on full 100K dataset:** 43 eliminated, 0 false positives.

**False positive analysis:** Initial H3 logic flagged 8 legitimate AI/ML candidates. Investigation showed these were senior professionals who simply did not list their full career history. H3 was redesigned to only flag direction A (role months exceed claimed YoE, which is genuinely impossible) and to never flag direction B (claimed YoE exceeds listed role months, which is completely normal). CAND_0095619 (NLP Engineer, 15.6yr claimed, 1 role listed at Nykaa, 4 correct expert skills) was the key case that drove this redesign.

### Dual-Chamber BM25

Two independent BM25 indices are run with different query vocabularies:

**Chamber A:** Raw JD terms tokenised and scored against all 99,957 clean candidates. Retrieves top 5,000. Captures exact technical keyword matches.

**Chamber B:** Expanded taxonomy terms (~102 terms derived from the 14 hard requirements) scored against all candidates. Retrieves top 5,000. Captures related skills and synonyms.

Union of both chambers, deduplicated by candidate_id, capped at 15,000. Result: ~8,000 unique candidates in practice (chambers overlap significantly for relevant profiles).

BM25 corpus (~500MB of tokenised strings) and index are explicitly deleted and garbage collected before Stage 3 begins. This is necessary to keep memory flat during embedding.

---

## Stage 3: Semantic Embedding + 6-Stream RRF

**Input:** ~8K BM25 pool + top_a, top_b, Q1-Q4 query vectors
**Output:** Top 2,000 candidates + rrf_score_map
**Runtime:** ~25s submission (estimated) | ~954s Colab free tier (2 cores, 1 thread)

### Text Blob Construction

Each candidate is represented as a single text blob for embedding:
- headline (full)
- summary (first 150 chars)
- top 2 career role descriptions (first 100 chars each)
- top 8 skills by proficiency + duration

Blobs are hard-capped at 400 chars. This was determined empirically: average blob length at 400 chars produces 9.5 texts/sec throughput on 2-core hardware vs 2.1 texts/sec for uncapped blobs (~1,582 chars average). The semantic signal for retrieval is concentrated in the first few sentences of each field.

### FAISS IndexFlatIP

Exact inner product search on L2-normalised vectors (equivalent to cosine similarity). At 8-15K vectors of 384 dimensions, exact search is preferable to approximate methods (HNSW, IVF). Approximate methods add index-building overhead for negligible speed benefit at this scale.

Four queries run independently:
- Q1: top 2,000 results
- Q2: top 1,000 results
- Q3: top 1,000 results
- Q4: top 1,000 results

### 6-Stream RRF Fusion (k=60)

```
RRF(candidate) = sum over all streams: 1 / (60 + rank_in_stream)
```

Streams: Q1, Q2, Q3, Q4, BM25_A, BM25_B

If a candidate is absent from a stream, that term contributes 0. This is mathematically correct: absence means no evidence, not negative evidence.

RRF with k=60 is empirically validated across IR benchmarks and requires no parameter tuning. Fusion result: top 2,000 candidates by RRF score from a pool of 7,683 unique candidates across all streams.

---

## Stage 4: Multi-Signal Reranking

**Input:** 2,000 candidates + rrf_score_map
**Output:** Top 100 candidates with _score and _score_breakdown
**Runtime:** ~55s submission (estimated) | ~671s Colab free tier

### Feature Extractors F1-F9

| Feature | Signal | Type |
|---------|--------|------|
| F1 Title Fit | Consulting-only hard gate; product company ratio; career progression | Hard gate (-1.0) or [0-1] |
| F2 Experience Quality | YoE band (5-9yr = 1.0), AI/ML years ratio, production deployment signal, pre-2022 ML work bonus, education tier | [0-1] additive |
| F3 Skills Match | Expanded requirement coverage, proficiency depth scoring, skill assessment scores, anti-stuffer penalty | [0-1] additive |
| F4 Vibe Score | Semantic cosine similarity to vibe cluster embeddings, writing quality proxy (lexical diversity), startup DNA ratio, GitHub activity score | [0-1] additive |
| F5 Availability | Recency exponential decay (half-life 60 days), open_to_work flag, recruiter response composite, notice period score | [0-1] additive + recency multiplier |
| F6 Market Validation | Log-normalised recruiter saves (30d), search appearances (30d), relevant skill endorsements only | [0-1] additive |
| F7 Location Fit | Tier-based multiplier: Pune/Noida=1.0, tier-2 cities=0.92, other India±relocate=0.82/0.65, outside India±relocate=0.55/0.30 | Multiplier |
| F8 Honeypot Gate | Applied in Stage 2, not repeated in Stage 4 | Already applied |
| F9 Salary Fit | Salary midpoint vs target range (15-80 LPA): in-range=1.0, edge=0.85, outside=0.65, missing=1.0 neutral | Multiplier |

### Structural Pre-filter

Candidates with F2 < 0.20 AND F3 < 0.20 skip the cross-encoder. Their CE score is pre-assigned 0.0. They remain in the pool so other signals (RRF, F4 vibe) can still rescue genuinely strong candidates with low keyword coverage. This reduced the cross-encoder candidate count from 2,000 to approximately 1,910 in practice.

### F1 Consulting-Only Hard Gate

Candidates with zero product company roles in their career history are hard-eliminated (F1 returns -1.0). This is an explicit JD disqualifier. In the full 100K run, 90 candidates were eliminated from the 2,000 retrieval pool by this gate.

Consulting-only detection uses company name matching against a list of 16 major IT services firms plus industry-string matching ("it services", "consulting", "outsourcing", "staffing").

### Cross-Encoder Reranking

ms-marco-MiniLM-L-6-v2 scores each candidate-JD pair. Input text is truncated to 600 chars combining: headline, summary (150 chars), top 3 career roles (80 chars each), top 6 skills with proficiency labels.

Cross-encoder logit range observed: -9.26 to -1.89. These are raw logits, not probabilities. Without normalisation, this range dominates the composite score regardless of assigned weight.

### MinMax Normalisation (Critical Fix)

All additive components are MinMax normalised over the 500-candidate reranking pool before the weighted sum:

```python
norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
```

Applied to: rrf_scores, ce_scores, f2_experience, f3_skills, f4_vibe, f5_availability, f6_market.

Without this normalisation, CE logits (range ~7 units) completely dominate components that are already in [0,1] (range 1 unit), regardless of the 0.25 assigned weight. Score distributions before fix showed mean tracking CE score almost exactly. After fix: min=0.034, max=0.710, mean=0.272.

### Composite Score Formula

```
final_score = (
    0.20 * rrf_norm
  + 0.25 * ce_norm
  + 0.15 * f2_norm
  + 0.12 * f3_norm
  + 0.18 * f4_norm
  + 0.07 * f5_norm
  + 0.03 * f6_norm
) * f7_location * f9_salary * recency_multiplier
```

**Weight rationale:**
- ce_norm 0.25: Strongest pairwise relevance signal, direct JD-candidate comparison
- f4_vibe 0.18: JD explicitly prioritises culture fit and "scrappy product-engineering attitude" over pure technical skills
- rrf_norm 0.20: Combined retrieval confidence from 6 independent streams
- f2_experience 0.15: Production deployment history, pre-2022 ML work, YoE band fit
- f3_skills 0.12: Deliberately below vibe weight, skills are teachable
- f5_availability 0.07: Recruiter actionability signal
- f6_market 0.03: Weak, noisy platform signal

**Tie-breaking:** score DESC, recruiter_response_rate DESC, candidate_id ASC (deterministic, matches validate_submission.py requirement).

---

## Stage 5: Reasoning + Output

**Input:** Top 100 candidates with score breakdowns
**Output:** ranked_output.csv
**Runtime:** ~35s

### Ranks 1-40: Qwen LLM Reasoning

Qwen2.5-0.5B (n_ctx=512, n_batch=32) generates a one-sentence recruiter summary per candidate. The prompt provides candidate title, YoE, top 2 skills, and strongest scoring signal.

Per-candidate try/except ensures one Qwen failure never affects other candidates. When Qwen output is too short, malformed, or contains preamble text, build_structured_reasoning() runs for that specific candidate.

**Known limitation:** Qwen 0.5B generates grounded, candidate-specific text but cannot follow strict JSON output formatting consistently at n_ctx=512. This is a fundamental model size limitation. Production upgrade path: Phi-3-mini-4k-instruct (3.8B) produces reliable structured output.

### Ranks 41-100: Dynamic Structured Assembly

build_structured_reasoning() reads _score_breakdown to identify the strongest signal per candidate, then leads with that signal:
- cross_encoder or rrf or experience signal: leads with title, company, YoE, top skills
- vibe signal: leads with culture-fit framing, startup roles, GitHub score
- other signals: leads with skills and experience summary

Appends availability snippet (notice period, open_to_work) if relevant. No two candidates produce identical strings unless their profiles are identical.

### Format Validation

Hard assertions matching validate_submission.py exactly:
- Exactly 100 rows
- Columns: candidate_id, rank, score, reasoning (exact order)
- Ranks 1-100, unique, no gaps
- Scores monotonically non-increasing
- No duplicate candidate_ids
- All IDs match CAND_[0-9]{7} pattern
- All IDs exist in source JSONL
- Scores not all identical
- Tie-breaking: equal scores -> candidate_id ascending

---

## Memory Architecture

Models are never loaded simultaneously. ModelContext manages lifecycle with explicit GC between each model:

```
Stage 1 Qwen     (~800MB) -> released -> GC gate
Stage 1 bge      (~133MB) -> released -> GC gate
Stage 2 BM25     (~500MB corpus) -> deleted before Stage 3 -> GC gate
Stage 3 bge      (~133MB) -> released -> GC gate
Stage 4 CrossEnc (~91MB)  -> released -> GC gate
Stage 5 Qwen     (~800MB, n_ctx=512) -> released
Peak observed: ~1.8GB RSS (well within 16GB budget)
```

`malloc_trim(0)` called on Linux between stages to return freed pages to the OS immediately.

---

## Two-Track Environment Configuration

```python
ENV = os.getenv("REDROB_ENV", "submission")
```

| Parameter | dev | submission |
|-----------|-----|-----------|
| DEV_CANDIDATE_LIMIT | 500 | None (all 100K) |
| BM25_UNION_CAP | 2,000 | 15,000 |
| EMBED_BATCH_SIZE | 32 | 512 |
| RETRIEVAL_FINAL_POOL | 200 | 2,000 |
| LLM_N_CTX | 512 | 2,048 |
| LLM_REASONING_TOP_N | 5 | 40 |

Dev profile enables fast local correctness testing without running the full 33-minute pipeline. All pipeline logic is identical between profiles.

---

## Technology Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| LLM inference | llama-cpp-python 0.3.30 | Qwen GGUF loading and inference |
| Embeddings | sentence-transformers 5.5.1 | bge-small and ms-marco |
| Vector search | faiss-cpu | IndexFlatIP exact cosine search |
| Sparse retrieval | rank-bm25 | BM25Okapi dual-chamber indexing |
| Data parsing | orjson | Fast JSONL loading |
| Numerics | numpy 1.26.4 | Vectorised feature extraction |
| Output | pandas | CSV generation |
| Validation | pydantic v2 | Output schema guardrails |
| Memory monitoring | psutil | RSS tracking between stages |

---

## Known Limitations and Upgrade Paths

| Limitation | Current State | Production Fix |
|-----------|---------------|---------------|
| JD parsing (7/9 fields) | Hand-tuned defaults, accurate for this JD | Phi-3-mini-4k-instruct (3.8B) |
| Qwen reasoning reliability | Grounded text but inconsistent JSON formatting | Phi-3-mini-4k-instruct (3.8B) |
| BM25 tokenisation speed | Sequential (ProcessPoolExecutor removed for Windows compat) | Linux fork-based parallelism |
| Timing on 2-core hardware | ~33 minutes (Colab free tier) | ~4-5 min on 8+ core hardware |
| F4 vibe scoring | Cosine similarity approximation | Trained classifier on labeled examples |
