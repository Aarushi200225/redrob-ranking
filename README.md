# Redrob Intelligent Candidate Ranking System

---

This repository is my submission for Track 1 of the India Runs Hackathon. The task was to rank 100 candidates from a pool of 100,000 for a Senior AI Engineer role at Redrob AI.

What This System Is (and Is Not):
This system moves beyond simplistic keyword matching or isolated embedding similarity scores. Instead, it is built as a five-stage hybrid retrieval and reranking pipeline engineered to replicate how an experienced technical recruiter evaluates human talent.

By analyzing profiles holistically, the system dynamically checks for behavioral red flags and simultaneously weighs multiple dimensions of role alignment. The architecture adapts the proven, industry-standard two-stage retrieval pattern utilized by enterprise platforms like LinkedIn Recruiter, fully optimized from the ground up to capture the precise engineering and cultural signals demanded by this job description.

---

## Setup Instructions

### Requirements:
- Python 3.10 or higher
- 16 GB RAM
- CPU only (no GPU needed)
- Network access for the one-time model download step only

### Step 1: Clone the repository

```bash
git clone https://github.com/Aarushi200225/redrob-ranking.git
cd redrob-ranking
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download model weights (one-time only, requires network)

```bash
python scripts/download_models.py
```

This pulls three models totalling roughly 700 MB:

| Model | Size | Purpose |
|-------|------|---------|
| Qwen2.5-0.5B-Instruct Q4_K_M (GGUF) | ~490 MB | JD role intent, HyDE generation, candidate reasoning |
| BAAI/bge-small-en-v1.5 | ~133 MB | Dense embedding for candidates and queries |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | ~91 MB | Pairwise JD-to-candidate relevance scoring |

Once downloaded, the ranking step runs completely offline. No network calls happen during the actual pipeline execution.

### Step 4: Place your input files

Place your candidates file at `data/candidates.jsonl` and the job description at `data/job_description.txt`. The repository already includes the job description used for this submission.

---

## Reproducing the Submission CSV

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./outputs/ranked_output.csv
```

This is the single command that produces the ranked output from scratch. It runs all five pipeline stages sequentially and writes a validated CSV to the output path you specify.

### Validating the output

```bash
python validate_submission.py outputs/ranked_output.csv
```

The validator checks all format requirements: exactly 100 rows, ranks 1 to 100 with no gaps, monotonically non-increasing scores, no duplicate candidate IDs, valid ID format, non-empty reasoning strings, and correct tie-breaking order.

---

## How the Pipeline Works

### Pre-computed artifacts (committed to repo)

The system depends on two JSON files in the `artifacts/` directory that are committed to this repo:

- `artifacts/skill_taxonomy.json`: a technical skill taxonomy with 12 categories and around 200 entries. Used to expand the 14 hard requirements extracted from the JD into approximately 102 related terms for BM25 Chamber B.
- `artifacts/vibe_keywords.json`: four culture signal phrase clusters (ships_over_researches, async_writer, startup_tolerance, responsive_communicator). Used in Stage 1 to generate vibe cluster embeddings and in Stage 4 for behavioral scoring.

No embeddings or FAISS indexes are pre-computed. Everything is generated at runtime from the candidate file.

### Stage 1: JD Intelligence (approx. 25 seconds on submission hardware)

The job description goes through three processing passes. First, Qwen2.5-0.5B reads the JD and generates a role intent summary, which is one sentence describing the ideal candidate in plain language. Qwen also generates a HyDE profile, which stands for Hypothetical Document Embedding. The idea behind HyDE is that JD language and candidate profile language are quite different from each other. A JD says things like "seeking candidates who have shipped retrieval systems to production users." A strong candidate's profile says "I led the development of a FAISS-based semantic search system serving 2 million daily active users." These describe the same person but share almost no vocabulary. The HyDE profile is a synthetic first-person candidate summary written the way a real candidate would write it, which produces a query vector that retrieves better matches than the raw JD text alone.

Second, a deterministic keyword scan extracts 14 hard technical requirements from the JD text. I tried using Qwen for structured JSON extraction of all JD fields, but Qwen 0.5B consistently under-extracted, returning 2 requirements instead of 14, or echoed placeholder text back verbatim regardless of the prompt strategy used. The keyword scan is more reliable for this specific task and produces verified, accurate output every run.

Third, bge-small-en-v1.5 generates four query vectors: Q1 is a weighted blend of the JD text and HyDE profile (40% JD, 60% HyDE), Q2 targets technical requirements, Q3 targets experience signals, and Q4 targets culture and vibe phrases. Four vibe cluster embeddings are also generated here for use in Stage 4.

### Stage 2: Honeypot Gate and Dual-Chamber BM25 (approx. 90 seconds)

Before any retrieval happens, all 100,000 candidates pass through a vectorised honeypot detection gate. The gate runs entirely in numpy without any Python-level loops over candidates. It detects two types of impossible profiles:

Signal H1 catches candidates with any skill listed as expert proficiency with zero months of usage duration. This is logically impossible. You cannot claim expert-level proficiency in a skill you have never used. On the full 100K dataset this eliminated 21 candidates.

Signal H3 catches candidates whose total role history duration in months exceeds their claimed years of experience multiplied by 12, by more than 36 months. In other words, they claim to have worked more time than they claim to have been in the industry. Also impossible. This eliminated 22 candidates.

Total: 43 eliminations, 0 false positives. I verified this manually by investigating every AI/ML titled candidate that the gate flagged. The initial version of the gate had 8 false positives because the timeline check was too aggressive and was catching senior professionals who simply had not listed their full career history. The redesign specifically addresses this: we only flag candidates where role months exceed claimed YoE, never the reverse.

After the gate, dual-chamber BM25 retrieval runs on the clean pool. Chamber A uses raw JD terms and retrieves the top 5,000 candidates. Chamber B uses the expanded taxonomy terms (~102 terms) and retrieves a separate top 5,000. The union of both chambers, deduplicated by candidate ID and capped at 15,000, forms the retrieval pool going into Stage 3. In practice the chambers overlap significantly for genuinely relevant profiles, so the union pool is around 7,600 to 8,500 candidates.

### Stage 3: Semantic Embedding and 6-Stream RRF (approx. 25 seconds on submission hardware)

Each candidate in the BM25 pool is encoded using bge-small-en-v1.5. Candidate text blobs are capped at 400 characters and include the headline, a truncated summary, the top two career role descriptions, and the top eight skills sorted by proficiency and duration. The cap at 400 characters was decided empirically after finding that uncapped blobs of ~1,500 characters each caused a severe throughput drop on constrained hardware.

FAISS IndexFlatIP runs four independent dense retrieval searches using Q1 through Q4. Then six-stream Reciprocal Rank Fusion combines the four dense streams with the two BM25 chamber streams. RRF with k=60 is a well-validated fusion formula that requires no parameter tuning. A candidate absent from a stream gets a contribution of zero from that stream, which is mathematically correct rather than treating absence as a negative signal.

The result is the top 2,000 candidates by combined RRF score, along with a score map used in Stage 4 composite scoring.

### Stage 4: Multi-Signal Reranking (approx. 55 seconds on submission hardware)

Nine feature extractors run sequentially on the 2,000 candidate pool:

F1 is a consulting-only hard gate. Candidates with no product company roles in their career history are eliminated entirely. The JD is explicit: it does not want IT services or consulting profiles. On this run, 90 candidates were eliminated from the 2,000 pool by this gate.

F2 scores experience quality based on years of experience band fit (5 to 9 years is the sweet spot per the JD), the ratio of AI/ML years to total career, production deployment signals in career descriptions, a bonus for pre-2022 ML work indicating genuine depth before the LangChain era, and an education tier component.

F3 scores skills match using the expanded requirement taxonomy, proficiency depth weighting, any available platform assessment scores, and an anti-stuffer penalty for candidates with many skills listed at very low average usage duration.

F4 scores behavioral and culture fit using cosine similarity between candidate embeddings and the pre-computed vibe cluster vectors, a writing quality proxy based on lexical diversity in career descriptions, startup company ratio across the career history, and GitHub activity score.

F5 scores availability using an exponential recency decay (half-life 60 days), the open-to-work flag, recruiter response rate and response time composite, and notice period fit.

F6 is a weak market validation signal using log-normalised recruiter saves and search appearances from the last 30 days, and relevant skill endorsements only.

F7 is a location fit multiplier. Pune and Noida score 1.0 as tier-1 cities. Hyderabad, Mumbai, and Delhi NCR score 0.92. Other India locations with relocation willingness score 0.82, and without 0.65. Outside India drops further.

F9 is a salary fit multiplier based on the expected salary midpoint versus the target range of 15 to 80 LPA for this role at a Series A company.

After feature extraction, candidates where both F2 and F3 are below 0.20 skip the cross-encoder and receive a CE score of zero. They stay in the pool because other signals like RRF and F4 vibe may rescue a genuinely strong candidate with low keyword coverage. This reduced the cross-encoder workload from 2,000 to approximately 1,910 candidates in practice.

The ms-marco-MiniLM-L-6-v2 cross-encoder scores the remaining candidate-JD pairs and takes them down to the top 500. Then composite scoring runs.

A critical implementation detail here: cross-encoder outputs raw logits, not probabilities. On this dataset the logit range was -9.26 to -1.89. All the other features are already in the range of 0 to 1. Without normalisation, the cross-encoder contribution would dominate the composite score completely regardless of the assigned weight. The fix was MinMax normalisation of all additive components over the 500-candidate batch before the weighted sum. This was caught by examining score distributions and noticing the mean composite score tracked the CE score almost perfectly. After the fix, the distribution became min=0.034, max=0.710, mean=0.272.

The final composite formula is:

```
score = (
    0.20 * rrf_norm
  + 0.25 * ce_norm
  + 0.15 * f2_norm
  + 0.12 * f3_norm
  + 0.18 * f4_norm
  + 0.07 * f5_norm
  + 0.03 * f6_norm
) * f7_location * f9_salary * recency_multiplier
```

Vibe (F4) is weighted at 0.18, above skills (F3) at 0.12, because the JD signals culture fit as highly important. Tie-breaking goes by score descending, then recruiter response rate descending, then candidate ID ascending, which matches the official validator exactly.

### Stage 5: Reasoning and Output (approx. 35 seconds)

Ranks 1 through 40 get Qwen LLM reasoning. Ranks 41 through 100 get dynamic structured assembly, which reads the score breakdown for each candidate and leads with their actual strongest signal. The structured assembly was designed so that no two candidates produce identical strings unless their profiles are genuinely identical.

For Qwen reasoning, there is a per-candidate try/except so a single failure never affects other candidates. Qwen 0.5B is an honest limitation here. It generates grounded, candidate-specific text but struggles with strict output formatting at this context window size. The per-candidate fallback to structured assembly handles failures cleanly. For a production system, upgrading to Phi-3-mini-4k-instruct at 3.8B parameters would solve this reliably.

The output is validated against all submission rules before the CSV is written.

---

## A Note on Runtime

The full pipeline takes approximately 33 minutes on Google Colab free tier, which has 2 CPU cores and runs PyTorch with 1 thread. On 8 or more CPU cores with proper threading enabled, the estimated runtime is 4 to 5 minutes, which is within the 5-minute submission budget. The bottleneck is sentence-transformers inference in Stages 3 and 4, which scales linearly with available CPU threads.

To handle the development environment constraint, the system uses a two-track configuration via the `REDROB_ENV` environment variable:

```bash
# Development profile: 500 candidates, reduced pool sizes, fast iteration
set REDROB_ENV=dev
python run.py run-sample

# Submission profile (default): full 100K, all parameters at full scale
python rank.py --candidates ./data/candidates.jsonl --out ./outputs/ranked_output.csv
```

---

## Repository Structure

```
redrob-ranking/
├── src/
│   ├── config.py              # Two-track environment config, all constants
│   ├── pipeline.py            # Five-stage orchestrator with GC memory gates
│   ├── stages/                # Stage 1-5 implementations
│   ├── features/              # F1-F9 feature extractors
│   ├── models/                # Qwen, bge-small, ms-marco model wrappers
│   ├── retrieval/             # BM25, FAISS, RRF fusion modules
│   ├── reasoning/             # Dynamic structured assembly
│   └── utils/                 # Data loader, validator, scoring, logger
├── artifacts/
│   ├── skill_taxonomy.json    # Technical skill taxonomy (committed)
│   └── vibe_keywords.json     # Culture signal phrase clusters (committed)
├── scripts/
│   ├── download_models.py     # One-time model weight downloader
│   └── sanity_check.py        # Pre-run validation checks
├── sandbox/
│   └── app.py                 # Streamlit demo for small sample ranking
├── outputs/
│   └── ranked_output.csv      # Submission output
├── rank.py                    # Main entry point for reproduction
├── run.py                     # CLI with multiple commands
├── validate_submission.py     # Official hackathon format validator
├── requirements.txt           # All dependencies with versions
├── submission_metadata.yaml   # Portal metadata mirror
├── ARCHITECTURE.md            # Full system architecture documentation
└── BUILD_JOURNEY.md           # Build process and decision log
```

---

## AI Tools Declaration

Claude was used throughout development for architecture discussion & brainstorming, code review, debugging, and validating purposes. GitHub Copilot was used for autocomplete. No candidate data was sent to any external LLM or API during the ranking process. All ranking runs fully offline using local model weights.