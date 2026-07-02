# Redrob AI Ranking System

This is my submission for the India Runs Hackathon by Redrob AI and Hack2Skill, Track 1: Intelligent Candidate Discovery.

The goal was simple to understand but genuinely hard to solve well: given 100,000 candidate profiles and one job description for a Senior AI Engineer role, find the best 100 people. Not just keyword match them. Actually rank them in a way that a good recruiter would agree with.

I spent a lot of time thinking about what "best" actually means here before writing any code. The JD has signals that most systems would miss — it explicitly says it does not want consulting-only careers, it wants people who shipped things to real users, it cares about culture fit as much as technical skills. So I built a system that tries to capture all of that, not just check for Python and FAISS in the profile.

## What I Built

The system is a five-stage ranking pipeline. Each stage does one job and hands off to the next:

**Stage 1** reads the job description and turns it into something the retrieval system can work with. It uses Qwen2.5 (a small local LLM) to generate a "hypothetical ideal candidate" profile that gets blended with the JD text to create better search queries. It also generates four different query vectors targeting different aspects of the role: the primary match, technical requirements, experience signals, and culture/vibe signals.

**Stage 2** runs the full 100K candidate pool through a honeypot detection gate, then does BM25 keyword retrieval using two separate query chambers (one for raw JD terms, one for an expanded technical taxonomy). The honeypot gate alone caught 43 logically impossible profiles with zero false positives, which I verified manually.

**Stage 3** takes the BM25 pool and runs semantic embedding using bge-small-en-v1.5, then does a six-stream Reciprocal Rank Fusion combining all four dense retrieval streams with the two BM25 streams. This is where hybrid retrieval actually happens.

**Stage 4** is where most of the ranking intelligence lives. It runs nine feature extractors across the 2,000 retrieved candidates, eliminates consulting-only profiles via a hard gate, runs the ms-marco cross-encoder for pairwise relevance scoring, then combines everything into a composite score with MinMax normalisation to prevent any single signal from dominating.

**Stage 5** generates recruiter-facing reasoning for each of the top 100 candidates. The top 40 go through Qwen for LLM-generated reasoning, the rest get dynamic structured assembly that reads from the actual score breakdown.

I want to be upfront: Qwen 0.5B is genuinely too small for reliable structured JSON output. It generates good reasoning text but struggles to wrap it in JSON consistently. The system handles this gracefully per candidate, falling back to structured assembly when needed. If I were building this for production I would swap Qwen for Phi-3-mini (3.8B) which handles this much better. I documented this decision in the code and architecture notes.

## The Part That Took the Most Work

Honestly, getting the honeypot gate right took more iteration than I expected. My first version was flagging legitimate senior candidates as honeypots because the timeline check was too aggressive. I had to diagnose which candidates were false positives, understand exactly why they were being flagged, and redesign the signals. The final version uses two hard signals (expert skills claimed with zero usage duration, and role history that exceeds claimed YoE by more than three years) and catches all the genuinely impossible profiles while leaving real candidates alone.

The other thing that took a lot of debugging was the composite scoring. My first implementation let the cross-encoder logits (which range from about -9 to +2) dominate the entire score because they were on a completely different scale than the 0-1 features. I caught this by looking at the score distributions and realising the mean score was basically tracking the CE score alone. The fix was MinMax normalisation across the batch for all additive components before the weighted sum.

## Environment Note

I built and tested this on Google Colab free tier (2 CPU cores, Intel Xeon 2.20GHz, 12GB RAM). On this machine, the full pipeline takes about 33 minutes. The hackathon sandbox has a 5-minute budget, which implies meaningfully more CPU cores. The system is designed to scale with available cores — the main bottleneck is sentence-transformers inference which is single-threaded by default on constrained environments.

To handle this, I built a two-track configuration system. Setting `REDROB_ENV=dev` switches to reduced parameters for local testing (500 candidates, smaller pools, faster run). The default `submission` profile runs the full pipeline with parameters tuned for the hackathon sandbox.

## Setup

```bash
git clone https://github.com/Aarushi200225/redrob-ranking.git
cd redrob-ranking
pip install -r requirements.txt
python scripts/download_models.py
```

The download script pulls three models (~700MB total). This requires network access and only needs to run once.

## Running the Pipeline

```bash
python run.py run
```

This produces `outputs/ranked_output.csv`. No network access required during the ranking step.

To validate the output format:

```bash
python validate_submission.py outputs/ranked_output.csv
```

For local testing on a small sample:

```bash
set REDROB_ENV=dev
python run.py run-sample
```

## Models Used

| Model | What it does | Size |
|---|---|---|
| Qwen2.5-0.5B-Instruct Q4_K_M (GGUF) | JD role intent generation, candidate reasoning | ~490MB |
| BAAI/bge-small-en-v1.5 | Candidate and query embedding | ~133MB |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | Pairwise JD-candidate relevance scoring | ~91MB |

## AI Tools

I used Claude for architecture discussions, code review, and debugging throughout the build. GitHub Copilot for autocomplete. No candidate data was sent to any external service. All ranking runs completely offline using local model weights.

## Repository Structure

```
redrob-ranking/
├── src/
│   ├── config.py              # Two-track environment configuration
│   ├── pipeline.py            # Five-stage orchestrator with GC memory gates
│   ├── stages/                # Stage 1-5 implementations
│   ├── features/              # F1-F9 feature extractors
│   ├── models/                # Qwen, bge-small, ms-marco wrappers
│   ├── retrieval/             # BM25, FAISS, RRF fusion
│   ├── reasoning/             # Dynamic structured assembly
│   └── utils/                 # Data loader, validator, scoring, logger
├── artifacts/
│   ├── skill_taxonomy.json    # Technical skill taxonomy (~200 entries)
│   └── vibe_keywords.json     # Culture signal phrase clusters
├── scripts/
│   ├── download_models.py     # Model weight downloader
│   └── sanity_check.py        # Pre-run validation checks
├── sandbox/
│   └── app.py                 # Streamlit demo for small sample ranking
├── outputs/
│   └── ranked_output.csv      # Submission output
├── run.py                     # Cross-platform CLI
├── requirements.txt
└── submission_metadata.yaml
```
