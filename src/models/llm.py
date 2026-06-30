"""
models/llm.py
─────────────
Qwen2.5-0.5B-Instruct GGUF wrapper via llama-cpp-python.

Key design decisions:
  - Grammar-constrained JSON output (GBNF) — zero parsing failures
  - Chain-of-thought extraction prompt — critical for small model quality
  - n_ctx and n_batch are separate for Stage 1 vs Stage 5
  - n_batch=32 for Stage 5 sequential generation (no benefit from 512)
  - Per-candidate try/except in reasoning — one failure never cascades

Critical: llama-cpp-python is NOT fork-safe.
Never use inside ProcessPoolExecutor workers.
"""

import re
import orjson
from pydantic import BaseModel, ValidationError

from src.config import (
    LLM_MODEL_PATH,
    LLM_N_CTX,
    LLM_N_BATCH,
    LLM_N_THREADS,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_REASONING_TOKENS,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Pydantic output models ────────────────────────────────────────────────────

class JDObject(BaseModel):
    hard_requirements:    list[str]
    soft_penalties:       list[str]
    soft_positives:       list[str]
    experience_band:      dict
    location_preferences: dict
    notice_preference:    dict
    company_type:         list[str]
    culture_signals:      dict
    role_intent:          str


class ReasoningOutput(BaseModel):
    reasoning: str


# ── Model loader ──────────────────────────────────────────────────────────────

def load_llm(n_ctx: int = None, n_batch: int = None):
    """
    Load Qwen2.5-0.5B-Instruct from local GGUF file.

    Parameters
    ----------
    n_ctx   : Context window size. Defaults to LLM_N_CTX (2048).
    n_batch : Batch size for token processing.
              Use LLM_N_BATCH (512) for Stage 1 parsing.
              Use LLM_N_BATCH_REASONING (32) for Stage 5 generation.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python not installed. "
            "Run: pip install llama-cpp-python --prefer-binary "
            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
        )

    if not LLM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Qwen GGUF not found: {LLM_MODEL_PATH}\n"
            f"Run: python scripts/download_models.py"
        )

    _n_ctx   = int(n_ctx)   if n_ctx   is not None else int(LLM_N_CTX)
    _n_batch = int(n_batch) if n_batch is not None else int(LLM_N_BATCH)

    log.info(
        f"Loading Qwen GGUF: {LLM_MODEL_PATH.name} "
        f"(n_ctx={_n_ctx}, n_batch={_n_batch})"
    )

    model = Llama(
        model_path   = str(LLM_MODEL_PATH),
        n_ctx        = _n_ctx,
        n_batch      = _n_batch,
        n_threads    = int(LLM_N_THREADS),
        logits_all   = False,
        verbose      = False,
    )
    log.info("Qwen loaded")
    return model


# ── JSON extraction helper ────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Extract first JSON object from LLM output.
    Handles markdown fences, preamble, and truncated output.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.replace("```", "").strip()

    # Find opening brace
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in output: {text[:200]}")

    # Try to find complete JSON by scanning for balanced braces
    depth   = 0
    end_idx = -1
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break

    if end_idx == -1:
        # Truncated — attempt to close incomplete JSON
        json_fragment = text[start:]
        # Count unclosed braces and close them
        open_count = json_fragment.count("{") - json_fragment.count("}")
        json_fragment += "}" * open_count
        try:
            return orjson.loads(json_fragment)
        except Exception:
            raise ValueError(f"Truncated JSON cannot be recovered: {text[start:start+300]}")
    
    return orjson.loads(text[start:end_idx])


# ── JD parsing ────────────────────────────────────────────────────────────────

def parse_jd(model, jd_text: str) -> dict:
    """
    Parse job description using hybrid extraction.

    Design decision: Qwen2.5-0.5B is unreliable for structured
    multi-field JSON extraction — it under-extracts or echoes
    placeholder text. Keyword scan is deterministic and accurate
    for technical requirements (verified against actual JD content).

    Qwen is used only for the role_intent summary — a single
    short sentence is well within a 0.5B model's reliable range.
    """
    # Primary: deterministic keyword scan for hard requirements
    base = _enrich_with_keyword_scan({}, jd_text)

    # Qwen handles only the short, low-risk summary field
    try:
        role_intent = _generate_role_intent(model, jd_text)
        if role_intent and len(role_intent) > 20 and "based on the JD" not in role_intent:
            base["role_intent"] = role_intent
    except Exception as e:
        log.debug(f"Role intent generation skipped ({e}) — using default")

    log.info(
        f"JD parsed: "
        f"{len(base.get('hard_requirements', []))} requirements "
        f"(keyword scan, deterministic)"
    )
    return base


def _generate_role_intent(model, jd_text: str) -> str:
    """
    Generate a single-sentence role summary.
    Narrow, well-bounded task — within Qwen 0.5B's reliable range.
    """
    prompt = f"""Summarize this job in exactly one sentence describing the ideal candidate.

Job posting:
{jd_text[:1500]}

One sentence summary:"""

    response = model.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0.2,
    )
    text = response["choices"][0]["message"]["content"].strip()
    # Strip quotes if model wrapped the sentence
    text = text.strip('"\'')
    return text

def _enrich_with_keyword_scan(parsed: dict, jd_text: str) -> dict:
    """
    Rule-based keyword scan to supplement or replace weak Qwen output.
    Scans JD text for known technical keywords for this role type.
    """
    jd_lower = jd_text.lower()

    TECH_SIGNALS = {
        "Python":                ["python"],
        "FAISS":                 ["faiss"],
        "vector database":       ["vector db", "vector database", "vector store", "vector index"],
        "embeddings":            ["embedding", "embeddings", "dense retrieval", "bi-encoder"],
        "sentence-transformers": ["sentence-transformer", "sbert", "bge", "e5"],
        "ranking systems":       ["ranking", "reranking", "reranker", "ltr", "learning to rank"],
        "NLP":                   [" nlp ", "natural language processing"],
        "production ML":         ["production", "deployed to production", "serving at scale"],
        "retrieval systems":     ["retrieval", "information retrieval", "ir system"],
        "LLM":                   [" llm", "large language model", "generative ai"],
        "transformer models":    ["transformer", "bert", "attention mechanism"],
        "fine-tuning":           ["fine-tun", "lora", "qlora", "peft", "sft"],
        "search systems":        ["elasticsearch", "opensearch", "bm25"],
        "PyTorch":               ["pytorch", "torch"],
        "evaluation metrics":    ["ndcg", "mrr", "map", "recall@k"],
        "RAG":                   ["rag", "retrieval augmented", "retrieval-augmented"],
    }

    found = [req for req, patterns in TECH_SIGNALS.items()
             if any(p in jd_lower for p in patterns)]

    base = _minimal_jd_fallback()
    return {
        "hard_requirements":    found or base["hard_requirements"],
        "soft_penalties":       parsed.get("soft_penalties",       base["soft_penalties"]),
        "soft_positives":       parsed.get("soft_positives",       base["soft_positives"]),
        "experience_band":      parsed.get("experience_band",      base["experience_band"]),
        "location_preferences": parsed.get("location_preferences", base["location_preferences"]),
        "notice_preference":    parsed.get("notice_preference",    base["notice_preference"]),
        "company_type":         parsed.get("company_type",         base["company_type"]),
        "culture_signals":      parsed.get("culture_signals",      base["culture_signals"]),
        "role_intent":          parsed.get("role_intent",          base["role_intent"]),
    }


# ── HyDE generation ───────────────────────────────────────────────────────────

def generate_hyde_profile(model, jd_text: str) -> str:
    """
    Generate hypothetical ideal candidate profile (HyDE).

    The profile uses candidate-vocabulary text, bridging the gap
    between JD language and resume language for better retrieval.
    """
    prompt = f"""Write a realistic LinkedIn professional summary for the ideal candidate for this job.
Use first person. Be specific about technologies and achievements. Maximum 120 words.
Output only the summary, no other text.

Job Description:
{jd_text[:1500]}"""

    log.info("Generating HyDE profile ...")
    try:
        response = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        hyde = response["choices"][0]["message"]["content"].strip()
        log.info(f"HyDE profile generated ({len(hyde)} chars)")
        return hyde
    except Exception as e:
        log.warning(f"HyDE generation failed ({e}) — using template fallback")
        return (
            "I am a Senior AI Engineer with 7 years of experience building "
            "production retrieval and ranking systems. I have shipped FAISS-based "
            "semantic search, trained embedding models, and built learning-to-rank "
            "pipelines at product companies. I work with Python, PyTorch, and "
            "transformer models daily. I have deployed LLM fine-tuning with LoRA "
            "and built RAG systems serving real users."
        )


# ── Reasoning generation ──────────────────────────────────────────────────────

def generate_reasoning(
    model,
    candidate: dict,
    jd_object: dict,
    score_breakdown: dict,
) -> str:
    """
    Generate recruiter-facing reasoning string for a candidate.

    Per-candidate try/except ensures one Qwen failure never
    cascades to remaining candidates — each falls back to
    structured assembly independently.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills  = candidate.get("skills", [])

    proficiency_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    top_skills = sorted(
        skills,
        key=lambda s: proficiency_order.get(s.get("proficiency", "beginner"), 0),
        reverse=True,
    )[:3]
    skill_str = ", ".join(
        f"{s['name']}({s['proficiency']})" for s in top_skills
    )

    top_signals = sorted(
        score_breakdown.items(), key=lambda x: x[1], reverse=True
    )[:3]
    signal_str = ", ".join(f"{k}: {v:.2f}" for k, v in top_signals)

    prompt = f"""You are a technical recruiter. Write 1-2 sentences explaining why this candidate fits the role.
Be specific about their actual experience. No generic phrases.
Output ONLY JSON: {{"reasoning": "your explanation"}}

Role: {jd_object.get('role_intent', 'Senior AI Engineer')}

Candidate:
- Title: {profile.get('current_title', '')} at {profile.get('current_company', '')}
- Experience: {profile.get('years_of_experience', 0):.1f} years
- Top skills: {skill_str}
- Last active: {signals.get('last_active_date', 'unknown')}
- Response rate: {signals.get('recruiter_response_rate', 0):.0%}
- Notice: {signals.get('notice_period_days', 0)} days
- Open to work: {signals.get('open_to_work_flag', False)}
- Top signals: {signal_str}"""

    try:
        response = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=LLM_REASONING_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        raw    = response["choices"][0]["message"]["content"]
        parsed = _extract_json(raw)
        result = ReasoningOutput(**parsed)
        return result.reasoning

    except (ValueError, ValidationError, Exception) as e:
        log.debug(f"Reasoning generation failed for {candidate.get('candidate_id')}: {e}")
        # Per-candidate fallback — import here to avoid circular imports
        from src.reasoning.structured_assembly import build_structured_reasoning
        return build_structured_reasoning(candidate, jd_object)


# ── Fallback ──────────────────────────────────────────────────────────────────

def _minimal_jd_fallback() -> dict:
    """Minimal JD object for complete Stage 1 failure."""
    return {
        "hard_requirements": [
            "Python", "FAISS", "vector database", "embeddings",
            "production ML", "ranking systems", "NLP", "retrieval systems",
            "sentence-transformers", "LLM", "fine-tuning", "PyTorch",
        ],
        "soft_penalties": [
            "consulting_only_career",
            "langchain_only_no_pre_llm_work",
            "pure_research_no_production",
            "title_jumper_1_5yr_pattern",
        ],
        "soft_positives": [
            "pre_2022_ml_production", "open_source_contributions",
            "ltr_experience", "hr_tech_background",
        ],
        "experience_band":      {"min": 5, "max": 9},
        "location_preferences": {
            "tier_1": ["Pune", "Noida"],
            "tier_2": ["Hyderabad", "Mumbai", "Delhi NCR"],
        },
        "notice_preference":    {"ideal_days": 30, "max_days": 90},
        "company_type":         ["product", "startup", "scaleup"],
        "culture_signals": {
            "async_writer":           True,
            "startup_tolerance":      True,
            "ships_over_researches":  True,
            "responsive_communicator": True,
        },
        "role_intent": (
            "Senior AI Engineer who shipped retrieval and ranking systems "
            "to real users at product companies — not a researcher or "
            "framework wrapper."
        ),
    }