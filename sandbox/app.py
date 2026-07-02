"""
sandbox/app.py
--------------
Streamlit demo for the Redrob Intelligent Candidate Ranking System.

Accepts up to 100 candidates via file upload or uses a pre-loaded
sample. Runs the full ranking pipeline and outputs a ranked CSV.

Designed for HuggingFace Spaces (CPU, free tier).
"""

import streamlit as st
import pandas as pd
import json
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="Redrob AI Candidate Ranker",
    page_icon="🎯",
    layout="wide"
)

st.title("Redrob Intelligent Candidate Ranking System")
st.caption("India Runs Hackathon — Track 1: Intelligent Candidate Discovery")

st.markdown("""
This demo ranks candidates for a Senior AI Engineer role using a five-stage
hybrid retrieval and reranking pipeline. Upload up to 100 candidates in JSONL
format to see the ranked output.
""")

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("System Info")
    st.markdown("""
    **Pipeline Stages**
    1. JD Intelligence (Qwen + bge-small)
    2. Honeypot Gate + BM25
    3. 6-Stream Hybrid RRF
    4. Multi-Signal Reranking
    5. Reasoning + Output

    **Models**
    - Qwen2.5-0.5B-Instruct GGUF
    - BAAI/bge-small-en-v1.5
    - ms-marco-MiniLM-L-6-v2

    **Constraint**
    CPU only, no network during ranking.
    Max 100 candidates in sandbox.
    """)

# ── Input section ─────────────────────────────────────────────────────────────
st.header("Input")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Candidates")
    uploaded = st.file_uploader(
        "Upload a JSONL file with up to 100 candidates",
        type=["jsonl", "json"],
        help="Each line should be a valid candidate JSON object"
    )

with col2:
    st.subheader("Or Use Sample")
    use_sample = st.button("Load Sample Candidates (10 profiles)")

# ── Load candidates ───────────────────────────────────────────────────────────
candidates_data = None

if uploaded is not None:
    try:
        content = uploaded.read().decode("utf-8").strip()
        if content.startswith("["):
            candidates_data = json.loads(content)[:100]
        else:
            sep = chr(10)
            parts = [p.strip() for p in content.split(sep) if p.strip()]
            candidates_data = [json.loads(p) for p in parts[:100]]
        st.success(f"Loaded {len(candidates_data)} candidates")
    except Exception as e:
        st.error(f"Failed to parse file: {e}")

elif use_sample:
    sample_path = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_candidates.json"
    if sample_path.exists():
        with open(sample_path) as f:
            candidates_data = json.load(f)[:10]
        st.success(f"Loaded {len(candidates_data)} sample candidates")
    else:
        st.warning("Sample file not found. Please upload a JSONL file.")

# ── Run pipeline ──────────────────────────────────────────────────────────────
if candidates_data is not None and st.button("Run Ranking Pipeline", type="primary"):

    with st.spinner("Running pipeline... this may take a few minutes on CPU."):
        try:
            import tempfile
            from src.pipeline import run_pipeline

            # Write candidates to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as tmp:
                for c in candidates_data:
                    tmp.write(json.dumps(c) + "\n")
                tmp_candidates = tmp.name

            # Output to temp file
            with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False
            ) as tmp_out:
                tmp_output = tmp_out.name

            jd_path = Path(__file__).parent.parent / "data" / "job_description.txt"

            os.environ["SANDBOX_MODE"] = "1"
            output_path = run_pipeline(
                jd_path=jd_path,
                candidates_path=Path(tmp_candidates),
                output_path=Path(tmp_output),
            )

            df = pd.read_csv(output_path)

            st.success("Pipeline complete!")

            # ── Results ───────────────────────────────────────────────────────
            st.header("Results")

            st.metric("Candidates Ranked", len(df))
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Top Score", f"{df['score'].max():.4f}")
            col_b.metric("Bottom Score", f"{df['score'].min():.4f}")
            col_c.metric("Mean Score", f"{df['score'].mean():.4f}")

            st.subheader("Top 10 Candidates")
            st.dataframe(
                df.head(10)[["rank", "candidate_id", "score", "reasoning"]],
                use_container_width=True
            )

            st.subheader("Full Ranking")
            st.dataframe(df, use_container_width=True)

            # Download button
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download ranked_output.csv",
                data=csv_bytes,
                file_name="ranked_output.csv",
                mime="text/csv",
            )

            # Cleanup
            os.unlink(tmp_candidates)
            os.unlink(tmp_output)

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)

elif candidates_data is None:
    st.info("Upload a candidates JSONL file or load the sample to get started.")
