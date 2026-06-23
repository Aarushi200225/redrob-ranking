.PHONY: run run-sample download-models validate test build-artifacts sandbox setup clean

# ── Full pipeline run ─────────────────────────────────────────────────────────
run:
	python -m src.pipeline \
		--jd data/job_description.txt \
		--candidates data/candidates.jsonl \
		--output outputs/ranked_output.csv

# ── Sample run (sandbox / quick test) ────────────────────────────────────────
run-sample:
	python -m src.pipeline \
		--jd data/job_description.txt \
		--candidates tests/fixtures/sample_candidates.json \
		--output outputs/ranked_output_sample.csv

# ── Download all model weights ────────────────────────────────────────────────
download-models:
	python scripts/download_models.py

# ── Validate output CSV format ────────────────────────────────────────────────
validate:
	python scripts/run_validation.py \
		outputs/ranked_output.csv \
		data/candidates.jsonl.gz

# ── Run test suite ────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Build static artifacts ────────────────────────────────────────────────────
build-artifacts:
	python scripts/build_taxonomy.py

# ── Launch Streamlit sandbox ──────────────────────────────────────────────────
sandbox:
	streamlit run sandbox/app.py

# ── Install dependencies ──────────────────────────────────────────────────────
setup:
	pip install -r requirements.txt

# ── Clean intermediate state ──────────────────────────────────────────────────
clean:
	rm -rf .cache/ outputs/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true