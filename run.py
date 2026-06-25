"""
run.py
──────
Cross-platform alternative to Makefile.
Usage: python run.py <command>

Commands:
  run           Full pipeline on candidates.jsonl
  run-sample    Pipeline on sample_candidates.json
  validate      Validate outputs/ranked_output.csv
  test          Run pytest suite
  download      Download model weights
  sandbox       Launch Streamlit demo
  clean         Remove cache and output files
"""

import sys
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    """Run a shell command using the current venv Python, exit on failure."""
     # Replace 'python' with the actual interpreter running this script
    # This ensures .venv Python is used, not the system/Anaconda Python
    if cmd[0] == "python":
        cmd[0] = sys.executable
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


COMMANDS = {
    "run": [
        "python", "-m", "src.pipeline",
        "--jd",         "data/job_description.txt",
        "--candidates", "data/candidates.jsonl",
        "--output",     "outputs/ranked_output.csv",
    ],
    "run-sample": [
        "python", "-m", "src.pipeline",
        "--jd",         "data/job_description.txt",
        "--candidates", "tests/fixtures/sample_candidates.json",
        "--output",     "outputs/ranked_output_sample.csv",
    ],
    "validate": [
        "python", "scripts/run_validation.py",
        "outputs/ranked_output.csv",
        "data/candidates.jsonl",
    ],
    "test": [
        "python", "-m", "pytest", "tests/", "-v",
        "--cov=src", "--cov-report=term-missing",
    ],
    "download": [
        "python", "scripts/download_models.py",
    ],
    "sandbox": [
        "python", "-m", "streamlit", "run", "sandbox/app.py",
    ],
    "clean": [
        "python", "-c",
        (
            "import shutil, pathlib; "
            "[shutil.rmtree(p, ignore_errors=True) "
            " for p in ['.cache']];"
            "[p.unlink() for p in pathlib.Path('outputs').glob('*.csv') "
            " if p.exists()];"
            "print('Cleaned')"
        ),
    ],
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python run.py <command>")
        print(f"Commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    command = sys.argv[1]
    run_cmd(COMMANDS[command])