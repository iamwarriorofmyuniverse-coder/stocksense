"""Cross-platform command runner mirroring Makefile targets for Windows, macOS, and Linux."""

import argparse
import subprocess
import sys


def run_cmd(cmd: str) -> None:
    """Execute shell command and forward exit code."""
    print(f"\n>> Executing: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code: {res.returncode}")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="StockSense Cross-Platform Task Runner")
    parser.add_argument(
        "target",
        choices=["setup", "data", "train", "evaluate", "simulate", "serve", "serve-api", "serve-app", "test", "lint"],
        help="Command target to execute",
    )
    args = parser.parse_args()

    targets = {
        "setup": f"{sys.executable} -m pip install -r requirements.txt -r requirements-dev.txt",
        "data": f"{sys.executable} -m src.data.synthetic --output-dir data/synthetic --num-days 180 --num-stores 2 --num-items 30",
        "train": f"{sys.executable} -m src.models.train",
        "evaluate": f"{sys.executable} -m src.evaluation.failure_analysis",
        "simulate": f"{sys.executable} -m src.simulation.run_simulation",
        "serve-api": f"{sys.executable} -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload",
        "serve-app": f"{sys.executable} -m streamlit run app/main.py",
        "serve": f"{sys.executable} -m streamlit run app/main.py",
        "test": f"{sys.executable} -m pytest tests/ -v",
        "lint": f"{sys.executable} -m ruff check src tests",
    }

    run_cmd(targets[args.target])


if __name__ == "__main__":
    main()
