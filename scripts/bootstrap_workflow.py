#!/usr/bin/env python3
"""Explicit operator-run bootstrap. Does not download model weights or datasets."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", required=True)
    parser.add_argument("--science", action="store_true", help="Install CPU PyTorch and scientific dependencies")
    args = parser.parse_args()
    if not (3, 12) <= sys.version_info[:2] <= (3, 14):
        raise SystemExit("Use Python 3.12–3.14; CI uses 3.12 and local validation records exact interpreter.")
    target = Path(args.venv).expanduser().resolve()
    if target.exists():
        raise SystemExit("Destination exists; choose a new environment to preserve existing work.")
    repo = Path(__file__).resolve().parents[1]
    venv.EnvBuilder(with_pip=True).create(target)
    python = str(target / "bin/python")
    def run(*cmd):
        subprocess.run([python, "-m", "pip", *cmd], check=True)
    if args.science:
        run("install", "torch==2.10.0", "torchvision==0.25.0", "--index-url", "https://download.pytorch.org/whl/cpu")
    run("install", "-r", str(repo / "requirements" / ("workflow-cpu.txt" if args.science else "workflow-light.txt")))
    run("install", "--no-deps", "-e", str(repo))
    print(f"Ready: {python} -m aurora_workflow --help")


if __name__ == "__main__":
    main()
