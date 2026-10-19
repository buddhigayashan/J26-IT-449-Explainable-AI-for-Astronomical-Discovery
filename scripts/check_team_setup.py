#!/usr/bin/env python3
"""Read-only, standard-library validation of the shared team workspace.

Never downloads data, imports model code, evaluates TEST, or writes outputs.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Check J26-IT-449 team workspace.")
    parser.add_argument("--profile", choices=("team", "buddhi"), default="team")
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Also verify lightweight shared Python dependencies are installed.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    required = (
        "README.md",
        "requirements.txt",
        "requirements/team-dev.txt",
        ".env.example",
        "docs/SETUP_GUIDE.md",
        "docs/TEAM_WORKFLOW.md",
        "website/frontend/README.md",
        "website/backend/auth/README.md",
        "research/buddhi/README.md",
        "research/edirisooriya/README.md",
        "research/rajapaksha/README.md",
        "research/sewwandi/README.md",
    )
    problems: list[str] = []

    print(f"Repository: {root}")
    print(f"Profile: {args.profile}")
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 12):
        problems.append("Python 3.12 is required for the documented team setup.")

    for name in required:
        if not (root / name).is_file():
            problems.append(f"Missing tracked setup file: {name}")

    if args.check_deps:
        for module in ("pytest", "yaml"):
            if importlib.util.find_spec(module) is None:
                problems.append(f"Missing shared dependency: {module}")

    if (root / ".env").exists():
        print("Local .env: present (contents not inspected)")
    else:
        print("Local .env: absent (optional until website backend is implemented)")

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1

    print("PASS: team workspace checks completed without accessing research data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
