#!/usr/bin/env bash
# J26-IT-449 macOS/Linux setup. Run: bash scripts/setup_team.sh team
set -euo pipefail
cd "$(dirname "$0")/.."

profile="${1:-team}"
case "$profile" in
  team|buddhi) ;;
  *) echo "Usage: bash scripts/setup_team.sh [team|buddhi] [--skip-install]" >&2; exit 2 ;;
esac
skip_install="${2:-}"
if [[ -n "$skip_install" && "$skip_install" != "--skip-install" ]]; then
  echo "Unknown option: $skip_install" >&2
  exit 2
fi

if [[ ! -x .venv/bin/python ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv
  else
    echo "Python 3.12 is required. Install it and retry." >&2
    exit 1
  fi
fi

.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'

if [[ "$skip_install" != "--skip-install" ]]; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements/team-dev.txt
  if [[ "$profile" == "buddhi" ]]; then
    echo "Installing AGN-XAI dependencies (may take time and disk space)..."
    .venv/bin/python -m pip install -r requirements.txt
  fi
  .venv/bin/python scripts/check_team_setup.py --profile "$profile" --check-deps
else
  .venv/bin/python scripts/check_team_setup.py --profile "$profile"
fi

echo "Setup complete. Optional activation: source .venv/bin/activate"
echo "No datasets, checkpoints, authentication or model evaluations were run."
