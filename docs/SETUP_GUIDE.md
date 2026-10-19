# J26-IT-449 — Team Setup Guide

This is the onboarding guide for the four-member research team. It supports **Windows 11 + GitHub Desktop + VS Code**, with a macOS/Linux option. The repository is private: a collaborator invitation is required before a member can clone it.

## 1. Prerequisites

- GitHub account with accepted repository invitation
- GitHub Desktop and VS Code
- **Python 3.12**, available as `py -3.12` on Windows
- Internet access for first-time package installation
- Sufficient local storage if choosing the heavier Buddhi AGN-XAI dependency profile

Do **not** clone into an existing folder containing unsaved work. Do **not** use a ZIP download for regular team development; cloning preserves Git history and branches.

## 2. Clone and select your branch (GitHub Desktop)

1. Sign into GitHub Desktop using your invited GitHub account.
2. File → Clone repository → URL: `https://github.com/buddhigayashan/J26-IT-449-IT23308602-AGN-XAI.git`
3. Select a local destination and click Clone.
4. Click **Current branch** → select **your own branch** (use Fetch origin if it is not visible):
   - P.K. Edirisooriya → `edirisooriya`
   - D.O. Rajapaksha → `rajapaksha`
   - K.H.S.D. Sewwandi → `sewwandi`
   - J.A.B.G. Jayawickrama → `buddhi`
5. Repository → Open in Visual Studio Code.

**Important:** The initial clone checks out `main`. The new team scaffolding is currently on `developer` and the other three member branches. Buddhi's `buddhi` branch was created from his existing research feature branch and has not yet received the shared scaffold. Buddhi must not overwrite or discard his local uncommitted work to change branches.

## 3. Windows: one-command team environment

Open the VS Code PowerShell terminal **in the repository root**:

```powershell
.\scripts\setup_team.ps1 -Profile team
```

This creates a local `.venv` using Python 3.12, installs lightweight shared tooling, and runs read-only setup verification. It does not install PyTorch or download datasets.

If PowerShell blocks local scripts, you can use the documented per-process policy after checking the script, or use the manual commands in section 5. Do not change system-wide policy just for this project.

### Buddhi AGN-XAI environment

Buddhi can run the heavier research profile **after the shared scaffold has been safely integrated into his branch**:

```powershell
.\scripts\setup_team.ps1 -Profile buddhi
```

This additionally installs root `requirements.txt`, including the ML and demo stack. It does **not** download the MiraBest archive, load model checkpoints, or run CNN/XAI.

### macOS / Linux

```bash
bash scripts/setup_team.sh team
# Buddhi only, when appropriate:
bash scripts/setup_team.sh buddhi
```

## 4. Optional local environment configuration

The tracked `.env.example` contains **placeholders only**. No actual authentication/database backend has been selected or implemented yet. It is not required to run the setup check.

For future website development, copy the template into an ignored local file:

```powershell
Copy-Item .env.example .env
```

Never commit `.env`, passwords, tokens, API keys, database credentials, or private user data. Do not assume `.env` is automatically loaded by the existing Gradio demo.

## 5. Manual setup (if scripts cannot run)

Windows PowerShell from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/team-dev.txt
.\.venv\Scripts\python.exe scripts/check_team_setup.py --profile team --check-deps
```

Buddhi can additionally install his own research dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

VS Code: Ctrl+Shift+P → Python: Select Interpreter → choose `.venv`.

## 6. Everyday workflow

1. GitHub Desktop → Fetch origin → Pull origin on **your branch**.
2. Edit only your own component or an agreed shared interface.
3. Run relevant tests and review changed files before committing.
4. Commit and push **your branch**, not `main`.
5. Open a pull request from your branch into `developer`.
6. After review and integration tests, promote `developer` into `main`.

See [TEAM_WORKFLOW.md](TEAM_WORKFLOW.md) and [CONTRIBUTING.md](../CONTRIBUTING.md).

## 7. Data, model and research safety

- Raw datasets, local frozen split files, model checkpoints, and generated evaluation artifacts may be absent from a clone. That is intentional.
- The root `src/`, `demo/`, `scripts/`, `tests/` and `requirements.txt` originated with Buddhi's component. They are not automatically team-wide APIs.
- No one should rerun the frozen TEST evaluation or alter locked CNN/XAI artifacts as part of setup.
- The new `website/` and `research/<member>/` directories are planning placeholders, not working Login/Register or integrated ML modules.
- No dataset or model is required for the shared environment verification.

## 8. Troubleshooting

- **Private repository not visible:** ask the repository owner to invite your GitHub username; accept the invitation.
- **Branch not listed:** click Fetch origin in GitHub Desktop and reopen Current branch.
- **Python 3.12 not found:** install Python 3.12, reopen the terminal, then check `py -3.12 --version`.
- **PowerShell execution policy:** use manual setup in section 5 or run the reviewed script with a temporary process-only execution policy.
- **Wrong Python in existing .venv:** setup stops without deleting it; back up/review that environment before recreating.
- **Torch installation is large:** use `-Profile team` unless you need Buddhi's ML stack.
- **Missing dataset/model:** expected on a fresh clone; do not download data or train as part of onboarding.
