# J26-IT-449 Windows setup (PowerShell).
# Run from the repository root: .\scripts\setup_team.ps1 -Profile team
[CmdletBinding()]
param(
    [ValidateSet("team", "buddhi")]
    [string]$Profile = "team",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
try {
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

    if (-not (Test-Path $venvPython)) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)"
            if ($LASTEXITCODE -ne 0) { throw "Install Python 3.12 before setup." }
            & py -3.12 -m venv .venv
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            & python -c "import sys; assert sys.version_info[:2] == (3, 12)"
            if ($LASTEXITCODE -ne 0) { throw "Python 3.12 is required." }
            & python -m venv .venv
        } else {
            throw "Python not found. Install Python 3.12 first."
        }
        if ($LASTEXITCODE -ne 0) { throw "Could not create .venv." }
    }

    & $venvPython -c "import sys; assert sys.version_info[:2] == (3, 12)"
    if ($LASTEXITCODE -ne 0) {
        throw "Existing .venv is not Python 3.12. Review it before recreating; no files were deleted."
    }

    if (-not $SkipInstall) {
        & $venvPython -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
        & $venvPython -m pip install -r requirements/team-dev.txt
        if ($LASTEXITCODE -ne 0) { throw "Shared dependencies failed to install." }

        if ($Profile -eq "buddhi") {
            Write-Host "Installing AGN-XAI dependencies (may take time and disk space)..."
            & $venvPython -m pip install -r requirements.txt
            if ($LASTEXITCODE -ne 0) { throw "AGN-XAI dependencies failed to install." }
        }
    }

    $checkArgs = @("scripts/check_team_setup.py", "--profile", $Profile)
    if (-not $SkipInstall) { $checkArgs += "--check-deps" }
    & $venvPython @checkArgs
    if ($LASTEXITCODE -ne 0) { throw "Workspace verification failed." }

    Write-Host "Setup complete. Python: .\.venv\Scripts\python.exe"
    Write-Host "Optional activation: .\.venv\Scripts\Activate.ps1"
    Write-Host "No datasets, checkpoints, authentication or model evaluations were run."
} finally {
    Pop-Location
}
