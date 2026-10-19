# J26-IT-449 — Team Git Workflow

This repository began as Buddhi Jayawickrama's AGN-XAI research component. Its existing root-level `src/`, `demo/`, `scripts/`, `tests/`, `notebooks/`, `outputs/`, and configuration are **not** moved or replaced by the team scaffold.

**Start here:** [SETUP_GUIDE.md](SETUP_GUIDE.md) · [ENVIRONMENT.md](ENVIRONMENT.md) · [CONTRIBUTING.md](../CONTRIBUTING.md)

## Branches
- `main`: stable, reviewed releases; do not push experiments directly.
- `developer`: integration of the team website and research components.
- `buddhi`: J.A.B.G. Jayawickrama (IT23308602), AGN-aware radio source structure CNN/XAI.
- `edirisooriya`: P.K. Edirisooriya (IT23330696).
- `rajapaksha`: D.O. Rajapaksha (IT23276932).
- `sewwandi`: K.H.S.D. Sewwandi (IT23228726).
- `feature/authentication`: future Login/Register implementation, not completed.
- `feature/external-radio-input`: existing Buddhi development history; retain it.

## Contribution process
1. Work on your own branch (or a short-lived feature branch from it).
2. Commit focused changes and run relevant tests.
3. Open a pull request into `developer` for review and integration.
4. Test the integrated website and all components.
5. Merge tested `developer` into `main` via reviewed pull request.

## Safety
- Never force-push `main` or overwrite another member's work.
- Never commit passwords, API keys, raw datasets, model checkpoints, or generated research outputs without explicit review.
- Preserve frozen CNN, evaluation, split and XAI artifacts; avoid rerunning locked TEST evaluations.
- Do not merge or relocate the existing Buddhi root-level implementation without a reviewed migration plan.
- Authentication should be developed and tested on its feature branch, then merged into `developer`.
- The web technology stack is not yet selected; the `website/` folders are placeholders, not an implemented Login/Register system.
- GitHub branch protection and collaborator invitations are separate administrative tasks, not guaranteed by this document.
