# Contributing to J26-IT-449

## Branches and ownership
- `main`: reviewed stable releases only.
- `developer`: reviewed team integration.
- `buddhi`, `edirisooriya`, `rajapaksha`, `sewwandi`: individual member development.
- `feature/authentication`: reserved for implementing shared website auth once architecture is approved.

Start with [Team Setup Guide](docs/SETUP_GUIDE.md). Read [Team Workflow](docs/TEAM_WORKFLOW.md).

## Pull request checklist
- [ ] Target branch is `developer`, unless promoting a reviewed release to `main`.
- [ ] Scope and affected components are explained.
- [ ] Relevant tests were run; report commands and outcomes.
- [ ] No secrets, raw datasets, large checkpoints, or generated outputs were added.
- [ ] Existing frozen evaluation/split/model/XAI artifacts remain unchanged.
- [ ] Integration interface changes are documented and coordinated with affected owners.
- [ ] No unreviewed root-level AGN-XAI files were moved or overwritten.

## Team conventions
- Python: 3.12, four-space indentation, clear docstrings and type hints where useful.
- Keep commits focused; prefer small reviewable PRs.
- Avoid force-push on shared branches and never commit another member's local work without review.
- Keep data and research reproducibility claims grounded in verified outputs.
- Authentication is not implemented; do not describe placeholders as a secure production system.

## Owners
- P.K. Edirisooriya — `edirisooriya`
- D.O. Rajapaksha — `rajapaksha`
- K.H.S.D. Sewwandi — `sewwandi`
- J.A.B.G. Jayawickrama — `buddhi`

Actual collaborator permissions and branch protection rules must be configured separately in GitHub settings.
