# Environment and Configuration Policy

## Supported runtime
- Python 3.12 for the existing AGN-XAI Python component and the lightweight shared setup.
- Every developer uses their own local `.venv/`, never committed to Git.
- The root `requirements.txt` is the **AGN-XAI component's** dependency list, not the minimal requirement for all members.
- `requirements/team-dev.txt` is the shared lightweight onboarding tooling. Members may define additional component-specific requirements later, subject to review.

## Configuration
- `.env.example` is a public-safe template; `.env` is local and ignored.
- No backend currently reads the proposed `APP_*`, `DATABASE_URL`, or `AUTH_SESSION_SECRET` variables automatically.
- Never commit credentials, production connection strings, personal data, or session secrets.
- Future authentication must use server-side secret storage, secure password hashing, session protections, input validation, and rate limiting; no auth is implemented yet.

## Research isolation
- The MiraBest raw archive, frozen split files, local model checkpoints, and generated results are not bootstrapped by the team setup.
- Do not run data download, model training, test evaluation, or XAI as part of environment verification.
- Integration interfaces must be agreed before moving the existing Buddhi research code.

## Reproducibility
- Current root dependency list is not fully version-pinned; the setup guide is an onboarding path, **not** a claim of bit-for-bit reproducibility.
- Capture per-component dependency locks after team agreement and verify them in CI before using them for research claims.
