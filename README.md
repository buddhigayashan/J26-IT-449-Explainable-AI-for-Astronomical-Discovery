# Explainable AI for Astronomical Discovery

**SLIIT Faculty of Computing | Final-Year Research Project | J26-IT-449 (2026)**

This repository is the **shared workspace for a four-member astronomy research project**, not the repository for any one member's component alone.

## Repository structure (proposed migration)

This layout is being reviewed in a **draft pull request**. Until that PR is merged, the default `main` branch still contains the historical Component 4 root files.

```text
components/
  01-edirisooriya-stellar-spectra/       # Component 1 — P.K. Edirisooriya
  02-rajapaksha-galaxy-morphology/      # Component 2 — D.O. Rajapaksha
  03-sewwandi-exoplanet-detection/      # Component 3 — K.H.S.D. Sewwandi
  04-buddhi-radio-galaxies/            # Component 4 — J.A.B.G. Jayawickrama
    config.yaml
    requirements.txt
    src/ scripts/ tests/ demo/ notebooks/ outputs/ report_assets/ docs/
docs/                                     # Group documentation (not component-specific)
integration/                              # Future reviewed integration
README.md                                 # Group overview
.gitignore                                # Repository-wide exclusions
```

The first three component directories currently contain **documentation placeholders only**, not their members' implementations. The Component 4 files are proposed as an **unchanged file-content relocation** from the historical `main` root; this does **not** import newer, divergent work from the `buddhi` branch. Work on the member branches should continue independently until reviewed. Do not assume a combined application or validated group-level results.

For Component 4, run project-relative commands **from inside** `components/04-buddhi-radio-galaxies/` (for example, `python -m pytest tests`) after installing its dependencies. Runtime tests and any CI workflows still need to be checked before merging this migration.

## Research overview

The project investigates explainable and uncertainty-aware artificial intelligence in four astronomical domains. Each member develops and evaluates a **separate research component**. The group aims to compare model performance, explanation quality, and scientific interpretability across the domains, subject to compatible methods and independently verified evidence.

**High-level workflow:** astronomical datasets → domain-specific preprocessing and models → explainability / uncertainty analysis → domain-informed scientific evaluation → group comparison and reporting.

## Research components and team

| Component | Research topic | Researcher | Student ID | Working branch |
| --- | --- | --- | --- | --- |
| 1 | **SHAP-Driven Feature Discovery in Stellar Spectra** | P.K. Edirisooriya | IT23330696 | [`edirisooriya`](../../tree/edirisooriya) |
| 2 | **XAI Audit of Galaxy Morphology Classifiers** | D.O. Rajapaksha | IT23276932 | [`rajapaksha`](../../tree/rajapaksha) |
| 3 | **Bayesian Uncertainty Quantification for Exoplanet Detection** | K.H.S.D. Sewwandi | IT23228726 | [`sewwandi`](../../tree/sewwandi) |
| 4 | **AGN-Aware Radio Source Structure Classification Using CNN + XAI** | J.A.B.G. Jayawickrama | IT23308602 | [`buddhi`](../../tree/buddhi) |

The topics and team allocation follow the group's research presentation. Branch labels are existing GitHub branches; **a branch's existence does not establish that its component is complete or merged**.

### Component scope

- **Component 1 — Stellar spectra:** SDSS/APOGEE spectra, model predictions, SHAP/TreeSHAP wavelength attribution, and astrophysical spectral-line interpretation.
- **Component 2 — Galaxy morphology:** Galaxy10/DECaLS and Galaxy Zoo context; CNN explanations and audits of faithfulness, stability, and alignment with morphology.
- **Component 3 — Exoplanets:** Kepler/TESS light curves, detection models, Bayesian/Monte Carlo Dropout uncertainty, and calibration evaluation.
- **Component 4 — Radio galaxies (RadioMorph AI):** MiraBest/Radio Galaxy Zoo context, FR-I/FR-II CNN classification, Grad-CAM, faithfulness/stability analysis, and AGN core/jet/lobe interpretation.

**RadioMorph AI is Component 4 only; it is not the name of the full group research project.**

## Branch and integration policy

| Branch | Intended role |
| --- | --- |
| [`main`](../../tree/main) | Shared project entry point and approved integration target |
| [`developer`](../../tree/developer) | Shared development / integration coordination |
| [`edirisooriya`](../../tree/edirisooriya) | Component 1 work |
| [`rajapaksha`](../../tree/rajapaksha) | Component 2 work |
| [`sewwandi`](../../tree/sewwandi) | Component 3 work |
| [`buddhi`](../../tree/buddhi) | Component 4 work |
| [`feature/external-radio-input`](../../tree/feature/external-radio-input) | Existing Component 4 feature work |
| [`feature/authentication`](../../tree/feature/authentication) | Existing feature branch; ownership/scope to be confirmed |

**Integration status:** The existence of branches or files must not be treated as evidence of a completed group integration. No member's results should be presented as independently verified until that member's evidence has been reviewed.

### Collaboration workflow

1. Work on your **own assigned branch**, not directly on `main`.
2. Keep component-specific source code, datasets, experiments, metrics, and documentation clearly identified.
3. Document environment requirements and reproducible commands **within your component branch or component documentation**.
4. Do not commit private credentials, raw restricted datasets, oversized model artifacts, or outputs without checking the repository's data and licensing policies.
5. When ready, open a pull request to the agreed integration branch. Review ownership, scientific evidence, compatibility, and conflicts **before** merging.
6. Only update group-level claims, benchmark tables, and shared conclusions after verifying the corresponding member contributions.

## Important note about existing repository contents

This repository was initially set up around **Component 4 (AGN radio-source classification)**. On the current `main` branch, root-level directories, scripts, configurations, figures, and historical README content describe Buddhi's work. This migration proposes moving those historical files to `components/04-buddhi-radio-galaxies/`, **not creating a completed unified four-component application**.

Do **not** assume that the Component 4 `config.yaml`, `requirements.txt`, `src/`, `scripts/`, `demo/`, `outputs/`, or `docs/` apply to all four components. The other members should maintain their own instructions and artifacts until an agreed integration structure is reviewed.

This group README intentionally does not reproduce Component 4's dataset statistics, training commands, checkpoints, or validation scores as group-wide results. For Component 4 details, consult its member branch and scientific reports.

## Project status

**Active multi-component research; group integration and combined evaluation are not yet confirmed.**

This README describes the project's intended structure and collaboration policy. It does not claim that all components, a shared benchmark, a unified dashboard, or public deployment are finished.

## Academic and data-use notes

Dataset licenses, redistribution terms, acknowledgements, and permissions must be checked separately for each component. Publication and deployment require the relevant approvals. Preserve experimental provenance and do not modify frozen scientific artifacts merely to simplify integration.

---

*J26-IT-449 — Explainable AI for Astronomical Discovery | SLIIT Faculty of Computing*
