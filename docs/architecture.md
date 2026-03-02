## Vulcan Architecture and Directory Overview

This file summarizes the overall architecture and directory structure of the Vulcan project. Read it together with `reconstruction_plan.md` and the `phase*.md` documents.

- Core code is under `src/vulcan/`, organized by domain:
  - `vulcan.framework`: training framework (models, datasets, losses, optimizers, representations, etc.).
  - `vulcan.lang`: language and code parsing/analysis.
  - `vulcan.datacollection`: data and vulnerability collection tools.
  - `vulcan.cli`: command-line entry points for training/validation/benchmark/export.
  - `vulcan.services`: backend services and API wrappers.
- Top-level helper directories:
  - `tools/`: thin CLI wrappers calling `vulcan.cli.*`.
  - `scripts/`: startup/operations scripts calling `vulcan.services.*`.
  - `tests/`: test directories mirroring package structure.
  - `docs/`: user and developer documentation.

For more details (including module responsibilities and migration plans), refer to `reconstruction_plan.md` and `phase1.md` through `phase4.md` in the project root.

