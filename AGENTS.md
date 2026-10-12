# Project literature context

For work involving EEG decoding, shared robotic-hand control, experimental
design, or literature citations, read `docs/literature_reference_summary.md`
before making recommendations or changes. It summarizes the 18 distinct sources
from the literature review, including citation-key mappings, source links,
methods, findings, limitations, and relevance to this project.

Use the original linked sources to verify precise claims and numerical results.
Preserve the citation keys and distinguish published findings from proposed
project work. The summary records a correction to the MOABB DOI.

Read `README.md` for the current implementation and evaluation protocol;
literature recommendations do not establish that a feature is implemented.

# Shared repository workflow

This is a shared research project. The GitHub repository is
`AL-DEV06/FRI-BCI-ML-RL-Decoder`; `origin` points to
`https://github.com/AL-DEV06/FRI-BCI-ML-RL-Decoder.git`.
The owner's account is `AL-DEV06`. Partner Mario is `Mohnclizzy`, invited using
`mario4406p@gmail.com`, and has accepted collaborator access.

- Inspect the working tree, current branch, and remote before Git operations.
  Preserve existing local edits and all collaborators' commits.
- Before committing project changes, use a focused `codex/` branch. Keep `main`
  as the shared integration branch; push directly to `main` only when the user
  explicitly requests it.
- Fetch the latest remote state before pushing. Inspect incoming commits and
  integrate relevant changes without discarding the partner's work. Resolve
  conflicts by reading both versions; ask only when their intended behavior
  cannot be reconciled from the available context.
- Push an explicitly named feature branch and prefer a pull request for review
  before merging into `main`. Attach any pull request created for this task.
- Never force-push, rewrite published history, delete shared branches, or use
  destructive Git recovery commands without explicit user authorization.
- Keep commits focused. Run checks appropriate to the changes and inspect the
  staged diff. Exclude EEG downloads unless the user explicitly requests a
  documented, licensed subset; stage only that subset and verify its checksums.
  Exclude virtual environments, caches, generated outputs, exports, credentials,
  and all other ignored files.
- Keep the repository private and preserve both contributors' access. Do not
  alter ownership, visibility, collaborator permissions, or the remote URL
  unless the user explicitly requests that change.
- After a push, verify the remote branch/commit and tracking state. Report what
  was pushed and what remains local; do not claim synchronization or mutual
  access without verification.
