# CI workflow (blocked on OAuth `workflow` scope)

Hermes/gh OAuth token scopes: `gist, read:org, repo` — **missing `workflow`**.

GitHub rejects creating/updating `.github/workflows/*` without the `workflow` scope
(error: *refusing to allow an OAuth App to create or update workflow*).

## Install CI (one of)

1. **GitHub UI:** Actions → New workflow → set up a workflow yourself → paste
   contents of [`ci-workflow.example.yml`](ci-workflow.example.yml) as
   `.github/workflows/ci.yml` → Commit to `main`.
2. **PAT with workflow scope:** create fine-grained or classic token including
   `workflow`, then:
   ```bash
   cp docs/ci-workflow.example.yml .github/workflows/ci.yml
   git add .github/workflows/ci.yml && git commit -m "ci: pytest Python 3.11" && git push
   ```
3. **`gh auth refresh -s workflow`** then push the same file.

## Example workflow

See [ci-workflow.example.yml](ci-workflow.example.yml) (pytest on Python 3.11, `PYTHONPATH=src`).

## Wave5 status

- Main tip includes Wave5 all-phases (process-pool, recursive quality v1, residual honesty, portfolio smoke).
- Local suite: 118 passed, 1 skipped (Orin).