# Architecture

See README for the high-level ADMM + multi-agent diagram.

## Packages

| Package | Role |
|---------|------|
| `pims_admm_llm.models` | Crude/product data + monolithic / block LP builders |
| `pims_admm_llm.admm` | ADMM coordinator, dual λ updates, consensus z |
| `pims_admm_llm.agents` | LLM sub-agent + master prompts / wrappers |
| `pims_admm_llm.solvers` | PuLP helpers, parallel runners |

## Math guarantees

- Hard constraints always enforced by the LP solver inside each block.
- At ADMM convergence, duals λ match full-LP shadow prices for linking constraints (PIMS-comparable economic signals).
- LLM layer never bypasses solver feasibility.
