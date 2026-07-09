# Pure-ADMM structural L∞ floor

**Path label:** always `dual_recovery_path = "pure-admm"`  
**Module:** `src/pims_admm_llm/admm/pure_plant_admm.py`  
**Rules:** dual sign `λ − αρ(prod−use)`, L1 consensus, blender sink, `λ ≥ 0`, **no mono dual injection**.

## What “floor” means

On the free-λ path, `lambda_vs_mono_Linf` (L∞ gap of pure ADMM prices vs mono economic / bal shadows) does **not** drive to zero at finite iteration. There is a **structural floor** — a positive gap that remains even when primal residuals are controlled — because pure ADMM is not dual recovery.

| Claim | True? |
|-------|--------|
| Primal residual / shortage can be tightened | Yes (harden + multi-stream accounting) |
| Free λ matches mono duals (L∞ ≈ 0) | **No** — structural floor |
| `duals_like_monolithic` filled on pure path | **Never** |
| Mono is plan truth for objectives / feeds | Yes |

Typical Orin ballpark after wave5 W2C residual alignment (not dual recovery):

- `||shortage||` ≈ 0–O(1) on core liquids when Gauss–Seidel + floor dispose engage  
- `||r||` O(1)–O(10) (free disposal allows leftover liquids; equality not forced)  
- `λ_vs_mono_econ_L∞` often ~10–40 (directional; structural floor remains)  
- Pre-harden unhinged paths were L∞ ~100–460 with exploding residual; post-wave4
  naive multi-stream accounting briefly widened `||short||` toward ~40 before W2C

## Why the floor exists (structural, not a bug)

1. **Free-disposal duals (`λ ≥ 0`)**  
   Multi-stream byproducts (dry gas, LPG, coke, H₂, lights, offgas, HDT) and leftover liquids are free disposal. At optimum those faces are often **slack**; duals are zero or **non-unique**. Comparing free ADMM λ to a particular mono dual basis is not a recovery theorem.

2. **Price-directed block LPs ≠ joint mono LP**  
   CDU / FCC / Coker / Reformer / Blender solve reduced local models with L1 consensus vs `z`. Mono solves one coupled arc-flow LP (pools, quality, process conditions). Even at residual 0 on linking streams, λ geometry need not match mono KKT duals.

3. **Wave4 multi-stream slate widens dual faces**  
   Expanding yields (gas/LPG/coke) adds free-disposal faces. If residual accounting naively treated every `routing.linking_streams` name as an equality balance, `||r||` and L∞ honesty metrics **falsely widen**. Pure path partitions:
   - **core balance links** — liquids duals update on; `r = prod − use`, `short = max(0, use − prod)`
   - **free-disposal byproducts** — produced and auto-sunk (fuel/coke credit); **excluded** from dual ascent equality residual

4. **Netback seeds + box `λ ∈ [0, λ_max]`**  
   Seeds are economic netbacks, not mono duals. Projection and damping keep λ in a planning box; mono bal duals can sit outside that geometry on some faces.

5. **No mono dual injection (hard rule)**  
   Closing L∞ by copying mono duals into λ would fake dual recovery. This path refuses that. For L∞ ≈ 0 by construction use `recovery_path="mono-oracle"`.

## Residual accounting (multi-stream free-disposal)

| Quantity | Scope | Meaning |
|----------|--------|---------|
| `primal_residual_norm` (`\|\|r\|\|`) | core liquids | L2 of `prod − use` |
| `shortage_residual_norm` | core liquids | L2 of `max(0, use − prod)` — hard feasibility stress |
| `free_disposal_oversupply_norm` | byproducts | structural oversupply; **not** a fail if auto-sunk |
| `dual_residual_norm` (`\|\|s\|\|`) | core | Boyd dual residual on `z` updates |

Convergence (pure path): small **shortage** + controlled dual residual. Oversupply on free-disposal faces is allowed (`λ ≥ 0`).

Cheap residual controls **without** mono duals:

- L1 consensus vs `z`  
- Market-clearing dual sign `λ ← λ − αρ r`  
- Blender **floor dispose** sink  
- Gauss–Seidel availability (conversion capped by upstream prod; blender capped by residual inventory)  
- Core vs free-disposal partition so wave4 byproducts do not inflate equality residual

## How to report honesty

Always print:

```text
dual_recovery_path: pure-admm
duals_like_monolithic: {}
||r||, ||s||, ||shortage||, ||fd_over||
λ_vs_mono_econ_L∞   # expect structural floor, not 0
```

Sample honesty string fragment:

```text
pure-admm: λ free of mono duals; … λ_vs_mono_econ_L∞=… (structural L∞ floor — see docs/pure_admm_floor.md; not dual recovery)
```

## When you need L∞ ≈ 0

Use the mono-oracle path:

```python
admm_price_directed_plant(recovery_path="mono-oracle")  # default
```

That path copies mono economic duals for reporting (L∞ gap 0 **by construction**). It is **not** pure ADMM.

## Related

- `references/pure-admm-harden.md` (skill) / harden history  
- `tests/test_pure_admm_harden.py` — residual bounds + path label  
- Wave4 unit streams expand free-disposal faces; residual tests stay **bounded**, not dual-recovery tight  
