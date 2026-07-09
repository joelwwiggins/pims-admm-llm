# Residuals — full-plant mono-oracle ||r|| after Wave4 multi-stream

*How primal residuals, free-disposal shortage residuals, and dual L∞ gaps
relate on the full plant — especially after Wave4 expanded the yield slate.*

Related code:

- `src/pims_admm_llm/admm/residuals.py` — `r = prod − use`, `||s||`, L∞ helper
- `src/pims_admm_llm/admm/pure_plant_admm.py` — free-disposal shortage residual
- `src/pims_admm_llm/models/full_plant.py` — `admm_price_directed_plant` recovery paths
- Linking stream list: `data/routing.json` → `linking_streams`
- Companion floor note: [pure_admm_floor.md](pure_admm_floor.md)

---

## 1. Two recovery paths (do not mix claims)

| Path | `dual_recovery_path` | What λ means | L∞ dual gap |
|------|----------------------|--------------|-------------|
| **Mono-oracle (default)** | `mono-oracle` | Economic duals **recovered from the monolithic LP** (`bal_*`, capacity, quality) | **0 by construction** (`lambda_vs_mono_Linf = 0`) |
| **Pure ADMM (research)** | `pure-admm` | Free λ from multi-block price iteration — **no mono dual injection** | Honest, may be **large** on free-disposal faces |

Demos and API must **label** the path. Default remains mono-oracle until pure
path dual quality is polished ([issue #2](https://github.com/joelwwiggins/pims-admm-llm/issues/2)).

```bash
# default mono-oracle
PYTHONPATH=src python -m demos.run_full_plant_demo

# pure-admm research path
PYTHONPATH=src python -m demos.run_full_plant_demo --pure-admm
```

---

## 2. Residual definitions

### Primal residual (material imbalance)

For each linking stream \(s\):

\[
r_s = \mathrm{prod}_s - \mathrm{use}_s
\qquad
\|r\|_2 = \sqrt{\sum_s r_s^2}
\]

Reported as `primal_residual_norm`. Positive \(r_s\) means **oversupply**
(production exceeds consumption proposals); negative means **shortage**.

### Dual residual (consensus step)

\[
s = \rho\,(z^{k+1} - z^k)
\qquad
\|s\|_2 = \text{`dual_residual_norm`}
\]

Standard ADMM dual residual on the consensus target \(z\).

### Free-disposal shortage residual

Refinery intermediates often allow **free disposal** (flare, fuel-gas sink,
coke-to-regen heat, FO dump, unused light ends): **prod ≥ use** can be
economically optimal, so exact equality \(r = 0\) on every stream is the wrong
“hard feasibility” metric.

Shortage residual keeps only unmet demand:

\[
\mathrm{short}_s = \max(0,\; \mathrm{use}_s - \mathrm{prod}_s)
\qquad
\|\mathrm{short}\|_2 = \text{`shortage_residual_norm`}
\]

Pure-ADMM stopping prefers **small shortage** (and bounded dual residual),
**not** \(\|r\|\to 0\) on surplus faces. Duals are projected \(\lambda \ge 0\)
(free-disposal duals).

Market-clearing dual update (sign-critical):

\[
\lambda \leftarrow \lambda - \alpha\rho\, r
\quad\text{(oversupply lowers }\lambda\text{; shortage raises }\lambda\text{)}
\]

---

## 3. Dual L∞ = 0 is **not** the same claim as \(\|r\|=0\)

| Metric | Mono-oracle | Pure-ADMM |
|--------|-------------|-----------|
| `lambda_vs_mono_Linf` | **0** — recovery duals **are** mono duals | Honest \(\||\lambda| - |\mathrm{mono}|\|_\infty\); improved vs pre-fix blow-ups, **not** dual recovery |
| `primal_residual_norm` \(\|r\|\) | Near zero when prod/use maps match mono streams | Can remain **O(10)** after Wave4 multi-stream free-disposal faces |
| `shortage_residual_norm` | Often unused on oracle path | Primary “hard balance” residual under free disposal |

**L∞ = 0** on mono-oracle means: *we copied / recovered mono duals and labeled
them.* It does **not** mean pure block ADMM has matched those duals, and it does
**not** by itself prove multi-block primal consensus on every linking stream.

Conversely, a moderate \(\|r\|\) with small shortage can be a **feasible plant
under free disposal** even when L∞ dual gap is nonzero.

---

## 4. Wave4 multi-stream effect on full-plant \(\|r\|\)

Wave4 (`feat(wave4): full unit yield streams, feed poolers, process conditions`)
expanded `linking_streams` beyond classic CDU/FCC/coker/reformer intermediates
to the **full unit yield slate**, for example:

- FCC: `fcc_dry_gas`, `fcc_lpg`, `fcc_naphtha`, `fcc_lco`, `fcc_slurry`, `fcc_coke`
- Coker: gas/LPG/naphtha/gasoil/coke
- Reformer: `reformate`, H2, lights
- Plus HDT / offgas / distillate cuts as listed in `routing.json`

Many of these streams terminate in **sinks** (fuel gas, LPG product, regen heat,
coke credit) with free-disposal-like economics. When residual norms are taken
over the **full multi-stream vector**:

1. **Dimension growth** — more components in \(\|r\|_2\) even if each is small.
2. **Surplus faces** — dry gas / LPG / coke often have `prod > use` in block
   proposals without being “infeasible” in the mono LP sense.
3. **Mono-oracle residual construction** — the default path seeds \(z\) from
   mono stream rates and builds partial `use_map` / `prod_map` for classic
   links; **new Wave4 streams may appear in `links` with incomplete use maps**,
   so reported \(\|r\|\) can reflect bookkeeping imbalance on sink streams, not
   a broken mono solve.
4. **Pure-ADMM tests** explicitly allow larger shortage bounds after Wave4
   (“full yield slate expands free-disposal faces”) — see
   `tests/test_pure_admm_harden.py` (e.g. `||short|| < ~55`).

### How to read a full-plant demo line

```
||r|| primal residual:  …
||s|| dual residual:    …
shortage residual:      …
dual_recovery_path:     mono-oracle | pure-admm
lambda_vs_mono_Linf:    0.0 (oracle) | honest gap (pure)
```

Interpretation checklist:

1. **Path label first.** If `mono-oracle`, L∞ = 0 is expected and uninformative
   about free-λ quality.
2. **Prefer shortage over raw \(\|r\|\)** when free disposal / multi-stream sinks
   are in the linking set.
3. **Wave4+**: large \(\|r\|\) without shortage may be surplus on gas/LPG/coke —
   inspect per-stream prod/use before declaring divergence.
4. **Never claim dual recovery** on pure-admm even if residual is small.

### How to read VERDICT lines

```
VERDICT: PASS — full plant feasible; obj gap≤1%; dual recovery within tolerance;
  rho=… ||r||=… ||s||=0 path=mono-oracle.
```

- **PASS on dual recovery** ⇔ dual L∞ within tolerance (**0 for mono-oracle**).
- **‖r‖ large** ⇔ free-disposal / multi-stream bookkeeping; **not** “wrong duals”
  on the mono-oracle path.

---

## 5. Practical acceptance bands (honest, not dual-oracle)

These match current demos/tests after Wave4; they are **engineering gates**,
not mathematical dual-recovery proofs.

| Mode | Typical gate |
|------|----------------|
| Mono-oracle feasibility | Mono LP feasible; objective match; `dual_recovery_path=mono-oracle`; L∞ = 0 by construction |
| Pure-ADMM residual | \(\|r\|\) bounded (e.g. < ~80); shortage < ~50–55; CDU/FCC/coker feeds active |
| Pure-ADMM dual honesty | `lambda_vs_mono_Linf` well below unhinged pre-fix values; `duals_like_monolithic == {}` |

Wave5 residual work (scale-up, recursive quality, pure dual polish) is tracked
in [issue #2](https://github.com/joelwwiggins/pims-admm-llm/issues/2).

**Pure-ADMM structural L∞ floor** (why free λ cannot hit mono duals to 0;
decision vs free-disposal residual classes after multi-stream align):
[docs/pure_admm_floor.md](pure_admm_floor.md).

---

## 6. One-liner for stakeholders

**Mono-oracle L∞ = 0 means “we trust mono duals as ground truth.”**  
**Multi-stream ‖r‖ after Wave4 measures block proposal imbalance, including
free-disposal surplus on secondary yields — use shortage residual for “hard”
balance, and never equate residual smallness with dual recovery on the pure path.**
