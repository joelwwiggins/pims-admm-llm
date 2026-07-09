# Plant routing — one-pager

*Canonical routes from [`data/routing.json`](../data/routing.json). Hard rules for the full-plant model (W7).*

Non-math view: crude enters the **CDU**; heavy cuts go to **tanks**, then conversion units (**FCC**, **coker**); naphthas from those units go to more tanks, then the **reformer**; light and finished streams feed the **blender**.

---

## Hard routing rules

| Rule | Path |
|------|------|
| CDU gasoil | **CDU → tank (gasoil) → FCC** |
| FCC naphtha | **FCC → tank (FCC naphtha) → Reformer** |
| CDU resid | **CDU → tank (resid) → Coker** |
| Coker naphtha | **Coker → tank (coker naphtha) → Reformer** |

Straight-run naphtha and distillate from the CDU, plus FCC LCO/slurry, coker gasoil, and reformate, go **directly to the blender** (no intermediate tank in the MVP routing table).

---

## Units (blocks)

| Unit | Role |
|------|------|
| **CDU** | Crude distillation — produces naphtha, distillate, gasoil, resid |
| **TANK_GASOIL** | Intermediate inventory between CDU gasoil and FCC |
| **TANK_RESID** | Intermediate inventory between CDU resid and coker |
| **FCC** | Fluid catalytic cracker — gasoil → naphtha, LCO, slurry |
| **COKER** | Delayed coker — resid → naphtha, gasoil (+ coke, outside blend LP) |
| **TANK_FCC_NAPH** | Intermediate inventory between FCC naphtha and reformer |
| **TANK_COKER_NAPH** | Intermediate inventory between coker naphtha and reformer |
| **REFORMER** | Catalytic reformer — naphtha feeds → reformate |
| **BLENDER** | Finished product pools (gasoline / diesel / fuel oil) |

Order in data:  
`CDU`, `TANK_GASOIL`, `TANK_RESID`, `FCC`, `COKER`, `TANK_FCC_NAPH`, `TANK_COKER_NAPH`, `REFORMER`, `BLENDER`.

---

## Route table (from JSON)

| From | Stream | To | Note |
|------|--------|-----|------|
| CDU | `cdu_gasoil` | TANK_GASOIL | Crude unit gasoil to tank then FCC |
| TANK_GASOIL | `cdu_gasoil` | FCC | Tank feeds FCC |
| CDU | `cdu_resid` | TANK_RESID | Crude unit resid to tank then coker |
| TANK_RESID | `cdu_resid` | COKER | Tank feeds delayed coker |
| FCC | `fcc_naphtha` | TANK_FCC_NAPH | FCC naphtha to tank then reformer |
| TANK_FCC_NAPH | `fcc_naphtha` | REFORMER | FCC naphtha reformer feed |
| COKER | `coker_naphtha` | TANK_COKER_NAPH | Coker naphtha to tank then reformer |
| TANK_COKER_NAPH | `coker_naphtha` | REFORMER | Coker naphtha reformer feed |
| CDU | `cdu_naphtha` | BLENDER | Straight-run naphtha to gasoline pool |
| CDU | `cdu_distillate` | BLENDER | Distillate to diesel pool |
| FCC | `fcc_lco` | BLENDER | LCO to diesel / fuel oil |
| FCC | `fcc_slurry` | BLENDER | Slurry to fuel oil |
| COKER | `coker_gasoil` | BLENDER | Coker gasoil to diesel / FO blend |
| REFORMER | `reformate` | BLENDER | Reformate to gasoline |

---

## Flow sketch

```
Crude
  │
  ▼
 CD U ── cdu_naphtha ──────────────────────────────► BLENDER
  │ ── cdu_distillate ─────────────────────────────► BLENDER
  │
  ├─ cdu_gasoil ─► TANK_GASOIL ─► FCC ─┬─ fcc_naphtha ─► TANK_FCC_NAPH ─┐
  │                                    ├─ fcc_lco ─────────────────────┼─► BLENDER
  │                                    └─ fcc_slurry ──────────────────┤
  │                                                                    │
  └─ cdu_resid ──► TANK_RESID ──► COKER ┬─ coker_naphtha ─► TANK_COKER_NAPH ─┐
                                        └─ coker_gasoil ───────────────────┼─► BLENDER
                                                                           │
                              REFORMER ◄── (FCC + coker naphtha tanks) ────┘
                                 │
                                 └─ reformate ─────────────────────────────► BLENDER
```

---

## Linking streams (ADMM consensus)

Used as plant-wide balance / price signals (λ):

- CDU: `cdu_naphtha`, `cdu_distillate`, `cdu_gasoil`, `cdu_resid`
- FCC: `fcc_naphtha`, `fcc_lco`, `fcc_slurry`
- Coker: `coker_naphtha`, `coker_gasoil`
- Reformer: `reformate`

Source of truth: edit **`data/routing.json`**, then keep this page and the architecture diagram in sync.

---

## Related

- [story.md](story.md) — Smart Team narrative (FCC / coker / reformer / tanks)
- [architecture.md](architecture.md) — full plant diagram + block table
- [../demos/run_full_plant_demo.py](../demos/run_full_plant_demo.py) — mono + dual recovery demo
- [../README.md](../README.md) — install and status

---

*Worker 7 · board `pims-admm-llm-20260708`*
