"""Planning-grade delta-base / index quality pooling for product blenders.

Gasoline RON and sulfur are the MVP targets. Algebra:

  Delta-base (classic PIMS-style row form)
  ----------------------------------------
  Choose a base quality Q_base (from a named base stream or fixed reference).
  Component deltas:  δ_i = Q_i − Q_base
  Volume pool V = Σ x_i
  Blend quality: Q = Q_base + (Σ δ_i x_i) / V
  Spec Q ≥ Q_min  ⇔  Σ δ_i x_i  ≥  (Q_min − Q_base) · V
  Spec Q ≤ Q_max  ⇔  Σ δ_i x_i  ≤  (Q_max − Q_base) · V

  When δ_i = Q_i − Q_base this is algebraically identical to linear volume
  pooling Σ Q_i x_i  ?  Q_spec · V. The delta-base *form* is what PIMS matrices
  use so that a base recipe can be updated without rewriting every column.

  Index pooling (RON only, optional)
  ----------------------------------
  Map RON → blending index BI(RON), average BI by volume, invert for reported
  RON. LP constraint for min RON uses BI directly:

      Σ BI(RON_i) x_i  ≥  BI(min_ron) · V

  if BI is strictly increasing. Default index:

      BI(r) = (r − ron_floor) / (ron_ceiling − r)     (Ethyl-style rational)

  with defaults floor=0, ceiling=120. Identity BI(r)=r recovers linear/delta-base.

Limitations vs full Aspen PIMS delta-base recursion
---------------------------------------------------
This MVP is **single-level, fixed-assay, non-recursive**:

1. Component properties are **fixed** in routing.json (planning-grade assays).
   PIMS re-estimates intermediate qualities when upstream recipes/severities
   change (composition → property response → re-blend).
2. **No multi-tank recursion**: intermediate tanks do not recompute quality
   from inflows and feed that quality into the next pool within the same LP.
   Full PIMS delta-base chains those pools (and may need successive LP / SLP
   when responses are nonlinear).
3. **No multi-property octane engine**: R+M/2, sensitivity, aromatics caps,
   benzene, RVP, distillation, driveability index, etc. are out of scope.
4. **Sulfur is volume-weighted wt%** (planning approximation). True mass-basis
   S needs density · volume weights.
5. **Diesel/FO** still use simple soft-HDT linear S credits in full_plant;
   only gasoline RON/S go through this module.
6. Index mode is a **planning index**, not a certified laboratory blending model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

# Optional pulp import kept soft so pure-math helpers work without solver.
try:
    import pulp
except ImportError:  # pragma: no cover
    pulp = None  # type: ignore


# ---------------------------------------------------------------------------
# Index transforms
# ---------------------------------------------------------------------------

def ron_blending_index(
    ron: float,
    *,
    mode: str = "identity",
    ron_floor: float = 0.0,
    ron_ceiling: float = 120.0,
) -> float:
    """Map research octane to a volume-blendable index.

    Modes:
      - identity: BI = RON (linear / delta-base equivalent)
      - ethyl / rational: BI = (RON - floor) / (ceiling - RON)
    """
    m = (mode or "identity").strip().lower()
    r = float(ron)
    if m in ("identity", "linear", "delta_base", "delta-base", "none"):
        return r
    if m in ("ethyl", "rational", "index", "ron_index"):
        den = max(float(ron_ceiling) - r, 1e-6)
        return (r - float(ron_floor)) / den
    raise ValueError(f"unknown RON index mode: {mode!r}")


def ron_from_blending_index(
    bi: float,
    *,
    mode: str = "identity",
    ron_floor: float = 0.0,
    ron_ceiling: float = 120.0,
) -> float:
    """Inverse of ron_blending_index (for reporting / closed-form checks)."""
    m = (mode or "identity").strip().lower()
    if m in ("identity", "linear", "delta_base", "delta-base", "none"):
        return float(bi)
    if m in ("ethyl", "rational", "index", "ron_index"):
        # BI = (r - floor)/(ceiling - r)  →  r = (BI*ceiling + floor)/(BI + 1)
        b = float(bi)
        return (b * float(ron_ceiling) + float(ron_floor)) / max(b + 1.0, 1e-12)
    raise ValueError(f"unknown RON index mode: {mode!r}")


# ---------------------------------------------------------------------------
# Spec / component data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityComponent:
    """Fixed planning-grade qualities for one blendstock stream."""

    stream: str
    ron: float
    sulfur_wt: float

    def delta_ron(self, base_ron: float) -> float:
        return float(self.ron) - float(base_ron)

    def delta_sulfur(self, base_s: float) -> float:
        return float(self.sulfur_wt) - float(base_s)


@dataclass
class GasolineQualityConfig:
    """How gasoline RON/S are enforced in the LP."""

    min_ron: float = 87.0
    max_sulfur_wt: float = 0.01
    # delta_base | linear | index
    model: str = "delta_base"
    base_stream: str = "reformate"
    # When base stream absent from pool, fall back to this absolute base RON/S
    base_ron: Optional[float] = None
    base_sulfur_wt: Optional[float] = None
    # RON index (only if model == index)
    ron_index_mode: str = "ethyl"
    ron_floor: float = 0.0
    ron_ceiling: float = 120.0

    @classmethod
    def from_routing(cls, routing: Mapping[str, Any]) -> "GasolineQualityConfig":
        gspec = dict((routing.get("product_quality_specs") or {}).get("gasoline") or {})
        qcfg = dict(routing.get("quality_model") or {})
        gas_q = dict(qcfg.get("gasoline") or qcfg)
        model = str(
            gas_q.get("model")
            or gspec.get("model")
            or qcfg.get("model")
            or "delta_base"
        )
        return cls(
            min_ron=float(gspec.get("min_ron", gas_q.get("min_ron", 87.0))),
            max_sulfur_wt=float(
                gspec.get("max_sulfur_wt", gas_q.get("max_sulfur_wt", 0.01))
            ),
            model=model,
            base_stream=str(
                gas_q.get("base_stream")
                or gspec.get("base_stream")
                or "reformate"
            ),
            base_ron=(
                float(gas_q["base_ron"])
                if gas_q.get("base_ron") is not None
                else (float(gspec["base_ron"]) if gspec.get("base_ron") is not None else None)
            ),
            base_sulfur_wt=(
                float(gas_q["base_sulfur_wt"])
                if gas_q.get("base_sulfur_wt") is not None
                else (
                    float(gspec["base_sulfur_wt"])
                    if gspec.get("base_sulfur_wt") is not None
                    else None
                )
            ),
            ron_index_mode=str(gas_q.get("ron_index_mode", "ethyl")),
            ron_floor=float(gas_q.get("ron_floor", 0.0)),
            ron_ceiling=float(gas_q.get("ron_ceiling", 120.0)),
        )


def load_component_qualities(
    routing: Mapping[str, Any],
    streams: Sequence[str],
    *,
    defaults: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, QualityComponent]:
    """Pull RON/S for named streams from routing.component_properties."""
    defaults = defaults or {}
    cp = routing.get("component_properties") or {}
    out: Dict[str, QualityComponent] = {}
    for s in streams:
        row = dict(cp.get(s) or {})
        dflt = dict(defaults.get(s) or {})
        out[s] = QualityComponent(
            stream=s,
            ron=float(row.get("ron", dflt.get("ron", 0.0))),
            sulfur_wt=float(row.get("sulfur_wt", dflt.get("sulfur_wt", 0.0))),
        )
    return out


def resolve_base(
    components: Mapping[str, QualityComponent],
    cfg: GasolineQualityConfig,
) -> Tuple[str, float, float]:
    """Return (base_label, base_ron, base_sulfur_wt)."""
    if cfg.base_stream in components:
        c = components[cfg.base_stream]
        return cfg.base_stream, float(c.ron), float(c.sulfur_wt)
    # explicit absolute base
    if cfg.base_ron is not None and cfg.base_sulfur_wt is not None:
        return "absolute", float(cfg.base_ron), float(cfg.base_sulfur_wt)
    # fall back to reformate defaults or first component
    if components:
        # prefer reformate-like if listed under another key
        for pref in ("reformate", "base", "gasoline_base"):
            if pref in components:
                c = components[pref]
                return pref, float(c.ron), float(c.sulfur_wt)
        # volume-weighted not available here — use max RON as high-quality base
        best = max(components.values(), key=lambda c: c.ron)
        return best.stream, float(best.ron), float(best.sulfur_wt)
    return "absolute", float(cfg.base_ron or 100.0), float(cfg.base_sulfur_wt or 0.0005)


def component_deltas(
    components: Mapping[str, QualityComponent],
    base_ron: float,
    base_s: float,
) -> Dict[str, Dict[str, float]]:
    return {
        s: {
            "ron": c.ron,
            "sulfur_wt": c.sulfur_wt,
            "delta_ron": c.delta_ron(base_ron),
            "delta_sulfur_wt": c.delta_sulfur(base_s),
        }
        for s, c in components.items()
    }


def blend_quality_closed_form(
    volumes: Mapping[str, float],
    components: Mapping[str, QualityComponent],
    cfg: GasolineQualityConfig,
) -> Dict[str, Any]:
    """Closed-form blend RON/S for a fixed recipe (tests / reporting)."""
    base_label, base_ron, base_s = resolve_base(components, cfg)
    vtot = sum(max(0.0, float(volumes.get(s, 0.0))) for s in components) or 0.0
    if vtot <= 1e-12:
        return {
            "volume": 0.0,
            "ron": base_ron,
            "sulfur_wt": base_s,
            "base_stream": base_label,
            "model": cfg.model,
        }

    model = (cfg.model or "delta_base").strip().lower().replace("-", "_")
    if model in ("index", "ron_index"):
        bi_sum = 0.0
        for s, c in components.items():
            x = max(0.0, float(volumes.get(s, 0.0)))
            bi_sum += ron_blending_index(
                c.ron,
                mode=cfg.ron_index_mode,
                ron_floor=cfg.ron_floor,
                ron_ceiling=cfg.ron_ceiling,
            ) * x
        ron = ron_from_blending_index(
            bi_sum / vtot,
            mode=cfg.ron_index_mode,
            ron_floor=cfg.ron_floor,
            ron_ceiling=cfg.ron_ceiling,
        )
    else:
        # delta-base / linear
        d_sum = sum(
            components[s].delta_ron(base_ron) * max(0.0, float(volumes.get(s, 0.0)))
            for s in components
        )
        ron = base_ron + d_sum / vtot

    s_sum = sum(
        components[s].delta_sulfur(base_s) * max(0.0, float(volumes.get(s, 0.0)))
        for s in components
    )
    sulfur = base_s + s_sum / vtot
    return {
        "volume": vtot,
        "ron": ron,
        "sulfur_wt": sulfur,
        "base_stream": base_label,
        "base_ron": base_ron,
        "base_sulfur_wt": base_s,
        "model": model,
    }


# ---------------------------------------------------------------------------
# PuLP constraint builders
# ---------------------------------------------------------------------------

@dataclass
class QualityConstraintMeta:
    """Metadata for dual names + planner report."""

    model: str
    base_stream: str
    base_ron: float
    base_sulfur_wt: float
    min_ron: float
    max_sulfur_wt: float
    ron_constraint: str
    sulfur_constraint: str
    deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ron_index_mode: str = "identity"
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "base_stream": self.base_stream,
            "base_ron": self.base_ron,
            "base_sulfur_wt": self.base_sulfur_wt,
            "min_ron": self.min_ron,
            "max_sulfur_wt": self.max_sulfur_wt,
            "ron_constraint": self.ron_constraint,
            "sulfur_constraint": self.sulfur_constraint,
            "deltas": self.deltas,
            "ron_index_mode": self.ron_index_mode,
            "notes": list(self.notes),
        }


def add_gasoline_quality_constraints(
    prob: Any,
    *,
    product_var: Any,
    volume_vars: Mapping[str, Any],
    components: Mapping[str, QualityComponent],
    cfg: Optional[GasolineQualityConfig] = None,
    ron_name: str = "qual_gas_min_ron",
    sulfur_name: str = "qual_gas_max_s",
) -> QualityConstraintMeta:
    """Add gasoline RON + S rows to a PuLP problem.

    volume_vars maps stream → non-negative pulp variable (component bbl).
    product_var is the gasoline pool volume (should equal Σ volume_vars).
    """
    if pulp is None:  # pragma: no cover
        raise RuntimeError("pulp is required to add quality constraints")

    cfg = cfg or GasolineQualityConfig()
    base_label, base_ron, base_s = resolve_base(components, cfg)
    deltas = component_deltas(components, base_ron, base_s)
    model = (cfg.model or "delta_base").strip().lower().replace("-", "_")
    notes: List[str] = [
        "Single-level fixed-assay delta-base/index MVP; not full PIMS recursion.",
    ]

    # --- RON ---
    if model in ("index", "ron_index"):
        bi_terms = []
        for s, c in components.items():
            if s not in volume_vars:
                continue
            bi = ron_blending_index(
                c.ron,
                mode=cfg.ron_index_mode,
                ron_floor=cfg.ron_floor,
                ron_ceiling=cfg.ron_ceiling,
            )
            bi_terms.append(bi * volume_vars[s])
        bi_min = ron_blending_index(
            cfg.min_ron,
            mode=cfg.ron_index_mode,
            ron_floor=cfg.ron_floor,
            ron_ceiling=cfg.ron_ceiling,
        )
        if bi_terms:
            prob += pulp.lpSum(bi_terms) >= bi_min * product_var, ron_name
        notes.append(
            f"RON index mode={cfg.ron_index_mode}: "
            f"Σ BI(RON_i) x_i ≥ BI({cfg.min_ron}) · V"
        )
        ron_index_mode = cfg.ron_index_mode
    else:
        # delta-base (and linear, same algebra)
        # Σ δ_i x_i ≥ (min_ron − Q_base) · V
        d_terms = []
        for s, c in components.items():
            if s not in volume_vars:
                continue
            d_terms.append(c.delta_ron(base_ron) * volume_vars[s])
        rhs_coeff = float(cfg.min_ron) - float(base_ron)
        if d_terms:
            prob += pulp.lpSum(d_terms) >= rhs_coeff * product_var, ron_name
        notes.append(
            f"Delta-base RON vs base={base_label}@{base_ron}: "
            f"Σ δ_ron x ≥ ({cfg.min_ron} − {base_ron}) · V"
        )
        if model in ("linear", "volume_linear"):
            notes.append("model=linear is algebraically identical to delta_base with δ=Q−Q_base.")
        ron_index_mode = "identity"

    # --- Sulfur (always delta-base / linear; no index) ---
    s_terms = []
    for s, c in components.items():
        if s not in volume_vars:
            continue
        s_terms.append(c.delta_sulfur(base_s) * volume_vars[s])
    s_rhs = float(cfg.max_sulfur_wt) - float(base_s)
    if s_terms:
        # Σ δ_s x_i ≤ (max_s − S_base) · V
        prob += pulp.lpSum(s_terms) <= s_rhs * product_var, sulfur_name
    notes.append(
        f"Delta-base S vs base={base_label}@{base_s}: "
        f"Σ δ_s x ≤ ({cfg.max_sulfur_wt} − {base_s}) · V"
    )

    return QualityConstraintMeta(
        model=model if model not in ("linear", "volume_linear") else "delta_base",
        base_stream=base_label,
        base_ron=base_ron,
        base_sulfur_wt=base_s,
        min_ron=float(cfg.min_ron),
        max_sulfur_wt=float(cfg.max_sulfur_wt),
        ron_constraint=ron_name,
        sulfur_constraint=sulfur_name,
        deltas=deltas,
        ron_index_mode=ron_index_mode,
        notes=notes,
    )


# Defaults used if routing.json is missing a stream
GASOLINE_COMPONENT_DEFAULTS: Dict[str, Dict[str, float]] = {
    "reformate": {"ron": 100.0, "sulfur_wt": 0.0005},
    "cdu_naphtha_light": {"ron": 72.0, "sulfur_wt": 0.005},
    "cdu_naphtha_heavy": {"ron": 58.0, "sulfur_wt": 0.01},
    "fcc_naphtha": {"ron": 93.0, "sulfur_wt": 0.005},
    "coker_naphtha_hdt": {"ron": 74.0, "sulfur_wt": 0.008},
}

DEFAULT_GASOLINE_STREAMS: Tuple[str, ...] = (
    "reformate",
    "cdu_naphtha_light",
    "cdu_naphtha_heavy",
    "fcc_naphtha",
    "coker_naphtha_hdt",
)
