"""Process-network agents: real area inputs + pushback + cross-unit couples.

This is the "refinery areas talk to each other" layer on top of a solved plant
plan (mono LP / ADMM). Each area receives **actual** feeds, products, duals, and
routing from the plan — not abstract prompts alone.

Hard rules:
- Plant LP / ADMM remains plan truth for rates and feasibility.
- Agents emit structured pushbacks and soft plan-feedback only.
- Cross-unit knowledge is first-class (e.g. FCC naphtha + reformer reformate
  jointly set gasoline octane / RON headroom).

Stub-deterministic: works offline without an LLM API key. Optional LLM can
narrate the same packets later without changing detection logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Structured I/O
# ---------------------------------------------------------------------------

SEVERITY_ORDER = ("info", "watch", "pushback", "critical")


@dataclass
class AreaInput:
    """What this area actually sees from the plan."""

    feeds: Dict[str, float] = field(default_factory=dict)
    products: Dict[str, float] = field(default_factory=dict)
    capacities: Dict[str, float] = field(default_factory=dict)
    util_frac: Dict[str, float] = field(default_factory=dict)
    binding_duals: Dict[str, float] = field(default_factory=dict)
    routing: Dict[str, float] = field(default_factory=dict)
    process_mode: Optional[str] = None
    neighbors: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Pushback:
    """Area refuses or flags a ridiculous / fragile ask."""

    area: str
    code: str
    severity: str  # info | watch | pushback | critical
    message: str
    related_streams: List[str] = field(default_factory=list)
    related_areas: List[str] = field(default_factory=list)
    plan_feedback: Optional[str] = None  # what master/plan should reconsider
    evidence: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AreaReport:
    area: str
    role: str
    inputs: AreaInput
    status: str  # ok | watch | pushback | critical
    summary: str
    wiggle_room: str  # ample | limited | none
    pushbacks: List[Pushback] = field(default_factory=list)
    cross_unit_notes: List[str] = field(default_factory=list)
    soft_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "area": self.area,
            "role": self.role,
            "inputs": self.inputs.to_dict(),
            "status": self.status,
            "summary": self.summary,
            "wiggle_room": self.wiggle_room,
            "pushbacks": [p.to_dict() for p in self.pushbacks],
            "cross_unit_notes": list(self.cross_unit_notes),
            "soft_suggestions": list(self.soft_suggestions),
        }


@dataclass
class ProcessNetworkRound:
    """One coordination round after a plant solve."""

    plant_objective: float
    plant_feasible: bool
    process_pool: Optional[Dict[str, Any]]
    areas: List[AreaReport]
    pushbacks: List[Pushback]
    cross_unit_couples: List[Dict[str, Any]]
    master_summary: str
    plan_feedback: List[str]
    reoptimize_recommended: bool
    severity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plant_objective": float(self.plant_objective),
            "plant_feasible": bool(self.plant_feasible),
            "process_pool": self.process_pool,
            "areas": [a.to_dict() for a in self.areas],
            "pushbacks": [p.to_dict() for p in self.pushbacks],
            "cross_unit_couples": list(self.cross_unit_couples),
            "master_summary": self.master_summary,
            "plan_feedback": list(self.plan_feedback),
            "reoptimize_recommended": bool(self.reoptimize_recommended),
            "severity": self.severity,
            "n_pushbacks": len(self.pushbacks),
            "n_areas": len(self.areas),
        }


# ---------------------------------------------------------------------------
# Helpers from plant result
# ---------------------------------------------------------------------------


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _util(feed: float, cap: float) -> float:
    if cap <= 1e-9:
        return 0.0
    return max(0.0, min(1.5, feed / cap))


def _top_duals(duals: Mapping[str, float], *, n: int = 8, thr: float = 1e-4) -> Dict[str, float]:
    items = [(k, _f(v)) for k, v in duals.items() if abs(_f(v)) >= thr]
    items.sort(key=lambda kv: -abs(kv[1]))
    return {k: v for k, v in items[:n]}


def _severity_max(levels: Sequence[str]) -> str:
    best = "info"
    for s in levels:
        if SEVERITY_ORDER.index(s) > SEVERITY_ORDER.index(best):
            best = s
    return best


def _wiggle_from_util_and_duals(
    util: float, binding: Mapping[str, float], *, dual_thr: float = 0.05
) -> str:
    big = any(abs(v) >= dual_thr for v in binding.values())
    if util >= 0.98 or (util >= 0.90 and big):
        return "none"
    if util >= 0.75 or big:
        return "limited"
    return "ample"


# ---------------------------------------------------------------------------
# Build per-area situations from FullPlantResult-like object / dict
# ---------------------------------------------------------------------------


def _plant_as_dict(plant: Any) -> Dict[str, Any]:
    if isinstance(plant, Mapping):
        return dict(plant)
    return {
        "objective": getattr(plant, "objective", 0.0),
        "feasible": getattr(plant, "feasible", False),
        "status": getattr(plant, "status", ""),
        "crude_rates": dict(getattr(plant, "crude_rates", {}) or {}),
        "unit_feeds": dict(getattr(plant, "unit_feeds", {}) or {}),
        "streams": dict(getattr(plant, "streams", {}) or {}),
        "products": dict(getattr(plant, "products", {}) or {}),
        "duals": dict(getattr(plant, "duals", {}) or {}),
        "quality_duals": dict(getattr(plant, "quality_duals", {}) or {}),
        "economic_shadows": dict(getattr(plant, "economic_shadows", {}) or {}),
        "routing_splits": dict(getattr(plant, "routing_splits", {}) or {}),
        "arc_flows": dict(getattr(plant, "arc_flows", {}) or {}),
        "yields_used": dict(getattr(plant, "yields_used", {}) or {}),
        "meta": dict(getattr(plant, "meta", {}) or {}),
    }


def build_area_situations(
    plant: Any,
    *,
    assays: Optional[Mapping[str, Any]] = None,
) -> Dict[str, AreaInput]:
    """Extract real area inputs from a solved plant."""
    p = _plant_as_dict(plant)
    feeds = p["unit_feeds"]
    streams = p["streams"]
    products = p["products"]
    arcs = p["arc_flows"]
    splits = p["routing_splits"]
    duals = p["duals"]
    qduals = p["quality_duals"]
    econ = p["economic_shadows"]
    meta = p["meta"]
    pool = meta.get("process_pool") if isinstance(meta.get("process_pool"), dict) else None
    caps = dict((assays or {}).get("capacities") or {})
    cdu_cap = _f(caps.get("cdu_kbd", 140.0), 140.0)
    fcc_cap = _f(caps.get("fcc_kbd", 55.0), 55.0)
    cok_cap = _f(caps.get("coker_kbd", 40.0), 40.0)
    ref_cap = _f(caps.get("reformer_kbd", 45.0), 45.0)

    cdu_charge = _f(feeds.get("cdu_charge"))
    fcc_feed = _f(feeds.get("fcc_feed"))
    cok_feed = _f(feeds.get("coker_feed"))
    ref_feed = _f(feeds.get("reformer_feed"))

    situations: Dict[str, AreaInput] = {
        "CDU": AreaInput(
            feeds=dict(p["crude_rates"]),
            products={
                "cdu_naphtha_light": _f(streams.get("cdu_naphtha_light")),
                "cdu_naphtha_heavy": _f(streams.get("cdu_naphtha_heavy")),
                "cdu_distillate": _f(streams.get("cdu_distillate")),
                "cdu_gasoil": _f(streams.get("cdu_gasoil")),
                "cdu_resid": _f(streams.get("cdu_resid")),
            },
            capacities={"cdu_kbd": cdu_cap},
            util_frac={"cdu": _util(cdu_charge, cdu_cap)},
            binding_duals={
                **{k: v for k, v in econ.items() if "cdu" in k.lower()},
                **{
                    k: _f(v)
                    for k, v in duals.items()
                    if k.startswith("cdu_") or k == "cdu_capacity"
                },
            },
            routing={
                "go_frac_fcc": _f(splits.get("go_frac_fcc")),
                "resid_frac_coker": _f(splits.get("resid_frac_coker")),
                "resid_frac_fo": _f(splits.get("resid_frac_fo")),
            },
            neighbors={
                "downstream": ["FCC", "Coker", "Reformer", "Blender"],
            },
        ),
        "FCC": AreaInput(
            feeds={"fcc_feed": fcc_feed, "gasoil_arc": _f(arcs.get("go_to_fcc"))},
            products={
                "fcc_naphtha": _f(streams.get("fcc_naphtha")),
                "fcc_lco": _f(streams.get("fcc_lco")),
                "fcc_slurry": _f(streams.get("fcc_slurry")),
                "fcc_lpg": _f(streams.get("fcc_lpg")),
                "fcc_coke": _f(streams.get("fcc_coke")),
            },
            capacities={"fcc_kbd": fcc_cap},
            util_frac={"fcc": _util(fcc_feed, fcc_cap)},
            binding_duals={
                k: _f(v)
                for k, v in duals.items()
                if "fcc" in k.lower() or k == "fcc_capacity"
            },
            routing={
                "fcc_naph_frac_gas": _f(splits.get("fcc_naph_frac_gas")),
                "fcc_naph_frac_reformer": _f(splits.get("fcc_naph_frac_reformer")),
            },
            process_mode=(pool or {}).get("fcc_mode"),
            neighbors={
                "upstream": ["CDU"],
                "downstream": ["Blender", "Reformer"],
                "octane_couple": "Reformer",
            },
        ),
        "Coker": AreaInput(
            feeds={"coker_feed": cok_feed, "resid_arc": _f(arcs.get("resid_to_coker"))},
            products={
                "coker_naphtha": _f(streams.get("coker_naphtha")),
                "coker_gasoil": _f(streams.get("coker_gasoil")),
                "coker_coke": _f(streams.get("coker_coke")),
            },
            capacities={"coker_kbd": cok_cap},
            util_frac={"coker": _util(cok_feed, cok_cap)},
            binding_duals={
                k: _f(v)
                for k, v in duals.items()
                if "coker" in k.lower() or k == "coker_capacity"
            },
            routing={
                "resid_frac_coker": _f(splits.get("resid_frac_coker")),
                "resid_frac_fo": _f(splits.get("resid_frac_fo")),
            },
            process_mode=(pool or {}).get("coker_mode"),
            neighbors={"upstream": ["CDU"], "downstream": ["Blender"]},
        ),
        "Reformer": AreaInput(
            feeds={"reformer_feed": ref_feed},
            products={
                "reformate": _f(streams.get("reformate")),
                "reformer_h2": _f(streams.get("reformer_h2")),
            },
            capacities={"reformer_kbd": ref_cap},
            util_frac={"reformer": _util(ref_feed, ref_cap)},
            binding_duals={
                k: _f(v)
                for k, v in duals.items()
                if "reform" in k.lower() or k == "reformer_capacity"
            },
            routing={},
            neighbors={
                "upstream": ["CDU", "FCC"],
                "downstream": ["Blender"],
                "octane_couple": "FCC",
            },
        ),
        "Blender": AreaInput(
            feeds={
                "gasoline": _f(products.get("gasoline")),
                "diesel": _f(products.get("diesel")),
                "fuel_oil": _f(products.get("fuel_oil")),
            },
            products=dict(products),
            capacities={},
            util_frac={},
            binding_duals={
                **{k: _f(v) for k, v in qduals.items()},
                **{
                    k: _f(v)
                    for k, v in duals.items()
                    if k.startswith("qual_") or "blend" in k.lower()
                },
            },
            routing={},
            neighbors={
                "upstream": ["CDU", "FCC", "Coker", "Reformer"],
                "quality_specs": ["RON", "sulfur"],
            },
        ),
    }
    return situations


# ---------------------------------------------------------------------------
# Area evaluation + pushback rules
# ---------------------------------------------------------------------------


def _evaluate_cdu(inp: AreaInput, plant: Mapping[str, Any]) -> AreaReport:
    push: List[Pushback] = []
    util = _f(inp.util_frac.get("cdu"))
    wiggle = _wiggle_from_util_and_duals(util, inp.binding_duals)
    resid_fo = _f(inp.routing.get("resid_frac_fo"))
    resid_cok = _f(inp.routing.get("resid_frac_coker"))
    notes: List[str] = []
    if resid_fo >= 0.99 and resid_cok < 0.01:
        notes.append(
            "All resid routed to FO — Coker sees zero charge; conversion severity "
            "and coker naphtha quality path are offline for this plan."
        )
    if util >= 0.98:
        push.append(
            Pushback(
                area="CDU",
                code="cdu_capacity_binding",
                severity="watch",
                message=(
                    f"CDU at {util:.0%} of capacity — little charge headroom; "
                    "crude slate changes will force product rebalance downstream."
                ),
                related_streams=["cdu_charge"],
                related_areas=["FCC", "Coker", "Reformer", "Blender"],
                plan_feedback="Consider crude slate or CDU capacity if demand grows.",
                evidence={"util_frac": util},
            )
        )
    status = _severity_max([p.severity for p in push] or ["info"])
    if status == "info" and notes:
        status = "watch"
    return AreaReport(
        area="CDU",
        role="Crude charge + cut yields; sets intermediate slate for conversion",
        inputs=inp,
        status=status if status != "info" else "ok",
        summary=(
            f"Charge util {util:.0%}; resid swing coker={resid_cok:.0%} FO={resid_fo:.0%}."
        ),
        wiggle_room=wiggle,
        pushbacks=push,
        cross_unit_notes=notes,
        soft_suggestions=[],
    )


def _evaluate_fcc(inp: AreaInput, plant: Mapping[str, Any]) -> AreaReport:
    push: List[Pushback] = []
    notes: List[str] = []
    sugg: List[str] = []
    util = _f(inp.util_frac.get("fcc"))
    feed = _f(inp.feeds.get("fcc_feed"))
    naph = _f(inp.products.get("fcc_naphtha"))
    to_gas = _f(inp.routing.get("fcc_naph_frac_gas"))
    to_ref = _f(inp.routing.get("fcc_naph_frac_reformer"))
    mode = inp.process_mode
    wiggle = _wiggle_from_util_and_duals(util, inp.binding_duals)

    qron = _f((plant.get("quality_duals") or {}).get("qual_gas_min_ron"))
    reformate = _f((plant.get("streams") or {}).get("reformate"))

    # Cross-unit: FCC naphtha + reformer reformate jointly set gasoline octane
    if abs(qron) > 1e-4 and feed > 1e-3 and to_gas >= 0.9:
        notes.append(
            "Gasoline RON is binding (quality dual active). FCC naphtha is going "
            "almost entirely to the gasoline pool — octane is shared with Reformer "
            "reformate; severity/mode and reformer feed are coupled, not independent."
        )
        if mode == "rot_high":
            sugg.append(
                "High-severity FCC (rot_high) already leans naphtha/octane precursors; "
                "further octane asks should go to Reformer rate/severity, not more FCC "
                "naph-to-gas without checking blender RON dual."
            )
        elif mode in (None, "rot_low", "rot_mid"):
            sugg.append(
                "If master wants more gasoline octane, consider higher FCC severity "
                "(process-pool rot_high) **and** reformer charge together — FCC alone "
                "cannot clear RON without reformate quality."
            )

    if feed < 1e-6:
        push.append(
            Pushback(
                area="FCC",
                code="fcc_zero_feed_ridiculous",
                severity="pushback",
                message=(
                    "FCC is asked for conversion economics with ~zero gasoil feed — "
                    "that is a ridiculous local ask; free gasoil/coker-GO or open go_to_fcc."
                ),
                related_streams=["cdu_gasoil", "fcc_feed"],
                related_areas=["CDU", "Blender"],
                plan_feedback="Do not expect FCC products/octane if go_to_fcc is closed.",
                evidence={"fcc_feed": feed},
            )
        )
    if util >= 0.95:
        push.append(
            Pushback(
                area="FCC",
                code="fcc_no_wiggle",
                severity="watch",
                message=f"FCC util {util:.0%} — little feed headroom for replan.",
                related_streams=["fcc_feed"],
                related_areas=["CDU"],
                plan_feedback="Gasoil swing is near FCC capacity.",
                evidence={"util_frac": util},
            )
        )
    if to_ref > 0.05:
        notes.append(
            f"Non-default: {to_ref:.0%} of FCC naphtha toward reformer — "
            "check chemistry defaults and reformer capacity."
        )

    status = _severity_max([p.severity for p in push] or (["watch"] if notes else ["info"]))
    return AreaReport(
        area="FCC",
        role="Gasoil conversion; naphtha to gasoline (octane couple with Reformer)",
        inputs=inp,
        status="ok" if status == "info" else status,
        summary=(
            f"Feed {feed:.1f} kbd util {util:.0%}; naphtha→gas {to_gas:.0%}"
            + (f"; mode={mode}" if mode else "")
            + f"; RON dual={qron:.3f}."
        ),
        wiggle_room=wiggle,
        pushbacks=push,
        cross_unit_notes=notes,
        soft_suggestions=sugg,
    )


def _evaluate_coker(inp: AreaInput, plant: Mapping[str, Any]) -> AreaReport:
    push: List[Pushback] = []
    notes: List[str] = []
    sugg: List[str] = []
    util = _f(inp.util_frac.get("coker"))
    feed = _f(inp.feeds.get("coker_feed"))
    resid_fo = _f(inp.routing.get("resid_frac_fo"))
    resid_cok = _f(inp.routing.get("resid_frac_coker"))
    mode = inp.process_mode
    wiggle = _wiggle_from_util_and_duals(util, inp.binding_duals)
    resid = _f((plant.get("streams") or {}).get("cdu_resid"))
    cok_cap = _f(inp.capacities.get("coker_kbd"), 40.0)

    if feed < 1e-6 and resid > 1.0 and resid_fo >= 0.99:
        push.append(
            Pushback(
                area="Coker",
                code="coker_idled_all_fo",
                severity="pushback",
                message=(
                    f"Resid ~{resid:.1f} kbd is all to FO while Coker is idle. "
                    "If the plan still prices conversion or needs coker naphtha/GO, "
                    "that is a ridiculous ask of Coker with zero charge."
                ),
                related_streams=["cdu_resid", "coker_feed"],
                related_areas=["CDU", "Blender"],
                plan_feedback=(
                    "Either accept FO disposal economics, or open resid_to_coker "
                    "and re-solve (process-pool can change coker yields)."
                ),
                evidence={
                    "coker_feed": feed,
                    "cdu_resid": resid,
                    "resid_frac_fo": resid_fo,
                    "coker_capacity": cok_cap,
                },
            )
        )
        sugg.append(
            "What-if: process_pool_modes / force resid swing to coker and compare margin."
        )
    if feed > 1e-3 and util < 0.15:
        notes.append("Coker lightly loaded — recycle mode has limited leverage.")
    if mode:
        notes.append(f"Coker process-pool mode={mode}.")

    status = _severity_max([p.severity for p in push] or (["watch"] if notes else ["info"]))
    return AreaReport(
        area="Coker",
        role="Resid conversion; naphtha/GO to pools; free-disposal coke",
        inputs=inp,
        status="ok" if status == "info" else status,
        summary=(
            f"Feed {feed:.1f} kbd util {util:.0%}; resid swing coker={resid_cok:.0%} "
            f"FO={resid_fo:.0%}"
            + (f"; mode={mode}" if mode else "")
            + "."
        ),
        wiggle_room=wiggle if feed > 1e-6 else "none",
        pushbacks=push,
        cross_unit_notes=notes,
        soft_suggestions=sugg,
    )


def _evaluate_reformer(inp: AreaInput, plant: Mapping[str, Any]) -> AreaReport:
    push: List[Pushback] = []
    notes: List[str] = []
    sugg: List[str] = []
    util = _f(inp.util_frac.get("reformer"))
    feed = _f(inp.feeds.get("reformer_feed"))
    reformate = _f(inp.products.get("reformate"))
    wiggle = _wiggle_from_util_and_duals(util, inp.binding_duals)
    qron = _f((plant.get("quality_duals") or {}).get("qual_gas_min_ron"))
    fcc_naph = _f((plant.get("streams") or {}).get("fcc_naphtha"))
    fcc_to_gas = _f((plant.get("routing_splits") or {}).get("fcc_naph_frac_gas"))

    notes.append(
        "Reformer is the primary octane machine (reformate). FCC naphtha to gasoline "
        "is a lower-octane blendstock — RON is a **joint** FCC↔Reformer↔Blender constraint."
    )
    if abs(qron) > 1e-4:
        notes.append(
            f"RON quality dual active ({qron:.4f}): octane is tight. "
            f"Reformate {reformate:.1f} kbd must cover pool RON with FCC naphtha "
            f"{fcc_naph:.1f} kbd ({fcc_to_gas:.0%} to gas)."
        )
        if util >= 0.9:
            push.append(
                Pushback(
                    area="Reformer",
                    code="reformer_octane_no_wiggle",
                    severity="critical",
                    message=(
                        "Reformer is both high util and RON-binding — no wiggle room on "
                        "octane. Asking for more gasoline RON without more reformer feed "
                        "or better FCC severity is ridiculous."
                    ),
                    related_streams=["reformate", "fcc_naphtha", "gasoline"],
                    related_areas=["FCC", "Blender", "CDU"],
                    plan_feedback=(
                        "Couple reformer charge and FCC severity for octane; do not "
                        "treat FCC naphtha-to-gas as free octane."
                    ),
                    evidence={
                        "util_frac": util,
                        "qual_gas_min_ron": qron,
                        "reformate": reformate,
                        "fcc_naphtha": fcc_naph,
                    },
                )
            )
        else:
            sugg.append(
                "RON tight but reformer has feed headroom — prefer more heavy SR to "
                "reformer over dumping FCC naphtha quality problems into blender alone."
            )
    if feed < 1e-6:
        push.append(
            Pushback(
                area="Reformer",
                code="reformer_zero_feed",
                severity="pushback",
                message="Reformer has zero feed but is the octane source of record.",
                related_streams=["cdu_naphtha_heavy", "reformate"],
                related_areas=["CDU", "Blender"],
                plan_feedback="Open heavy-SR-to-reformer path before demanding RON.",
                evidence={"reformer_feed": feed},
            )
        )

    status = _severity_max([p.severity for p in push] or (["watch"] if abs(qron) > 1e-4 else ["info"]))
    return AreaReport(
        area="Reformer",
        role="Octane (reformate) for gasoline; couples with FCC naphtha quality",
        inputs=inp,
        status="ok" if status == "info" else status,
        summary=(
            f"Feed {feed:.1f} kbd util {util:.0%}; reformate {reformate:.1f}; "
            f"RON dual={qron:.4f}."
        ),
        wiggle_room=wiggle,
        pushbacks=push,
        cross_unit_notes=notes,
        soft_suggestions=sugg,
    )


def _evaluate_blender(inp: AreaInput, plant: Mapping[str, Any]) -> AreaReport:
    push: List[Pushback] = []
    notes: List[str] = []
    sugg: List[str] = []
    qron = _f(inp.binding_duals.get("qual_gas_min_ron"))
    qs = _f(inp.binding_duals.get("qual_gas_max_s"))
    qd = _f(inp.binding_duals.get("qual_diesel_max_s"))
    gas = _f(inp.products.get("gasoline"))
    diesel = _f(inp.products.get("diesel"))
    fo = _f(inp.products.get("fuel_oil"))

    if abs(qron) > 1e-4:
        push.append(
            Pushback(
                area="Blender",
                code="gasoline_ron_binding",
                severity="pushback",
                message=(
                    f"Gasoline min-RON is binding (dual={qron:.4f}) on {gas:.1f} kbd. "
                    "Blender has **no octane wiggle room** — only reformate quality/rate "
                    "and FCC naphtha severity/routing can fix this; not blender magic."
                ),
                related_streams=["gasoline", "reformate", "fcc_naphtha"],
                related_areas=["Reformer", "FCC"],
                plan_feedback=(
                    "Master must coordinate Reformer + FCC for octane; blender cannot "
                    "manufacture RON without blendstock quality."
                ),
                evidence={"qual_gas_min_ron": qron, "gasoline": gas},
            )
        )
        notes.append(
            "Cross-unit octane couple: Reformer reformate (high RON) + FCC naphtha "
            "(to gas) + light SR → gasoline pool. Treat as one quality system."
        )
    if abs(qs) < 1e-6 and abs(qd) < 1e-6:
        notes.append("Sulfur specs not binding on this plan (duals ~0).")
    if fo > gas * 0.3 and fo > 5.0:
        sugg.append(
            f"Large FO sink ({fo:.1f} kbd) — check resid swing vs coker if FO is a "
            "disposal path rather than valued product."
        )

    wiggle = "none" if abs(qron) > 1e-4 else "limited"
    status = _severity_max([p.severity for p in push] or ["info"])
    return AreaReport(
        area="Blender",
        role="Product pools + quality (RON/S); sinks conversion products",
        inputs=inp,
        status="ok" if status == "info" else status,
        summary=(
            f"Gas {gas:.1f} / diesel {diesel:.1f} / FO {fo:.1f} kbd; "
            f"RON dual={qron:.4f}."
        ),
        wiggle_room=wiggle,
        pushbacks=push,
        cross_unit_notes=notes,
        soft_suggestions=sugg,
    )


_EVALUATORS = {
    "CDU": _evaluate_cdu,
    "FCC": _evaluate_fcc,
    "Coker": _evaluate_coker,
    "Reformer": _evaluate_reformer,
    "Blender": _evaluate_blender,
}


def evaluate_octane_couple(
    plant: Mapping[str, Any],
    areas: Mapping[str, AreaReport],
) -> Dict[str, Any]:
    """Explicit FCC ↔ Reformer ↔ Blender octane coupling report."""
    qron = _f((plant.get("quality_duals") or {}).get("qual_gas_min_ron"))
    fcc = areas.get("FCC")
    ref = areas.get("Reformer")
    bl = areas.get("Blender")
    fcc_naph = _f((fcc.inputs.products if fcc else {}).get("fcc_naphtha"))
    reformate = _f((ref.inputs.products if ref else {}).get("reformate"))
    gas = _f((bl.inputs.products if bl else {}).get("gasoline"))
    tight = abs(qron) > 1e-4
    return {
        "couple_id": "fcc_reformer_blender_octane",
        "name": "FCC naphtha + Reformer reformate → gasoline RON",
        "binding": tight,
        "qual_gas_min_ron_dual": qron,
        "fcc_naphtha_kbd": fcc_naph,
        "reformate_kbd": reformate,
        "gasoline_kbd": gas,
        "message": (
            "RON is tight: octane is a joint FCC+Reformer problem for the blender."
            if tight
            else "RON not binding: octane couple is latent (monitor if naphtha slate shifts)."
        ),
        "areas": ["FCC", "Reformer", "Blender"],
    }


def run_process_network_round(
    plant: Any,
    *,
    assays: Optional[Mapping[str, Any]] = None,
    areas: Optional[Sequence[str]] = None,
) -> ProcessNetworkRound:
    """Run one process-network agent round on a solved plant plan.

    Each area gets real feeds/products/duals; emits pushbacks; master aggregates
    plan feedback. Deterministic (no LLM required).
    """
    p = _plant_as_dict(plant)
    situations = build_area_situations(plant, assays=assays)
    want = list(areas) if areas is not None else list(_EVALUATORS.keys())
    reports: List[AreaReport] = []
    for name in want:
        if name not in _EVALUATORS or name not in situations:
            continue
        reports.append(_EVALUATORS[name](situations[name], p))

    by_name = {r.area: r for r in reports}
    couples = [evaluate_octane_couple(p, by_name)]
    # Attach couple messages onto FCC/Reformer/Blender
    if couples[0]["binding"]:
        for a in ("FCC", "Reformer", "Blender"):
            if a in by_name:
                by_name[a].cross_unit_notes.append(couples[0]["message"])

    all_push: List[Pushback] = []
    for r in reports:
        all_push.extend(r.pushbacks)

    feedback: List[str] = []
    for pb in all_push:
        if pb.plan_feedback:
            feedback.append(f"[{pb.area}/{pb.code}] {pb.plan_feedback}")
    for r in reports:
        for s in r.soft_suggestions:
            feedback.append(f"[{r.area}/suggest] {s}")

    sev = _severity_max(
        [r.status if r.status != "ok" else "info" for r in reports]
        + [pb.severity for pb in all_push]
        or ["info"]
    )
    reopt = any(
        pb.severity in ("pushback", "critical") for pb in all_push
    ) or any(r.wiggle_room == "none" and r.status in ("pushback", "critical") for r in reports)

    no_wiggle = [r.area for r in reports if r.wiggle_room == "none"]
    master = (
        f"Plant obj={_f(p.get('objective')):.2f} feasible={bool(p.get('feasible'))}. "
        f"Areas reporting: {', '.join(r.area for r in reports)}. "
        f"Pushbacks={len(all_push)} severity={sev}. "
    )
    if no_wiggle:
        master += f"No wiggle room: {', '.join(no_wiggle)}. "
    if couples[0]["binding"]:
        master += (
            "Octane couple ACTIVE (FCC naphtha + reformate → gasoline RON). "
            "Do not optimize FCC or Reformer octane in isolation. "
        )
    if reopt:
        master += "Master recommends plan feedback / what-if re-solve before treating plan as robust."
    else:
        master += "No critical ridiculous asks; plan is coherent across areas."

    pool = p.get("meta", {}).get("process_pool") if isinstance(p.get("meta"), dict) else None

    return ProcessNetworkRound(
        plant_objective=_f(p.get("objective")),
        plant_feasible=bool(p.get("feasible")),
        process_pool=pool if isinstance(pool, dict) else None,
        areas=reports,
        pushbacks=all_push,
        cross_unit_couples=couples,
        master_summary=master,
        plan_feedback=feedback,
        reoptimize_recommended=reopt,
        severity=sev if sev != "info" else "ok",
    )


def format_process_network_round(round_: ProcessNetworkRound) -> str:
    """Human-readable console report."""
    lines: List[str] = []
    lines.append("=== Process-network agent round ===")
    lines.append(
        f"Plant obj={round_.plant_objective:.4f} feasible={round_.plant_feasible} "
        f"severity={round_.severity} reoptimize={round_.reoptimize_recommended}"
    )
    if round_.process_pool and round_.process_pool.get("enabled"):
        lines.append(
            f"Process-pool: FCC={round_.process_pool.get('fcc_mode')} "
            f"Coker={round_.process_pool.get('coker_mode')}"
        )
    lines.append("")
    for a in round_.areas:
        lines.append(f"--- {a.area} [{a.status}] wiggle={a.wiggle_room} ---")
        lines.append(f"  role: {a.role}")
        lines.append(f"  {a.summary}")
        for n in a.cross_unit_notes:
            lines.append(f"  cross: {n}")
        for pb in a.pushbacks:
            lines.append(f"  PUSHBACK ({pb.severity}/{pb.code}): {pb.message}")
        for s in a.soft_suggestions:
            lines.append(f"  suggest: {s}")
        lines.append("")
    lines.append("--- Cross-unit couples ---")
    for c in round_.cross_unit_couples:
        lines.append(
            f"  {c['name']}: binding={c['binding']} dual={c.get('qual_gas_min_ron_dual')}"
        )
        lines.append(f"    {c['message']}")
    lines.append("")
    lines.append("--- Master ---")
    lines.append(round_.master_summary)
    if round_.plan_feedback:
        lines.append("Plan feedback:")
        for f in round_.plan_feedback:
            lines.append(f"  • {f}")
    lines.append(
        f"VERDICT: process_network_{'reoptimize' if round_.reoptimize_recommended else 'ok'}"
        f" severity={round_.severity} pushbacks={len(round_.pushbacks)}"
    )
    return "\n".join(lines)
