"""Multi-level / recursive quality: intermediate tanks → next-pool deltas.

Wave5 W2B planning-grade **single intermediate step** (not full Aspen SLP):

1. Intermediate tank quality is the **volume-weighted** property blend of its
   inflows (leaf assays from ``routing.component_properties``), optional heel,
   and optional process transforms (e.g. soft HDT absolute targets).
2. Recomputed tank quality is written onto the outflow stream that the next
   pool (e.g. gasoline blender) treats as a component.
3. Next-pool **delta-base** rows use the updated component Q (δ = Q − Q_base).

Default plant path remains **fixed-assay**. Enable via
``solve_full_plant_with_recursive_quality(recursive_quality=True)`` (standalone
wrapper — does not require editing ``full_plant.py``) or successive refine /
routing patch helpers.

Limitations vs full PIMS recursion
----------------------------------
* One intermediate level (tank → product pool), not arbitrary depth / SLP.
* Linear volume-weighted properties + absolute HDT targets only.
* Sulfur remains volume-weighted wt% (planning approx).
* Simultaneous LP with endogenous intermediate recipes (bilinear Q·x) is out
  of scope; use open-loop volumes / successive refine.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from .quality_blender import (
    DEFAULT_GASOLINE_STREAMS,
    GASOLINE_COMPONENT_DEFAULTS,
    GasolineQualityConfig,
    QualityComponent,
    blend_quality_closed_form,
    component_deltas,
    load_component_qualities,
    resolve_base,
)

# ---------------------------------------------------------------------------
# Property primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityProps:
    """Minimal planning-grade quality vector (RON + sulfur wt%)."""

    ron: float
    sulfur_wt: float

    def as_dict(self) -> Dict[str, float]:
        return {"ron": float(self.ron), "sulfur_wt": float(self.sulfur_wt)}

    def as_component(self, stream: str) -> QualityComponent:
        return QualityComponent(
            stream=stream, ron=float(self.ron), sulfur_wt=float(self.sulfur_wt)
        )


@dataclass(frozen=True)
class TankInflow:
    """Named volume contribution into an intermediate tank."""

    stream: str
    volume: float

    def __iter__(self):  # unpacking convenience
        yield self.stream
        yield self.volume


@dataclass(frozen=True)
class TransformSpec:
    """Process transform applied after a tank blend (planning-grade).

    Absolute targets (``ron`` / ``sulfur_wt`` set) replace the blended
    property — used for soft-HDT product assays (e.g. coker naph → HDT).
    Relative deltas (``d_ron`` / ``d_sulfur_wt``) adjust when absolute is None.
    """

    name: str = "identity"
    ron: Optional[float] = None
    sulfur_wt: Optional[float] = None
    d_ron: float = 0.0
    d_sulfur_wt: float = 0.0

    def apply(self, q: QualityProps) -> QualityProps:
        r = float(self.ron) if self.ron is not None else float(q.ron) + float(self.d_ron)
        s = (
            float(self.sulfur_wt)
            if self.sulfur_wt is not None
            else float(q.sulfur_wt) + float(self.d_sulfur_wt)
        )
        return QualityProps(ron=r, sulfur_wt=s)


def volume_weighted_quality(
    inflows: Sequence[Tuple[float, QualityProps]],
) -> Tuple[QualityProps, float]:
    """Volume-weighted blend of (volume, QualityProps) pairs.

    Returns ``(blended_props, total_volume)``. Zero total → (0,0) props, 0 vol.
    """
    tot = 0.0
    r_sum = 0.0
    s_sum = 0.0
    for vol, q in inflows:
        v = max(0.0, float(vol))
        if v <= 0.0:
            continue
        tot += v
        r_sum += v * float(q.ron)
        s_sum += v * float(q.sulfur_wt)
    if tot <= 1e-12:
        return QualityProps(0.0, 0.0), 0.0
    return QualityProps(ron=r_sum / tot, sulfur_wt=s_sum / tot), tot


def _as_props(obj: Any) -> QualityProps:
    if isinstance(obj, QualityProps):
        return obj
    if isinstance(obj, QualityComponent):
        return QualityProps(ron=float(obj.ron), sulfur_wt=float(obj.sulfur_wt))
    if isinstance(obj, Mapping):
        return QualityProps(
            ron=float(obj.get("ron", 0.0)),
            sulfur_wt=float(obj.get("sulfur_wt", 0.0)),
        )
    raise TypeError(f"cannot coerce quality from {type(obj)!r}")


def compute_tank_quality(
    inflows: Sequence[Union[TankInflow, Tuple[str, float]]],
    sources: Mapping[str, Any],
    *,
    tank_name: str = "tank",
    heel_volume: float = 0.0,
    heel_quality: Optional[Any] = None,
    transform: Optional[TransformSpec] = None,
) -> QualityProps:
    """Volume-weighted intermediate tank quality from named inflows + optional heel.

    ``sources`` maps stream → QualityComponent | QualityProps | dict.
    Missing source streams contribute nothing (skipped with zero if absent).
    """
    pairs: List[Tuple[float, QualityProps]] = []
    heel_v = max(0.0, float(heel_volume or 0.0))
    if heel_v > 0.0 and heel_quality is not None:
        pairs.append((heel_v, _as_props(heel_quality)))

    for item in inflows:
        if isinstance(item, TankInflow):
            stream, vol = item.stream, item.volume
        else:
            stream, vol = item[0], item[1]
        v = max(0.0, float(vol))
        if v <= 0.0:
            continue
        if stream not in sources:
            continue
        pairs.append((v, _as_props(sources[stream])))

    q, _tot = volume_weighted_quality(pairs)
    if transform is not None:
        q = transform.apply(q)
    # tank_name reserved for future meta; keep signature stable
    _ = tank_name
    return q


# ---------------------------------------------------------------------------
# Quality graph (nodes: unit products, tanks, transforms, pool components)
# ---------------------------------------------------------------------------


@dataclass
class QualityNode:
    """One node in the multi-level quality graph."""

    name: str
    kind: str  # "leaf" | "tank" | "transform" | "pool_component"
    # for leaf: source stream assay key
    source_stream: Optional[str] = None
    # for tank: default production inflow stream(s)
    inflow_streams: Tuple[str, ...] = ()
    # optional heel volume key in volumes dict (e.g. heel_fcc_naph)
    heel_volume_key: Optional[str] = None
    # optional transform after blend
    transform: Optional[TransformSpec] = None
    # for pool_component: which node quality feeds this gasoline component
    feeds_component: Optional[str] = None
    description: str = ""


@dataclass
class QualityGraph:
    """Ordered intermediate-quality graph (single level → pool)."""

    nodes: List[QualityNode] = field(default_factory=list)
    pool_streams: Tuple[str, ...] = DEFAULT_GASOLINE_STREAMS

    def node_names(self) -> List[str]:
        return [n.name for n in self.nodes]

    def get(self, name: str) -> Optional[QualityNode]:
        for n in self.nodes:
            if n.name == name:
                return n
        return None


def build_default_gasoline_quality_graph(
    routing: Optional[Mapping[str, Any]] = None,
) -> QualityGraph:
    """Default Wave5 single-step graph: FCC/coker naph tanks → gasoline pool.

    Topology (planning-grade):
      leaf unit products → intermediate tanks (+ optional heel) →
      optional HDT transform on coker path → gasoline component slots.
    """
    _ = routing  # reserved for routing-driven customization
    soft_hdt = TransformSpec(name="soft_hdt", ron=74.0, sulfur_wt=0.008)
    nodes = [
        QualityNode(
            name="reformate",
            kind="leaf",
            source_stream="reformate",
            feeds_component="reformate",
            description="C5+ reformate (gasoline base).",
        ),
        QualityNode(
            name="cdu_naphtha_light",
            kind="leaf",
            source_stream="cdu_naphtha_light",
            feeds_component="cdu_naphtha_light",
            description="Light SR naphtha direct to gasoline.",
        ),
        QualityNode(
            name="cdu_naphtha_heavy",
            kind="leaf",
            source_stream="cdu_naphtha_heavy",
            feeds_component="cdu_naphtha_heavy",
            description="Heavy SR bypass to gasoline (optional).",
        ),
        QualityNode(
            name="tank_fcc_naph",
            kind="tank",
            source_stream="fcc_naphtha",
            inflow_streams=("fcc_naphtha",),
            heel_volume_key="heel_fcc_naph",
            feeds_component="fcc_naphtha",
            description="FCC naphtha intermediate tank (volume-weighted + heel).",
        ),
        QualityNode(
            name="tank_coker_naph",
            kind="tank",
            source_stream="coker_naphtha",
            inflow_streams=("coker_naphtha",),
            heel_volume_key="heel_coker_naph",
            description="Raw coker naphtha tank before soft HDT.",
        ),
        QualityNode(
            name="coker_naphtha_hdt",
            kind="transform",
            source_stream="coker_naphtha",
            inflow_streams=("coker_naphtha",),
            transform=soft_hdt,
            feeds_component="coker_naphtha_hdt",
            description="Soft-HDT coker naphtha (absolute planning assay).",
        ),
    ]
    return QualityGraph(nodes=nodes, pool_streams=DEFAULT_GASOLINE_STREAMS)


# ---------------------------------------------------------------------------
# Recursive evaluation result
# ---------------------------------------------------------------------------


@dataclass
class RecursiveQualityResult:
    """Closed-form multi-level quality evaluation for reporting / refine."""

    component_qualities: Dict[str, QualityComponent]
    node_qualities: Dict[str, QualityProps]
    deltas: Dict[str, Dict[str, float]]
    base_stream: str
    base_ron: float
    base_sulfur_wt: float
    blend: Dict[str, Any] = field(default_factory=dict)
    volumes_used: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    model: str = "recursive_multi_level_v1"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "base_stream": self.base_stream,
            "base_ron": self.base_ron,
            "base_sulfur_wt": self.base_sulfur_wt,
            "deltas": self.deltas,
            "components": {
                s: {"ron": c.ron, "sulfur_wt": c.sulfur_wt}
                for s, c in self.component_qualities.items()
            },
            "node_qualities": {
                n: q.as_dict() for n, q in self.node_qualities.items()
            },
            "blend": dict(self.blend),
            "volumes_used": dict(self.volumes_used),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def _leaf_sources(
    routing: Mapping[str, Any],
    *,
    defaults: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, QualityComponent]:
    defaults = defaults or GASOLINE_COMPONENT_DEFAULTS
    cp = routing.get("component_properties") or {}
    streams = set(cp.keys()) | set(defaults.keys()) | set(DEFAULT_GASOLINE_STREAMS)
    streams |= {"coker_naphtha", "hdt_naphtha"}
    return load_component_qualities(routing, sorted(streams), defaults=defaults)


def evaluate_recursive_quality(
    routing: Mapping[str, Any],
    volumes: Mapping[str, float],
    *,
    graph: Optional[QualityGraph] = None,
    heel_qualities: Optional[Mapping[str, Any]] = None,
    multi_source_inflows: Optional[Mapping[str, Sequence[Union[TankInflow, Tuple[str, float]]]]] = None,
    blend_volumes: Optional[Mapping[str, float]] = None,
    cfg: Optional[GasolineQualityConfig] = None,
    defaults: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> RecursiveQualityResult:
    """Recompute intermediate tank qualities and feed next-pool deltas.

    Parameters
    ----------
    routing:
        Plant routing (``component_properties``, quality specs).
    volumes:
        Named production / heel volumes (e.g. ``fcc_naphtha``, ``heel_fcc_naph``).
    graph:
        Quality graph; default gasoline single-step graph.
    heel_qualities:
        Optional ``tank_name → QualityComponent|QualityProps`` for heel material.
    multi_source_inflows:
        Optional override of tank inflows: ``tank_name → [TankInflow, ...]``.
        Use this to mix leaf streams into an intermediate tank.
    blend_volumes:
        Recipe for closed-form product pool blend. Defaults to ``volumes``
        projected onto gasoline streams.
    """
    graph = graph or build_default_gasoline_quality_graph(routing)
    sources = _leaf_sources(routing, defaults=defaults)
    heel_qualities = heel_qualities or {}
    multi_source_inflows = multi_source_inflows or {}
    vols = {str(k): max(0.0, float(v)) for k, v in volumes.items()}
    cfg = cfg or GasolineQualityConfig.from_routing(routing)

    node_q: Dict[str, QualityProps] = {}
    notes: List[str] = [
        "Wave5 recursive quality v1: volume-weighted intermediate tanks → pool deltas.",
    ]

    # Seed leaf assays into node_qualities for pure streams
    for n in graph.nodes:
        if n.kind == "leaf" and n.source_stream and n.source_stream in sources:
            node_q[n.name] = _as_props(sources[n.source_stream])

    # Evaluate tanks / transforms in declaration order
    for n in graph.nodes:
        if n.kind == "leaf":
            continue

        # Inflows
        if n.name in multi_source_inflows:
            raw_inflows = list(multi_source_inflows[n.name])
        else:
            raw_inflows = []
            for s in n.inflow_streams or ((n.source_stream,) if n.source_stream else ()):
                if s is None:
                    continue
                raw_inflows.append(TankInflow(s, vols.get(s, 0.0)))

        # Heel
        heel_v = 0.0
        heel_q = None
        if n.heel_volume_key:
            heel_v = vols.get(n.heel_volume_key, 0.0)
        if n.name in heel_qualities:
            heel_q = heel_qualities[n.name]
        elif n.heel_volume_key and n.heel_volume_key in heel_qualities:
            heel_q = heel_qualities[n.heel_volume_key]

        # If heel volume set but no quality, use unit assay of primary stream
        if heel_v > 0.0 and heel_q is None and n.source_stream and n.source_stream in sources:
            heel_q = sources[n.source_stream]

        # Transform-only node without multi-source: apply transform to primary
        if n.kind == "transform" and n.transform is not None and n.name not in multi_source_inflows:
            # Prefer tank_coker_naph quality if already computed
            if "tank_coker_naph" in node_q:
                base_q = node_q["tank_coker_naph"]
            elif n.source_stream and n.source_stream in sources:
                base_q = _as_props(sources[n.source_stream])
            else:
                base_q = QualityProps(0.0, 0.0)
            # If production volume is zero, still report transform of assay
            node_q[n.name] = n.transform.apply(base_q)
            continue

        q = compute_tank_quality(
            raw_inflows,
            sources,
            tank_name=n.name,
            heel_volume=heel_v,
            heel_quality=heel_q,
            transform=n.transform,
        )
        # If no volume at all, fall back to pure source assay (identity)
        tot_in = sum(
            max(0.0, float(i.volume if isinstance(i, TankInflow) else i[1]))
            for i in raw_inflows
        ) + heel_v
        if tot_in <= 1e-12 and n.source_stream and n.source_stream in sources:
            q = _as_props(sources[n.source_stream])
            if n.transform is not None:
                q = n.transform.apply(q)
        node_q[n.name] = q

    # Map node qualities → gasoline components
    component_qualities: Dict[str, QualityComponent] = {}
    # start from fixed assays for all pool streams
    fixed = load_component_qualities(
        routing, list(graph.pool_streams), defaults=defaults or GASOLINE_COMPONENT_DEFAULTS
    )
    for s, c in fixed.items():
        component_qualities[s] = c

    for n in graph.nodes:
        if not n.feeds_component:
            continue
        if n.name in node_q:
            q = node_q[n.name]
            component_qualities[n.feeds_component] = q.as_component(n.feeds_component)
        elif n.source_stream and n.source_stream in sources:
            component_qualities[n.feeds_component] = sources[n.source_stream]

    # Ensure coker HDT / fcc from tanks if mapped
    if "tank_fcc_naph" in node_q:
        component_qualities["fcc_naphtha"] = node_q["tank_fcc_naph"].as_component(
            "fcc_naphtha"
        )
    if "coker_naphtha_hdt" in node_q:
        component_qualities["coker_naphtha_hdt"] = node_q[
            "coker_naphtha_hdt"
        ].as_component("coker_naphtha_hdt")

    base_label, base_ron, base_s = resolve_base(component_qualities, cfg)
    deltas = component_deltas(component_qualities, base_ron, base_s)

    # Closed-form blend
    if blend_volumes is None:
        blend_volumes = {
            s: vols.get(s, 0.0) for s in graph.pool_streams if s in component_qualities
        }
    blend = blend_quality_closed_form(blend_volumes, component_qualities, cfg)

    return RecursiveQualityResult(
        component_qualities=component_qualities,
        node_qualities=node_q,
        deltas=deltas,
        base_stream=base_label,
        base_ron=base_ron,
        base_sulfur_wt=base_s,
        blend=blend,
        volumes_used=dict(vols),
        notes=notes,
    )


def component_overrides_from_recursive(
    rec: RecursiveQualityResult,
) -> Dict[str, Dict[str, float]]:
    """Extract {stream: {ron, sulfur_wt}} overrides for routing patch."""
    return {
        s: {"ron": float(c.ron), "sulfur_wt": float(c.sulfur_wt)}
        for s, c in rec.component_qualities.items()
    }


def patch_routing_component_properties(
    routing: Mapping[str, Any],
    overrides: Mapping[str, Mapping[str, Any]],
    *,
    mark_recursive: bool = True,
) -> Dict[str, Any]:
    """Shallow-copy routing with selected ``component_properties`` updated.

    Does not mutate the input. Other streams are preserved.
    """
    r = deepcopy(dict(routing))
    cp = dict(r.get("component_properties") or {})
    for stream, props in overrides.items():
        row = dict(cp.get(stream) or {})
        if "ron" in props:
            row["ron"] = float(props["ron"])
        if "sulfur_wt" in props:
            row["sulfur_wt"] = float(props["sulfur_wt"])
        if mark_recursive:
            row["recursive_quality"] = True
            src = str(row.get("source") or "")
            if "recursive" not in src.lower():
                row["source"] = (src + "; recursive_quality_recompute").strip("; ")
        cp[stream] = row
    r["component_properties"] = cp
    return r


def successive_recursive_refine(
    routing: Mapping[str, Any],
    volumes: Mapping[str, float],
    *,
    heel_qualities: Optional[Mapping[str, Any]] = None,
    multi_source_inflows: Optional[Mapping[str, Sequence[Any]]] = None,
    blend_volumes: Optional[Mapping[str, float]] = None,
    max_steps: int = 1,
) -> Tuple[Dict[str, Any], RecursiveQualityResult]:
    """Open-loop successive refine: recompute tanks → patch component_properties.

    Returns ``(patched_routing, last_result)``. Marks
    ``quality_model.gasoline.recursive_quality = True`` on the patched routing.
    Multiple steps re-evaluate on the patched assays (identity after step 1 for
    pure volume-weight / absolute HDT transforms).
    """
    r = deepcopy(dict(routing))
    rec: Optional[RecursiveQualityResult] = None
    for _ in range(max(1, int(max_steps))):
        rec = evaluate_recursive_quality(
            r,
            volumes,
            heel_qualities=heel_qualities,
            multi_source_inflows=multi_source_inflows,
            blend_volumes=blend_volumes,
        )
        overrides = component_overrides_from_recursive(rec)
        r = patch_routing_component_properties(r, overrides)
    assert rec is not None
    qcfg = dict(r.get("quality_model") or {})
    gas = dict(qcfg.get("gasoline") or {})
    gas["recursive_quality"] = True
    qcfg["gasoline"] = gas
    qcfg["recursive_quality"] = True
    r["quality_model"] = qcfg
    return r, rec


def evaluate_from_plant_result(
    plant_result: Any,
    routing: Optional[Mapping[str, Any]] = None,
    *,
    heel_qualities: Optional[Mapping[str, Any]] = None,
) -> RecursiveQualityResult:
    """Build volumes from a ``FullPlantResult`` and evaluate recursive quality.

    Standalone post-process hook — does not re-solve the plant.
    """
    if routing is None:
        from .assay_loader import load_routing

        routing = load_routing()

    streams = dict(getattr(plant_result, "streams", None) or {})
    products = dict(getattr(plant_result, "products", None) or {})
    arcs = dict(getattr(plant_result, "arc_flows", None) or {})

    volumes: Dict[str, float] = {}
    # Unit products
    for key in (
        "fcc_naphtha",
        "coker_naphtha",
        "reformate",
        "cdu_naphtha_light",
        "cdu_naphtha_heavy",
        "coker_naphtha_hdt",
    ):
        if key in streams:
            volumes[key] = float(streams[key])

    # Arc aliases when stream table is sparse
    if "fcc_naphtha" not in volumes and "fcc_naph_to_gas" in arcs:
        volumes["fcc_naphtha"] = float(arcs.get("fcc_naph_to_gas", 0.0))
    if "reformate" not in volumes:
        volumes["reformate"] = float(streams.get("reformate", products.get("gasoline", 0.0) * 0.0))

    # Prefer explicit stream production
    for k, v in streams.items():
        volumes.setdefault(k, float(v))

    blend_volumes = {
        "reformate": float(arcs.get("reformate_to_gas", volumes.get("reformate", 0.0))),
        "cdu_naphtha_light": float(
            arcs.get("sr_light_to_gas", volumes.get("cdu_naphtha_light", 0.0))
        ),
        "cdu_naphtha_heavy": float(
            arcs.get("sr_heavy_to_gas", volumes.get("cdu_naphtha_heavy", 0.0))
        ),
        "fcc_naphtha": float(arcs.get("fcc_naph_to_gas", volumes.get("fcc_naphtha", 0.0))),
        "coker_naphtha_hdt": float(
            arcs.get("coker_naph_to_hdt_gas", volumes.get("coker_naphtha_hdt", 0.0))
        ),
    }
    # If arcs missing names, fall back to stream / product split
    if sum(blend_volumes.values()) <= 1e-9 and products.get("gasoline", 0) > 0:
        blend_volumes = {
            "reformate": volumes.get("reformate", 0.0),
            "cdu_naphtha_light": volumes.get("cdu_naphtha_light", 0.0),
            "cdu_naphtha_heavy": 0.0,
            "fcc_naphtha": volumes.get("fcc_naphtha", 0.0),
            "coker_naphtha_hdt": volumes.get("coker_naphtha", 0.0),
        }

    return evaluate_recursive_quality(
        routing,
        volumes,
        heel_qualities=heel_qualities,
        blend_volumes=blend_volumes,
    )


def solve_full_plant_with_recursive_quality(
    *args: Any,
    recursive_quality: bool = True,
    max_refine_steps: int = 1,
    heel_qualities: Optional[Mapping[str, Any]] = None,
    multi_source_inflows: Optional[Mapping[str, Sequence[Any]]] = None,
    **kwargs: Any,
) -> Any:
    """Thin wrapper around ``solve_full_plant`` with optional recursive refine.

    When ``recursive_quality=False``, behaves like ``solve_full_plant`` and
    attaches ``meta['quality_recursive'] = {enabled: False}``.

    When ``True`` (default for this helper):
      1. Solve plant with fixed assays.
      2. Evaluate intermediate tank qualities from solution volumes.
      3. Successive-refine patch ``component_properties``.
      4. Optionally re-solve once with patched routing (``max_refine_steps``).
      5. Attach recursive meta without mutating the default plant path.

    Does **not** edit ``full_plant.py`` — safe optional flag path.
    """
    from .full_plant import solve_full_plant

    # Extract routing if provided so we can patch across steps
    routing = kwargs.get("routing")
    if routing is None:
        from .assay_loader import load_routing

        routing = load_routing()
        kwargs = dict(kwargs)
        kwargs["routing"] = routing

    history: List[Dict[str, Any]] = []

    if not recursive_quality:
        res = solve_full_plant(*args, **kwargs)
        meta = dict(getattr(res, "meta", None) or {})
        meta["quality_recursive"] = {
            "enabled": False,
            "model": "fixed_assay",
            "notes": ["recursive_quality=False — fixed assay path."],
        }
        res.meta = meta
        return res

    res = solve_full_plant(*args, **kwargs)
    history.append(
        {
            "step": 0,
            "objective": float(getattr(res, "objective", 0.0) or 0.0),
            "feasible": bool(getattr(res, "feasible", False)),
            "phase": "initial_fixed_assay",
        }
    )

    last_rec: Optional[RecursiveQualityResult] = None
    patched = dict(routing)

    for step in range(max(1, int(max_refine_steps))):
        last_rec = evaluate_from_plant_result(
            res, patched, heel_qualities=heel_qualities
        )
        # If multi-source overrides provided, re-evaluate with explicit volumes
        if multi_source_inflows is not None:
            streams = dict(getattr(res, "streams", None) or {})
            last_rec = evaluate_recursive_quality(
                patched,
                streams,
                heel_qualities=heel_qualities,
                multi_source_inflows=multi_source_inflows,
            )
        overrides = component_overrides_from_recursive(last_rec)
        patched, last_rec = successive_recursive_refine(
            patched,
            dict(getattr(res, "streams", None) or {}),
            heel_qualities=heel_qualities,
            multi_source_inflows=multi_source_inflows,
            max_steps=1,
        )
        # re-solve with patched assays
        kw2 = dict(kwargs)
        kw2["routing"] = patched
        res = solve_full_plant(*args, **kw2)
        history.append(
            {
                "step": step + 1,
                "objective": float(getattr(res, "objective", 0.0) or 0.0),
                "feasible": bool(getattr(res, "feasible", False)),
                "phase": "refine_with_recomputed_assays",
                "overrides": overrides,
            }
        )

    meta = dict(getattr(res, "meta", None) or {})
    q = dict(meta.get("quality") or {})
    if last_rec is not None:
        q["recursive_deltas"] = last_rec.deltas
        q["recursive_components"] = {
            s: {"ron": c.ron, "sulfur_wt": c.sulfur_wt}
            for s, c in last_rec.component_qualities.items()
        }
        q["recursive_nodes"] = {
            n: p.as_dict() for n, p in last_rec.node_qualities.items()
        }
        # Prefer reporting recursive deltas alongside fixed LP deltas
        if last_rec.deltas:
            q.setdefault("deltas", last_rec.deltas)
    meta["quality"] = q
    meta["quality_recursive"] = {
        "enabled": True,
        "model": "recursive_multi_level_v1",
        "history": history,
        "result": last_rec.as_dict() if last_rec is not None else {},
        "notes": [
            "Optional successive refine: intermediate tank Q → patch assays → re-solve.",
            "Default solve_full_plant remains fixed-assay.",
        ],
    }
    res.meta = meta
    return res


# ---------------------------------------------------------------------------
# Compatibility aliases (docs / thin hooks)
# ---------------------------------------------------------------------------


def recompute_tank_and_pool_deltas(
    routing: Mapping[str, Any],
    volumes: Mapping[str, float],
    **kwargs: Any,
) -> RecursiveQualityResult:
    """Alias for :func:`evaluate_recursive_quality` (docs name)."""
    return evaluate_recursive_quality(routing, volumes, **kwargs)


def apply_recursive_to_routing_quality(
    routing: Mapping[str, Any],
    rec: Union[RecursiveQualityResult, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Patch routing component_properties from a recursive evaluation result."""
    if isinstance(rec, RecursiveQualityResult):
        overrides = component_overrides_from_recursive(rec)
    else:
        overrides = dict(rec.get("components") or rec)
    return patch_routing_component_properties(routing, overrides)


def resolve_gasoline_components(
    routing: Mapping[str, Any],
    streams: Optional[Sequence[str]] = None,
    *,
    recursive_quality: bool = False,
    volumes: Optional[Mapping[str, float]] = None,
    heel_qualities: Optional[Mapping[str, Any]] = None,
    multi_source_inflows: Optional[Mapping[str, Sequence[Any]]] = None,
    defaults: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Tuple[Dict[str, QualityComponent], Dict[str, Any]]:
    """Flag path for full_plant-style callers.

    ``recursive_quality=False`` (default) → fixed assay via
    ``load_component_qualities``. When True, evaluate multi-level tanks and
    return recomputed components + meta.
    """
    streams = list(streams or DEFAULT_GASOLINE_STREAMS)
    defaults = defaults if defaults is not None else GASOLINE_COMPONENT_DEFAULTS
    if not recursive_quality:
        comps = load_component_qualities(routing, streams, defaults=defaults)
        return comps, {
            "recursive_quality": False,
            "mode": "fixed_assay",
            "model": "fixed_assay",
        }
    vols = dict(volumes or {s: 1.0 for s in streams})
    rec = evaluate_recursive_quality(
        routing,
        vols,
        heel_qualities=heel_qualities,
        multi_source_inflows=multi_source_inflows,
        blend_volumes={s: vols.get(s, 0.0) for s in streams},
        defaults=defaults,
    )
    comps = {s: rec.component_qualities[s] for s in streams if s in rec.component_qualities}
    # fill any missing from fixed
    fixed = load_component_qualities(routing, streams, defaults=defaults)
    for s in streams:
        comps.setdefault(s, fixed[s])
    return comps, {
        "recursive_quality": True,
        "mode": "multi_level_volume_weighted",
        "model": rec.model,
        "result": rec.as_dict(),
    }
