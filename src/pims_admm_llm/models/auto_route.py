"""Property-based destination guessing when the flowsheet has no edge.

Given a stream composition (or stream name from the library), score candidate
sinks / units and return a ranked suggestion list. Used by:
  - CDU→FCC plant default exits
  - API POST /api/connect soft scoring
  - graph auto-complete of missing product edges
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .stream_composition import (
    FAMILY_DISTILLATE,
    FAMILY_GASOIL,
    FAMILY_LIGHT_ENDS,
    FAMILY_NAPHTHA,
    FAMILY_PRODUCT,
    FAMILY_RESID,
    FAMILY_SOLID,
    StreamComposition,
    get_stream,
)


# Candidate destinations known to the planning UI / LP sinks
DEFAULT_CANDIDATES: List[str] = [
    "FCC",
    "COKER",
    "REFORMER",
    "HDT_NAPH",
    "BLENDER",
    "GASOLINE",
    "DIESEL",
    "FO",
    "LPG",
    "FUEL_GAS",
    "REGEN_HEAT",
    "COKE_SALES",
    "H2_GRID",
    "SELL",
    "POOL_FCC",
    "POOL_COKER",
    "POOL_REFORMER",
]


@dataclass
class RouteGuess:
    sink: str
    score: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"sink": self.sink, "score": self.score, "reason": self.reason}


def _family_affinity() -> Dict[str, Dict[str, float]]:
    """Base score by stream family → sink."""
    return {
        FAMILY_LIGHT_ENDS: {
            "FUEL_GAS": 0.95,
            "LPG": 0.85,
            "SELL": 0.40,
            "H2_GRID": 0.50,
            "BLENDER": 0.10,
        },
        FAMILY_NAPHTHA: {
            "GASOLINE": 0.80,
            "BLENDER": 0.75,
            "REFORMER": 0.70,
            "HDT_NAPH": 0.65,
            "SELL": 0.35,
            "FUEL_GAS": 0.05,
        },
        FAMILY_DISTILLATE: {
            "DIESEL": 0.90,
            "BLENDER": 0.70,
            "SELL": 0.40,
            "FO": 0.35,
            "FCC": 0.15,
        },
        FAMILY_GASOIL: {
            "FCC": 0.95,
            "POOL_FCC": 0.90,
            "DIESEL": 0.45,
            "SELL": 0.40,
            "FO": 0.25,
            "COKER": 0.10,
        },
        FAMILY_RESID: {
            "FO": 0.80,
            "COKER": 0.85,
            "POOL_COKER": 0.80,
            "SELL": 0.35,
            "FCC": 0.15,  # only if low CCR/metals
        },
        FAMILY_SOLID: {
            "REGEN_HEAT": 0.95,
            "COKE_SALES": 0.85,
            "SELL": 0.40,
        },
        FAMILY_PRODUCT: {
            "SELL": 0.70,
            "BLENDER": 0.50,
        },
    }


def _property_boosts(comp: StreamComposition, sink: str) -> tuple[float, str]:
    """Additional score adjustments from composition properties."""
    boost = 0.0
    reasons: List[str] = []

    # High-octane naphtha → gasoline, not reformer
    if comp.family == FAMILY_NAPHTHA:
        if comp.ron >= 88.0 and sink in ("GASOLINE", "BLENDER"):
            boost += 0.20
            reasons.append(f"high RON={comp.ron:.0f} → pool")
        if comp.ron >= 88.0 and sink == "REFORMER":
            boost -= 0.45
            reasons.append("cat/cracked naphtha should not default to reformer")
        if comp.ron < 70.0 and sink == "REFORMER":
            boost += 0.25
            reasons.append(f"low RON={comp.ron:.0f} SR heavy → reformer")
        if comp.olefins_vol > 0.15 and sink == "REFORMER":
            boost -= 0.35
            reasons.append("olefinic → not reformer")
        if comp.olefins_vol > 0.15 and sink == "HDT_NAPH":
            boost += 0.20
            reasons.append("olefinic → soft HDT")
        if comp.sulfur_wt > 0.05 and sink == "HDT_NAPH":
            boost += 0.15
            reasons.append(f"S={comp.sulfur_wt:.3f} → HDT")

    # Gasoil: CCR/metals gate FCC
    if comp.family == FAMILY_GASOIL and sink in ("FCC", "POOL_FCC"):
        if comp.ccr_wt > 3.0 or comp.metals_ni_v_ppm > 15.0:
            boost -= 0.40
            reasons.append("high CCR/metals — FCC feed risk")
        else:
            boost += 0.10
            reasons.append("VGO quality OK for FCC")

    # Resid: high CCR → coker over FO slightly
    if comp.family == FAMILY_RESID:
        if sink in ("COKER", "POOL_COKER") and comp.ccr_wt >= 5.0:
            boost += 0.15
            reasons.append(f"CCR={comp.ccr_wt:.1f} favors coker")
        if sink == "FCC" and (comp.ccr_wt > 4.0 or comp.metals_ni_v_ppm > 10):
            boost -= 0.50
            reasons.append("resid not preferred FCC feed")

    # LPG vs fuel gas for C3/C4 with high RVP
    if comp.family == FAMILY_LIGHT_ENDS:
        if comp.rvp_psi > 50 and sink == "LPG":
            boost += 0.15
            reasons.append("C3/C4 RVP → LPG")
        if comp.rvp_psi < 20 and sink == "FUEL_GAS":
            boost += 0.10
            reasons.append("dry gas → fuel")
        if "h2" in comp.name.lower() and sink == "H2_GRID":
            boost += 0.40
            reasons.append("hydrogen stream")

    # Solids
    if comp.family == FAMILY_SOLID:
        if "fcc" in comp.name.lower() and sink == "REGEN_HEAT":
            boost += 0.20
            reasons.append("FCC coke → regenerator")
        if "coker" in comp.name.lower() and sink == "COKE_SALES":
            boost += 0.20
            reasons.append("petcoke sales")

    # TBP mid boiling for distillate vs gasoil edge cases
    if 400 <= comp.tbp_50_f <= 620 and sink == "DIESEL":
        boost += 0.08
        reasons.append(f"TBP50={comp.tbp_50_f:.0f}F diesel range")
    if 650 <= comp.tbp_50_f <= 1000 and sink in ("FCC", "POOL_FCC"):
        boost += 0.08
        reasons.append(f"TBP50={comp.tbp_50_f:.0f}F gasoil range")

    reason = "; ".join(reasons) if reasons else "family default"
    return boost, reason


def guess_route(
    stream: str | StreamComposition,
    candidates: Optional[Sequence[str]] = None,
    top_k: int = 5,
    min_score: float = 0.15,
) -> List[RouteGuess]:
    """Rank sinks for a stream by composition/family heuristics."""
    if isinstance(stream, StreamComposition):
        comp = stream
    else:
        comp = get_stream(str(stream))

    cands = list(candidates) if candidates is not None else list(DEFAULT_CANDIDATES)
    fam_map = _family_affinity().get(comp.family, {"SELL": 0.5})

    ranked: List[RouteGuess] = []
    for sink in cands:
        base = float(fam_map.get(sink, 0.05))
        boost, reason = _property_boosts(comp, sink)
        score = max(0.0, min(1.0, base + boost))
        if score < min_score and sink != "SELL":
            continue
        ranked.append(RouteGuess(sink=sink, score=score, reason=reason))

    ranked.sort(key=lambda g: g.score, reverse=True)
    if not ranked:
        ranked = [RouteGuess(sink="SELL", score=0.5, reason="fallback sell")]
    return ranked[:top_k]


def best_route(
    stream: str | StreamComposition,
    candidates: Optional[Sequence[str]] = None,
) -> RouteGuess:
    return guess_route(stream, candidates=candidates, top_k=1)[0]


def complete_missing_edges(
    produced_streams: Sequence[str],
    existing_edges: Sequence[Mapping[str, Any]],
    *,
    stream_key: str = "stream",
    compositions: Optional[Mapping[str, StreamComposition]] = None,
) -> List[Dict[str, Any]]:
    """For each produced stream with no outbound edge, invent a default edge via guess_route.

    existing_edges items should have at least {stream, from?, to?}.
    Returns list of suggested edges: {stream, to, score, reason, auto: True}.
    """
    covered = {str(e.get(stream_key) or e.get("stream")) for e in existing_edges if e.get(stream_key) or e.get("stream")}
    suggestions: List[Dict[str, Any]] = []
    for s in produced_streams:
        if s in covered:
            continue
        comp = compositions[s] if compositions and s in compositions else get_stream(s)
        g = best_route(comp)
        suggestions.append(
            {
                "stream": s,
                "to": g.sink,
                "score": g.score,
                "reason": g.reason,
                "auto": True,
                "composition_family": comp.family,
            }
        )
    return suggestions


def connect_score(
    source_unit: str,
    target_unit: str,
    stream: Optional[str] = None,
    composition: Optional[StreamComposition] = None,
) -> Dict[str, Any]:
    """API-friendly connect verdict using property heuristics."""
    if composition is None and stream:
        composition = get_stream(stream)
    if composition is None:
        return {
            "allowed": True,
            "score": 0.5,
            "reason": "types unknown — allowing connection (stub)",
            "guesses": [],
        }

    guesses = guess_route(composition, top_k=8, min_score=0.05)
    target = target_unit.upper()
    # normalize aliases
    aliases = {
        "BLENDER_GASOLINE": "GASOLINE",
        "HYDROTREAT_NAPH": "HDT_NAPH",
        "TANK_GASOIL": "POOL_FCC",
        "TANK_RESID": "POOL_COKER",
    }
    target_n = aliases.get(target, target)

    match = next((g for g in guesses if g.sink.upper() == target_n), None)
    if match is None:
        # partial name match
        match = next((g for g in guesses if target_n in g.sink.upper() or g.sink.upper() in target_n), None)

    if match and match.score >= 0.35:
        return {
            "allowed": True,
            "score": match.score,
            "reason": match.reason,
            "guesses": [g.to_dict() for g in guesses],
            "best": guesses[0].to_dict() if guesses else None,
        }
    if match and match.score >= 0.15:
        return {
            "allowed": True,
            "score": match.score,
            "reason": f"weak match: {match.reason}",
            "guesses": [g.to_dict() for g in guesses],
            "best": guesses[0].to_dict() if guesses else None,
        }
    # Disallow chemically absurd links at low score
    best = guesses[0] if guesses else RouteGuess("SELL", 0.5, "fallback")
    absurd = (
        composition.family == FAMILY_SOLID and target_n in ("REFORMER", "GASOLINE", "DIESEL")
    ) or (
        composition.family == FAMILY_LIGHT_ENDS and target_n in ("FCC", "COKER", "REFORMER")
    ) or (
        composition.ron >= 88 and composition.family == FAMILY_NAPHTHA and target_n == "REFORMER"
    )
    if absurd:
        return {
            "allowed": False,
            "score": match.score if match else 0.05,
            "reason": f"chemically unlikely {composition.name} → {target_n}; prefer {best.sink}",
            "guesses": [g.to_dict() for g in guesses],
            "best": best.to_dict(),
        }
    return {
        "allowed": True,
        "score": match.score if match else 0.25,
        "reason": f"permitted with low confidence; auto-prefer {best.sink}",
        "guesses": [g.to_dict() for g in guesses],
        "best": best.to_dict(),
    }
