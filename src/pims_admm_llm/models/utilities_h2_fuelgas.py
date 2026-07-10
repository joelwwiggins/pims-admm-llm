"""W3: H2 purchases + fuel-gas BTU/bbl light-ends sales (planning).

Planning conventions
--------------------
* H2 is a **purchased** utility (kscf). FCC consumes a stoichiometric rate
  (kscf/bbl feed). Buy variable ≥ requirement; surplus buy has no value so
  optimum sets buy = req.
* Light ends (CDU offgas, FCC dry gas, FCC LPG) exit to a **fuel-gas pool**
  sold on a **BTU basis**: volume × (MMBTU/bbl) × ($/MMBTU).
* Factors are planning-grade liquid-equivalent BTU contents, not custody
  metering calorific values.

Units
-----
* rates: kbd (kbbl/day) for liquid-equivalent light ends / FCC feed
* H2: kscf/day
* fuel gas energy: MMBTU/day
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import pulp

# ---- defaults (planning) ----

# MMBTU per bbl liquid-equivalent of light-end stream
BTU_MMBTU_PER_BBL: Dict[str, float] = {
    "fcc_dry_gas": 3.8,
    "fcc_lpg": 3.5,
    "cdu_offgas": 3.2,
}

# kscf H2 purchased per bbl FCC feed (planning, not full hydrotreating)
H2_KSCF_PER_BBL_FCC: float = 0.15

DEFAULT_UTIL_PRICES: Dict[str, float] = {
    "h2_usd_per_kscf": 8.0,
    "fuel_gas_usd_per_mmbtu": 3.5,
}

# Canonical light-end → fuel-gas pool map for this case plant
LIGHT_END_STREAMS: Tuple[str, ...] = ("fcc_dry_gas", "fcc_lpg", "cdu_offgas")


@dataclass
class H2FuelGasResult:
    """Post-solve utilities snapshot for VERDICT / tests."""

    h2_req_kscf: float
    h2_buy_kscf: float
    h2_cost_usd: float
    h2_kscf_per_bbl_fcc: float
    fuel_gas_mmbtu: float
    fuel_gas_revenue_usd: float
    fuel_gas_usd_per_mmbtu: float
    light_ends_bbl: Dict[str, float] = field(default_factory=dict)
    light_ends_mmbtu: Dict[str, float] = field(default_factory=dict)
    btu_mmbtu_per_bbl: Dict[str, float] = field(default_factory=dict)
    ok: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "h2_req_kscf": self.h2_req_kscf,
            "h2_buy_kscf": self.h2_buy_kscf,
            "h2_cost_usd": self.h2_cost_usd,
            "h2_kscf_per_bbl_fcc": self.h2_kscf_per_bbl_fcc,
            "fuel_gas_mmbtu": self.fuel_gas_mmbtu,
            "fuel_gas_revenue_usd": self.fuel_gas_revenue_usd,
            "fuel_gas_usd_per_mmbtu": self.fuel_gas_usd_per_mmbtu,
            "light_ends_bbl": dict(self.light_ends_bbl),
            "light_ends_mmbtu": dict(self.light_ends_mmbtu),
            "btu_mmbtu_per_bbl": dict(self.btu_mmbtu_per_bbl),
            "ok": self.ok,
            "notes": self.notes,
        }


def h2_requirement_kscf(
    fcc_feed_kbd: float,
    *,
    rate_kscf_per_bbl: float = H2_KSCF_PER_BBL_FCC,
) -> float:
    """kscf/day H2 required for given FCC feed (kbd)."""
    return float(rate_kscf_per_bbl) * float(fcc_feed_kbd)


def light_ends_mmbtu_by_stream(
    rates_kbd: Mapping[str, float],
    *,
    btu_table: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Per-stream MMBTU/day = rate_kbd × (MMBTU/bbl)."""
    table = dict(BTU_MMBTU_PER_BBL if btu_table is None else btu_table)
    out: Dict[str, float] = {}
    for name, rate in rates_kbd.items():
        factor = float(table.get(name, 0.0))
        out[name] = float(rate) * factor
    return out


def total_fuel_gas_mmbtu(
    rates_kbd: Mapping[str, float],
    *,
    btu_table: Optional[Mapping[str, float]] = None,
) -> float:
    return float(sum(light_ends_mmbtu_by_stream(rates_kbd, btu_table=btu_table).values()))


def evaluate_h2_fuel_gas(
    *,
    fcc_feed_kbd: float,
    light_ends_kbd: Mapping[str, float],
    h2_buy_kscf: Optional[float] = None,
    prices: Optional[Mapping[str, float]] = None,
    h2_rate: float = H2_KSCF_PER_BBL_FCC,
    btu_table: Optional[Mapping[str, float]] = None,
) -> H2FuelGasResult:
    """Closed-form utilities evaluation (no LP)."""
    px = dict(DEFAULT_UTIL_PRICES)
    if prices:
        px.update(prices)
    table = dict(BTU_MMBTU_PER_BBL if btu_table is None else btu_table)
    req = h2_requirement_kscf(fcc_feed_kbd, rate_kscf_per_bbl=h2_rate)
    buy = float(req if h2_buy_kscf is None else h2_buy_kscf)
    by_mmbtu = light_ends_mmbtu_by_stream(light_ends_kbd, btu_table=table)
    total_mmbtu = float(sum(by_mmbtu.values()))
    h2_cost = buy * float(px["h2_usd_per_kscf"])
    fg_rev = total_mmbtu * float(px["fuel_gas_usd_per_mmbtu"])
    ok = buy + 1e-9 >= req and all(v >= -1e-9 for v in light_ends_kbd.values())
    return H2FuelGasResult(
        h2_req_kscf=req,
        h2_buy_kscf=buy,
        h2_cost_usd=h2_cost,
        h2_kscf_per_bbl_fcc=float(h2_rate),
        fuel_gas_mmbtu=total_mmbtu,
        fuel_gas_revenue_usd=fg_rev,
        fuel_gas_usd_per_mmbtu=float(px["fuel_gas_usd_per_mmbtu"]),
        light_ends_bbl={k: float(v) for k, v in light_ends_kbd.items()},
        light_ends_mmbtu=by_mmbtu,
        btu_mmbtu_per_bbl={k: float(table.get(k, 0.0)) for k in light_ends_kbd},
        ok=ok,
        notes="planning H2 buy + fuel-gas BTU sales; liquid-eq MMBTU/bbl",
    )


def add_h2_purchase_to_lp(
    prob: pulp.LpProblem,
    fcc_feed: Union[pulp.LpVariable, pulp.LpAffineExpression, float],
    *,
    rate_kscf_per_bbl: float = H2_KSCF_PER_BBL_FCC,
    price_usd_per_kscf: float = DEFAULT_UTIL_PRICES["h2_usd_per_kscf"],
    max_buy: Optional[float] = None,
    name: str = "buy_h2_kscf",
) -> Tuple[pulp.LpVariable, pulp.LpAffineExpression, pulp.LpAffineExpression]:
    """Add buy_h2 ≥ rate * fcc_feed. Returns (buy_var, requirement_expr, cost_expr)."""
    ub = max_buy if max_buy is not None else None
    buy = pulp.LpVariable(name, lowBound=0, upBound=ub)
    # requirement as affine expression
    req = float(rate_kscf_per_bbl) * fcc_feed
    prob += buy >= req, f"{name}_covers_req"
    cost = float(price_usd_per_kscf) * buy
    return buy, req, cost  # type: ignore[return-value]


def add_fuel_gas_sales_to_lp(
    light_end_flows: Mapping[str, Any],
    *,
    btu_table: Optional[Mapping[str, float]] = None,
    price_usd_per_mmbtu: float = DEFAULT_UTIL_PRICES["fuel_gas_usd_per_mmbtu"],
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Build MMBTU and revenue expressions from light-end flow expressions/vars.

    Returns (mmbtu_expr, revenue_expr, detail_factors).
    """
    table = dict(BTU_MMBTU_PER_BBL if btu_table is None else btu_table)
    pieces = []
    detail: Dict[str, Any] = {"btu_mmbtu_per_bbl": {}, "streams": []}
    for name, flow in light_end_flows.items():
        factor = float(table.get(name, 0.0))
        detail["btu_mmbtu_per_bbl"][name] = factor
        detail["streams"].append(name)
        pieces.append(factor * flow)
    if not pieces:
        mmbtu: Any = 0.0
    else:
        mmbtu = pulp.lpSum(pieces) if len(pieces) > 1 else pieces[0]
    revenue = float(price_usd_per_mmbtu) * mmbtu
    return mmbtu, revenue, detail


def snapshot_from_solved(
    *,
    fcc_feed_kbd: float,
    light_ends_kbd: Mapping[str, float],
    h2_buy_kscf: float,
    prices: Mapping[str, float],
    h2_rate: float = H2_KSCF_PER_BBL_FCC,
    btu_table: Optional[Mapping[str, float]] = None,
) -> H2FuelGasResult:
    """Post-solve snapshot; enforces buy ≥ req within tolerance for ok flag."""
    res = evaluate_h2_fuel_gas(
        fcc_feed_kbd=fcc_feed_kbd,
        light_ends_kbd=light_ends_kbd,
        h2_buy_kscf=h2_buy_kscf,
        prices=prices,
        h2_rate=h2_rate,
        btu_table=btu_table,
    )
    if h2_buy_kscf + 1e-6 < res.h2_req_kscf:
        res.ok = False
        res.notes = "h2 buy short of requirement"
    return res
