"""PIMS-style shadow price report + economic interpretation + scale linearity.

Maps LP duals (and ADMM λ when available) onto make-buy-sell language:

  - Marginal value of extra intermediate stream (kbd)
  - Value of extra CDU / tank capacity
  - Crude flexibility (value of extra supply of each crude)
  - Product demand relief (value of extra product outlet)

Also checks local linearity: Δobjective ≈ shadow_price × ΔRHS for small
RHS moves — the same property planners rely on in Aspen PIMS marginals.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pulp

from pims_admm_llm.models.blocks import MonolithicResult, solve_monolithic
from pims_admm_llm.models.data import CrudeAssay, RefineryData, load_crude_data


@dataclass
class MarginalValueRow:
    item: str
    category: str  # intermediate | capacity | crude | product | admm_link
    shadow_price: float  # USD per unit of RHS relaxation (margin $/kbd-day scale)
    unit: str
    binding: bool
    interpretation: str
    make_buy_sell: str
    source_dual: str
    raw_dual: float


@dataclass
class LinearityCheck:
    name: str
    delta_rhs: float
    predicted_dobj: float
    actual_dobj: float
    abs_error: float
    rel_error: float
    passed: bool
    notes: str = ""


@dataclass
class ShadowPriceReport:
    baseline_objective: float
    baseline_status: str
    crude_rates: Dict[str, float]
    product_rates: Dict[str, float]
    intermediate_prod: Dict[str, float]
    intermediate_use: Dict[str, float]
    table: List[MarginalValueRow]
    linearity: List[LinearityCheck]
    admm_shadow_prices: Dict[str, float] = field(default_factory=dict)
    admm_vs_mono: Dict[str, float] = field(default_factory=dict)
    raw_duals: Dict[str, float] = field(default_factory=dict)
    reduced_costs: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_objective": self.baseline_objective,
            "baseline_status": self.baseline_status,
            "crude_rates": self.crude_rates,
            "product_rates": self.product_rates,
            "intermediate_prod": self.intermediate_prod,
            "intermediate_use": self.intermediate_use,
            "table": [asdict(r) for r in self.table],
            "linearity": [asdict(c) for c in self.linearity],
            "admm_shadow_prices": self.admm_shadow_prices,
            "admm_vs_mono": self.admm_vs_mono,
            "raw_duals": self.raw_duals,
            "reduced_costs": self.reduced_costs,
        }


def _economic_sign_for_le_capacity(dual: float) -> float:
    """Maximize LP: dual of a ≤ capacity constraint is ≥0 when binding.

    Positive dual = $ margin gained per unit extra capacity.
    """
    return float(dual)


def _intermediate_value_from_duals(duals: Dict[str, float], name: str) -> Tuple[float, str, float]:
    """Pick the cleanest dual proxy for intermediate marginal value.

    Prefer yield equality dual (value of one extra unit of intermediate
    produced / available). Then inv_balance, blend_use, balance.
    """
    ykey = f"yield_{name}"
    ibkey = f"inv_balance_{name}"
    bkey = f"balance_{name}"
    ukey = f"blend_use_{name}"
    if ykey in duals and abs(float(duals[ykey])) > 1e-12:
        return float(duals[ykey]), ykey, float(duals[ykey])
    if ibkey in duals and abs(float(duals[ibkey])) > 1e-12:
        # inv_balance: start + prod - use - end = 0; dual π → value of +start ≈ -π
        raw = float(duals[ibkey])
        return -raw, ibkey, raw
    if ukey in duals and abs(float(duals[ukey])) > 1e-12:
        raw = float(duals[ukey])
        return -raw, ukey, raw
    if bkey in duals:
        raw = float(duals[bkey])
        return -raw, bkey, raw
    if ykey in duals:
        return float(duals[ykey]), ykey, float(duals[ykey])
    return 0.0, "none", 0.0


def _buy_sell_from_value(value: float, kind: str, item: str) -> str:
    """Translate marginal value into a short make-buy-sell cue."""
    eps = 1e-6
    if kind == "intermediate":
        if value > eps:
            return (
                f"BUY / MAKE more {item}: internal transfer value "
                f"${value:.2f}/bbl — pay up to this for spot intermediate or yield shift."
            )
        if value < -eps:
            return f"SELL / DESTROY {item}: negative marginal value ${value:.2f}/bbl."
        return f"NEUTRAL on {item}: unconstrained / zero marginal at current basis."
    if kind == "capacity":
        if value > eps:
            return (
                f"EXPAND {item}: worth ${value:.2f} margin per unit capacity "
                f"— justify debottleneck / rental if cost < this."
            )
        return f"IDLE capacity signal on {item}: dual ≈ 0 — no short-run value to expand."
    if kind == "crude":
        if value > eps:
            return (
                f"BUY more {item}: supply is binding; each extra bbl of availability "
                f"adds ~${value:.2f} margin (after current price already in obj)."
            )
        if value < -eps:
            return (
                f"AVOID / REDUCE {item}: negative flexibility value ${value:.2f} "
                f"— at margin this crude hurts relative to alternatives."
            )
        return f"FLEXIBLE / non-binding on {item}: zero reduced-cost signal at optimum."
    if kind == "product":
        if value > eps:
            return (
                f"FIND outlet for more {item}: demand cap binds; +1 kbd demand "
                f"worth ${value:.2f} margin."
            )
        return f"Demand slack on {item}: dual ≈ 0 — market not limiting."
    return ""


def extract_reduced_costs(result: MonolithicResult) -> Dict[str, float]:
    """Variable reduced costs (dj) from the solved CBC basis."""
    out: Dict[str, float] = {}
    for v in result.problem.variables():
        try:
            dj = v.dj
            if dj is None:
                continue
            out[v.name] = float(dj)
        except Exception:
            continue
    return out


def build_marginal_table(
    result: MonolithicResult,
    data: RefineryData,
    admm_lambda: Optional[Dict[str, float]] = None,
) -> List[MarginalValueRow]:
    duals = result.duals
    rows: List[MarginalValueRow] = []
    reduced = extract_reduced_costs(result)

    # --- Intermediates ---
    for i in data.intermediates:
        val, src, raw = _intermediate_value_from_duals(duals, i)
        binding = abs(raw) > 1e-8
        rows.append(
            MarginalValueRow(
                item=i,
                category="intermediate",
                shadow_price=val,
                unit="USD / bbl intermediate",
                binding=binding,
                interpretation=(
                    f"Marginal value of +1 bbl {i} available to the plan "
                    f"(PIMS-style stream MV). Source dual '{src}'."
                ),
                make_buy_sell=_buy_sell_from_value(val, "intermediate", i),
                source_dual=src,
                raw_dual=raw,
            )
        )

    # --- CDU capacity ---
    raw_cdu = float(duals.get("cdu_capacity", 0.0))
    cdu_val = _economic_sign_for_le_capacity(raw_cdu)
    rows.append(
        MarginalValueRow(
            item="CDU_capacity",
            category="capacity",
            shadow_price=cdu_val,
            unit="USD margin / kbd capacity",
            binding=abs(raw_cdu) > 1e-8,
            interpretation=(
                "Value of relaxing CDU charge limit by 1 kbd. "
                "Classic PIMS unit capacity dual."
            ),
            make_buy_sell=_buy_sell_from_value(cdu_val, "capacity", "CDU"),
            source_dual="cdu_capacity",
            raw_dual=raw_cdu,
        )
    )

    # --- Tank capacities (if present on model) ---
    for i in data.intermediates:
        tkey = f"tank_{i}"
        if tkey not in duals:
            continue
        raw_t = float(duals[tkey])
        t_val = _economic_sign_for_le_capacity(raw_t)
        rows.append(
            MarginalValueRow(
                item=f"tank_{i}",
                category="capacity",
                shadow_price=t_val,
                unit="USD margin / kbd tank space",
                binding=abs(raw_t) > 1e-8,
                interpretation=(
                    f"Value of +1 kbd intermediate storage for {i} "
                    f"(tank-farm debottleneck dual on ending inventory)."
                ),
                make_buy_sell=_buy_sell_from_value(t_val, "capacity", f"tank/{i}"),
                source_dual=tkey,
                raw_dual=raw_t,
            )
        )

    # --- Utility capacities ---
    for name, raw_u in duals.items():
        if not name.startswith("utility_cap_"):
            continue
        uname = name.replace("utility_cap_", "", 1)
        u_val = _economic_sign_for_le_capacity(float(raw_u))
        rows.append(
            MarginalValueRow(
                item=f"utility_{uname}",
                category="capacity",
                shadow_price=u_val,
                unit="USD margin / utility unit",
                binding=abs(float(raw_u)) > 1e-8,
                interpretation=f"Value of +1 unit of shared utility {uname}.",
                make_buy_sell=_buy_sell_from_value(u_val, "capacity", f"utility/{uname}"),
                source_dual=name,
                raw_dual=float(raw_u),
            )
        )

    # Shared tank farm (optional legacy key)
    if "tank_farm_total" in duals:
        raw_tf = float(duals["tank_farm_total"])
        tf_val = _economic_sign_for_le_capacity(raw_tf)
        rows.append(
            MarginalValueRow(
                item="tank_farm_total",
                category="capacity",
                shadow_price=tf_val,
                unit="USD margin / kbd total tank",
                binding=abs(raw_tf) > 1e-8,
                interpretation="Value of extra shared tank-farm ullage.",
                make_buy_sell=_buy_sell_from_value(tf_val, "capacity", "tank farm"),
                source_dual="tank_farm_total",
                raw_dual=raw_tf,
            )
        )

    # --- Crude flexibility via supply duals or reduced costs ---
    for c in data.crudes:
        # Prefer explicit supply constraint dual if present
        skey = f"crude_supply_{c.name}"
        if skey in duals:
            raw = float(duals[skey])
            # crude_rate <= max_supply → dual ≥ 0 under max = value of extra supply
            val = _economic_sign_for_le_capacity(raw)
            src = skey
        else:
            # Reduced cost on upper-bounded crude: for maximize, dj of nonbasic at ub
            # can be negative; value of relaxing ub ≈ max(-dj, 0) depending on convention.
            # PuLP/CBC: for max problems, reduced cost of variable at upper bound is ≤0
            # when optimal; -dj is the value of increasing the upper bound by 1.
            vname = f"crude_{c.name}"
            dj = float(reduced.get(vname, 0.0))
            rate = float(result.crude_rates.get(c.name, 0.0))
            at_ub = abs(rate - c.max_supply_kbd) < 1e-5
            at_lb = rate < 1e-5
            if at_ub:
                val = -dj  # value of extra supply
            elif at_lb:
                val = dj  # often ≤0 meaning don't enter
            else:
                val = 0.0  # basic / free within bounds
            raw = dj
            src = f"dj:{vname}"
        rows.append(
            MarginalValueRow(
                item=c.name,
                category="crude",
                shadow_price=val,
                unit="USD margin / bbl extra supply",
                binding=abs(raw) > 1e-8
                or abs(result.crude_rates.get(c.name, 0.0) - c.max_supply_kbd) < 1e-5,
                interpretation=(
                    f"Crude flexibility for {c.name} (price already ${c.price_usd_per_bbl:.2f}/bbl "
                    f"in objective). Positive = buy more if available; zero at interior."
                ),
                make_buy_sell=_buy_sell_from_value(val, "crude", c.name),
                source_dual=src,
                raw_dual=raw,
            )
        )

    # --- Product demand caps ---
    for pname, spec in data.products.items():
        dkey = f"product_demand_{pname}"
        if dkey in duals:
            raw = float(duals[dkey])
            val = _economic_sign_for_le_capacity(raw)
            src = dkey
        else:
            vname = f"product_{pname}"
            dj = float(reduced.get(vname, 0.0))
            rate = float(result.product_rates.get(pname, 0.0))
            at_ub = abs(rate - spec.max_demand_kbd) < 1e-5
            if at_ub:
                val = -dj
            else:
                val = 0.0
            raw = dj
            src = f"dj:{vname}"
        rows.append(
            MarginalValueRow(
                item=pname,
                category="product",
                shadow_price=val,
                unit="USD margin / kbd extra demand",
                binding=abs(raw) > 1e-8
                or abs(result.product_rates.get(pname, 0.0) - spec.max_demand_kbd) < 1e-5,
                interpretation=(
                    f"Value of +1 kbd product placement for {pname} "
                    f"(price ${spec.price_usd_per_bbl:.2f}/bbl already in obj)."
                ),
                make_buy_sell=_buy_sell_from_value(val, "product", pname),
                source_dual=src,
                raw_dual=raw,
            )
        )

    # --- ADMM linking duals (if provided) ---
    if admm_lambda:
        for i, lam in admm_lambda.items():
            mono_val, _, _ = _intermediate_value_from_duals(duals, i)
            rows.append(
                MarginalValueRow(
                    item=f"admm_λ_{i}",
                    category="admm_link",
                    shadow_price=float(lam),
                    unit="USD / bbl (ADMM dual)",
                    binding=abs(float(lam)) > 1e-8,
                    interpretation=(
                        f"ADMM dual λ on intermediate consensus for {i}. "
                        f"At convergence should track monolithic stream MV "
                        f"(~{mono_val:.2f})."
                    ),
                    make_buy_sell=(
                        f"Transfer price signal for {i}: use as internal make-buy "
                        f"clearing price between CDU and Blender blocks."
                    ),
                    source_dual="admm_lambda",
                    raw_dual=float(lam),
                )
            )

    return rows


def _clone_data(data: RefineryData) -> RefineryData:
    return copy.deepcopy(data)


def run_linearity_checks(
    data: RefineryData,
    baseline: MonolithicResult,
    deltas: Optional[Sequence[Tuple[str, float]]] = None,
    rel_tol: float = 0.15,
    abs_tol: float = 5.0,
) -> List[LinearityCheck]:
    """Re-solve with small RHS moves; compare Δobj to dual prediction.

    Defaults:
      - CDU capacity +1 and +0.5 kbd
      - Each crude max_supply +1 (if binding)
      - Free gift of +1 intermediate via synthetic supply (balance RHS)
    """
    duals = baseline.duals
    checks: List[LinearityCheck] = []

    if deltas is None:
        deltas = [
            ("cdu_capacity", 1.0),
            ("cdu_capacity", 0.5),
            ("cdu_capacity", 2.0),
        ]
        for c in data.crudes:
            if baseline.crude_rates.get(c.name, 0.0) >= c.max_supply_kbd - 1e-5:
                deltas.append((f"crude_supply:{c.name}", 1.0))
        # product demand relief when binding
        for pname, spec in data.products.items():
            if baseline.product_rates.get(pname, 0.0) >= spec.max_demand_kbd - 1e-5:
                deltas.append((f"product_demand:{pname}", 1.0))
        # intermediate free gift for highest-value stream
        best_i = None
        best_v = -1e18
        for i in data.intermediates:
            v, _, _ = _intermediate_value_from_duals(duals, i)
            if v > best_v:
                best_v = v
                best_i = i
        if best_i is not None and best_v > 1e-6:
            deltas.append((f"gift_intermediate:{best_i}", 1.0))
            deltas.append((f"gift_intermediate:{best_i}", 0.5))

    base_obj = baseline.objective

    for name, delta in deltas:
        d = _clone_data(data)
        predicted = 0.0
        notes = ""

        if name == "cdu_capacity":
            predicted = float(duals.get("cdu_capacity", 0.0)) * delta
            d.cdu_capacity_kbd = data.cdu_capacity_kbd + delta
            notes = "relax CDU capacity"
        elif name.startswith("crude_supply:"):
            cname = name.split(":", 1)[1]
            skey = f"crude_supply_{cname}"
            if skey in duals:
                predicted = float(duals[skey]) * delta
            else:
                reduced = extract_reduced_costs(baseline)
                dj = float(reduced.get(f"crude_{cname}", 0.0))
                predicted = (-dj) * delta
            for c in d.crudes:
                if c.name == cname:
                    c.max_supply_kbd = c.max_supply_kbd + delta
            notes = f"relax crude supply cap for {cname}"
        elif name.startswith("product_demand:"):
            pname = name.split(":", 1)[1]
            dkey = f"product_demand_{pname}"
            predicted = float(duals.get(dkey, 0.0)) * delta
            if pname in d.products:
                d.products[pname].max_demand_kbd = d.products[pname].max_demand_kbd + delta
            notes = f"relax product demand for {pname}"
        elif name.startswith("gift_intermediate:"):
            iname = name.split(":", 1)[1]
            val, src, raw = _intermediate_value_from_duals(duals, iname)
            predicted = val * delta
            # implement gift by solving augmented LP
            res = _solve_with_intermediate_gift(d, iname, delta)
            actual = res.objective - base_obj
            abs_err = abs(actual - predicted)
            rel = abs_err / max(abs(predicted), 1e-6)
            # larger gifts often change basis (same as PIMS MV neighborhood limits)
            tol_scale = 1.0 + 0.5 * max(0.0, abs(delta) - 0.5)
            passed = abs_err <= abs_tol * tol_scale or rel <= rel_tol * tol_scale
            notes = f"gift via inventory start bump; dual source {src}"
            if not passed and abs(delta) >= 1.0 - 1e-9:
                notes += " (basis likely changed at this step size — expected PIMS-style)"
            checks.append(
                LinearityCheck(
                    name=name,
                    delta_rhs=delta,
                    predicted_dobj=predicted,
                    actual_dobj=actual,
                    abs_error=abs_err,
                    rel_error=rel,
                    passed=passed,
                    notes=notes,
                )
            )
            continue
        elif name.startswith("tank:"):
            # not default; reserved
            notes = name
            predicted = float(duals.get(name.replace(":", "_"), 0.0)) * delta
        else:
            notes = f"unknown perturbation {name}"
            checks.append(
                LinearityCheck(
                    name=name,
                    delta_rhs=delta,
                    predicted_dobj=0.0,
                    actual_dobj=0.0,
                    abs_error=0.0,
                    rel_error=0.0,
                    passed=False,
                    notes=notes,
                )
            )
            continue

        res = solve_monolithic(d, msg=False)
        actual = res.objective - base_obj
        abs_err = abs(actual - predicted)
        rel = abs_err / max(abs(predicted), 1e-6)
        # basis may change — allow looser pass for larger deltas
        tol_scale = 1.0 + 0.25 * max(0.0, abs(delta) - 1.0)
        passed = abs_err <= abs_tol * tol_scale or rel <= rel_tol * tol_scale
        checks.append(
            LinearityCheck(
                name=name,
                delta_rhs=delta,
                predicted_dobj=predicted,
                actual_dobj=actual,
                abs_error=abs_err,
                rel_error=rel,
                passed=passed,
                notes=notes,
            )
        )

    return checks


def _solve_with_intermediate_gift(
    data: RefineryData, intermediate: str, gift_kbd: float
) -> MonolithicResult:
    """Monolithic solve where `gift_kbd` free barrels of intermediate are available.

    Preferred: bump inventory starting stock (start + gift + prod = use + end).
    Fallback: free gift variable into the material balance.
    """
    d = copy.deepcopy(data)
    inv = (d.inventory or {}).get(intermediate)
    if inv is not None:
        inv.start_kbd = float(inv.start_kbd) + float(gift_kbd)
        # keep capacity feasible for ending inventory path
        if inv.capacity_kbd < inv.start_kbd:
            inv.capacity_kbd = inv.start_kbd
        return solve_monolithic(d, msg=False)

    # Fallback without inventory objects: rebuild LP with gift var
    import time

    from pims_admm_llm.models.blocks import build_monolithic_lp, extract_monolithic_solution

    prob = build_monolithic_lp(d)
    gift = pulp.LpVariable(f"gift_{intermediate}", lowBound=gift_kbd, upBound=gift_kbd)
    use_var = next(v for v in prob.variables() if v.name == f"use_{intermediate}")
    prod_var = next(v for v in prob.variables() if v.name == f"prod_{intermediate}")
    bal = f"balance_{intermediate}"
    if bal in prob.constraints:
        del prob.constraints[bal]
    prob += prod_var + gift >= use_var, bal
    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
    t1 = time.perf_counter()
    return extract_monolithic_solution(prob, d, t1 - t0)


def build_shadow_price_report(
    data: Optional[RefineryData] = None,
    admm_result: Any = None,
    run_linearity: bool = True,
) -> ShadowPriceReport:
    """Full PIMS-style shadow price package for the toy refinery."""
    if data is None:
        data = load_crude_data()

    mono = solve_monolithic(data, msg=False)
    admm_lam: Dict[str, float] = {}
    if admm_result is not None:
        admm_lam = dict(getattr(admm_result, "shadow_prices", {}) or {})

    table = build_marginal_table(mono, data, admm_lambda=admm_lam or None)
    linearity = run_linearity_checks(data, mono) if run_linearity else []
    reduced = extract_reduced_costs(mono)

    admm_vs: Dict[str, float] = {}
    for i in data.intermediates:
        mono_val, _, _ = _intermediate_value_from_duals(mono.duals, i)
        if i in admm_lam:
            admm_vs[i] = float(admm_lam[i]) - mono_val
        elif f"λ_{i}" in admm_lam:
            admm_vs[i] = float(admm_lam[f"λ_{i}"]) - mono_val

    return ShadowPriceReport(
        baseline_objective=mono.objective,
        baseline_status=mono.status,
        crude_rates=mono.crude_rates,
        product_rates=mono.product_rates,
        intermediate_prod=mono.intermediate_prod,
        intermediate_use=mono.intermediate_use,
        table=table,
        linearity=linearity,
        admm_shadow_prices=admm_lam,
        admm_vs_mono=admm_vs,
        raw_duals=dict(mono.duals),
        reduced_costs=reduced,
    )


def format_report_text(report: ShadowPriceReport) -> str:
    """Human-readable PIMS-style marginal value report."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("PIMS-STYLE SHADOW PRICE / MARGINAL VALUE REPORT")
    lines.append("pims-admm-llm — Worker 7 economic interpretation")
    lines.append("=" * 78)
    lines.append(
        f"Baseline status: {report.baseline_status}   "
        f"Objective (margin): {report.baseline_objective:,.4f} USD-scale"
    )
    lines.append("")
    lines.append("--- Primal snapshot (kbd) ---")
    lines.append(f"Crudes:        { {k: round(v, 4) for k, v in report.crude_rates.items()} }")
    lines.append(f"Products:      { {k: round(v, 4) for k, v in report.product_rates.items()} }")
    lines.append(
        f"Interm. prod:  { {k: round(v, 4) for k, v in report.intermediate_prod.items()} }"
    )
    lines.append(
        f"Interm. use:   { {k: round(v, 4) for k, v in report.intermediate_use.items()} }"
    )
    lines.append("")
    lines.append("--- Marginal value table ---")
    lines.append(
        f"{'ITEM':<22} {'CAT':<12} {'MV':>12} {'UNIT':<28} {'BIND':>5}  MAKE-BUY-SELL"
    )
    lines.append("-" * 78)
    for r in report.table:
        lines.append(
            f"{r.item:<22} {r.category:<12} {r.shadow_price:12.4f} "
            f"{r.unit:<28} {'Y' if r.binding else 'n':>5}  "
            f"{r.make_buy_sell[:60]}"
        )
    lines.append("")
    lines.append("--- Economic interpretation (detail) ---")
    for r in report.table:
        lines.append(f"* {r.item} [{r.category}]  MV={r.shadow_price:.4f} {r.unit}")
        lines.append(f"    dual source: {r.source_dual}  raw={r.raw_dual:.6f}")
        lines.append(f"    {r.interpretation}")
        lines.append(f"    → {r.make_buy_sell}")
    lines.append("")
    lines.append("--- Scale linearity check (Δobj ≈ MV × ΔRHS) ---")
    if not report.linearity:
        lines.append("(skipped)")
    else:
        lines.append(
            f"{'NAME':<32} {'ΔRHS':>8} {'PRED':>10} {'ACTUAL':>10} "
            f"{'|ERR|':>10} {'REL':>8} PASS"
        )
        lines.append("-" * 78)
        for c in report.linearity:
            lines.append(
                f"{c.name:<32} {c.delta_rhs:8.3f} {c.predicted_dobj:10.4f} "
                f"{c.actual_dobj:10.4f} {c.abs_error:10.4f} {c.rel_error:8.3f} "
                f"{'PASS' if c.passed else 'FAIL'}"
            )
            if c.notes:
                lines.append(f"    notes: {c.notes}")
        n_pass = sum(1 for c in report.linearity if c.passed)
        lines.append(f"Linearity summary: {n_pass}/{len(report.linearity)} checks passed")
    lines.append("")
    if report.admm_shadow_prices:
        lines.append("--- ADMM λ vs monolithic stream MV ---")
        for k, v in report.admm_shadow_prices.items():
            gap = report.admm_vs_mono.get(k, float("nan"))
            lines.append(f"  λ[{k}] = {v:.4f}   gap_to_mono = {gap:.4f}")
    else:
        lines.append(
            "--- ADMM λ --- (not supplied; pass admm_result to compare at convergence)"
        )
    lines.append("")
    lines.append("--- Raw duals (solver) ---")
    for k, v in sorted(report.raw_duals.items()):
        lines.append(f"  {k:30s} {v: .6f}")
    lines.append("")
    lines.append("--- Reduced costs (selected) ---")
    for k, v in sorted(report.reduced_costs.items()):
        if k.startswith("crude_") or k.startswith("product_"):
            lines.append(f"  {k:30s} {v: .6f}")
    lines.append("=" * 78)
    return "\n".join(lines)
