"""Planning-grade stream composition vectors.

Every intermediate / product stream carries a property bag so:
  - base-delta unit models can emit composition with yields
  - auto-routing can score destinations when no edge is drawn
  - quality / blender layers consume a common schema

Keep this independent of full-plant topology so CDU→FCC can be proven first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Iterable, List, Mapping, Optional


# Boiling-range / family tags used by auto-route heuristics
FAMILY_LIGHT_ENDS = "light_ends"  # C1–C4, H2, fuel gas
FAMILY_NAPHTHA = "naphtha"
FAMILY_DISTILLATE = "distillate"  # kero / diesel range
FAMILY_GASOIL = "gasoil"  # VGO / LCO
FAMILY_RESID = "resid"  # slurry, VR, FO
FAMILY_SOLID = "solid"  # coke
FAMILY_PRODUCT = "finished_product"


@dataclass
class StreamComposition:
    """Property vector attached to a stream name at a planning node."""

    name: str
    family: str = FAMILY_GASOIL
    # bulk
    api: float = 30.0
    specific_gravity: float = 0.876  # ~API 30
    sulfur_wt: float = 0.5
    ccr_wt: float = 0.5
    nitrogen_ppm: float = 500.0
    # gasoline-relevant
    ron: float = 0.0
    rvp_psi: float = 0.0
    olefins_vol: float = 0.0
    # PNA (vol frac, sum ≈ 1 for liquid hydrocarbons when known)
    paraffins_vol: float = 0.33
    naphthenes_vol: float = 0.33
    aromatics_vol: float = 0.34
    # distillation markers (°F) — planning, not full TBP
    tbp_10_f: float = 200.0
    tbp_50_f: float = 500.0
    tbp_90_f: float = 800.0
    # metals / FCC poison proxies
    metals_ni_v_ppm: float = 0.0
    basic_nitrogen_ppm: float = 0.0
    # misc
    viscosity_cst_210f: float = 5.0
    hydrogen_wt: float = 0.12
    note: str = ""
    extras: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_mapping(cls, name: str, data: Mapping[str, Any]) -> "StreamComposition":
        kw: Dict[str, Any] = {"name": name}
        known = {f.name for f in fields(cls)}
        extras: Dict[str, float] = {}
        for k, v in data.items():
            if k in known and k not in ("name", "extras"):
                kw[k] = v
            elif k != "name":
                try:
                    extras[str(k)] = float(v)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    if k == "note":
                        kw["note"] = str(v)
                    elif k == "family":
                        kw["family"] = str(v)
        if extras:
            kw["extras"] = extras
        return cls(**kw)  # type: ignore[arg-type]

    @property
    def n_plus_a(self) -> float:
        return self.naphthenes_vol + self.aromatics_vol

    def blend(self, other: "StreamComposition", w_self: float, w_other: float) -> "StreamComposition":
        """Volume-weighted blend (planning approximation)."""
        t = max(w_self + w_other, 1e-12)
        a, b = w_self / t, w_other / t

        def mix(x: float, y: float) -> float:
            return a * x + b * y

        return StreamComposition(
            name=f"blend({self.name}+{other.name})",
            family=self.family if self.family == other.family else FAMILY_PRODUCT,
            api=mix(self.api, other.api),
            specific_gravity=mix(self.specific_gravity, other.specific_gravity),
            sulfur_wt=mix(self.sulfur_wt, other.sulfur_wt),
            ccr_wt=mix(self.ccr_wt, other.ccr_wt),
            nitrogen_ppm=mix(self.nitrogen_ppm, other.nitrogen_ppm),
            ron=mix(self.ron, other.ron),
            rvp_psi=mix(self.rvp_psi, other.rvp_psi),
            olefins_vol=mix(self.olefins_vol, other.olefins_vol),
            paraffins_vol=mix(self.paraffins_vol, other.paraffins_vol),
            naphthenes_vol=mix(self.naphthenes_vol, other.naphthenes_vol),
            aromatics_vol=mix(self.aromatics_vol, other.aromatics_vol),
            tbp_10_f=mix(self.tbp_10_f, other.tbp_10_f),
            tbp_50_f=mix(self.tbp_50_f, other.tbp_50_f),
            tbp_90_f=mix(self.tbp_90_f, other.tbp_90_f),
            metals_ni_v_ppm=mix(self.metals_ni_v_ppm, other.metals_ni_v_ppm),
            basic_nitrogen_ppm=mix(self.basic_nitrogen_ppm, other.basic_nitrogen_ppm),
            viscosity_cst_210f=mix(self.viscosity_cst_210f, other.viscosity_cst_210f),
            hydrogen_wt=mix(self.hydrogen_wt, other.hydrogen_wt),
            note="volume_blend",
        )


def api_to_sg(api: float) -> float:
    return 141.5 / (api + 131.5)


def sg_to_api(sg: float) -> float:
    return 141.5 / max(sg, 0.5) - 131.5


# ---------------------------------------------------------------------------
# Canonical library for CDU / FCC products (planning defaults)
# ---------------------------------------------------------------------------

STREAM_LIBRARY: Dict[str, StreamComposition] = {
    "crude": StreamComposition(
        name="crude",
        family=FAMILY_RESID,
        api=30.0,
        specific_gravity=api_to_sg(30.0),
        sulfur_wt=1.0,
        ccr_wt=2.0,
        nitrogen_ppm=1000.0,
        tbp_10_f=150.0,
        tbp_50_f=550.0,
        tbp_90_f=1050.0,
        note="parent crude (assay-driven overrides)",
    ),
    "cdu_offgas": StreamComposition(
        name="cdu_offgas",
        family=FAMILY_LIGHT_ENDS,
        api=120.0,
        specific_gravity=0.45,
        sulfur_wt=0.0,
        ccr_wt=0.0,
        ron=0.0,
        tbp_10_f=-50.0,
        tbp_50_f=0.0,
        tbp_90_f=50.0,
        hydrogen_wt=0.25,
        note="CDU overhead fuel gas",
    ),
    "cdu_naphtha_light": StreamComposition(
        name="cdu_naphtha_light",
        family=FAMILY_NAPHTHA,
        api=68.0,
        specific_gravity=api_to_sg(68.0),
        sulfur_wt=0.005,
        ccr_wt=0.0,
        ron=72.0,
        rvp_psi=10.0,
        paraffins_vol=0.55,
        naphthenes_vol=0.30,
        aromatics_vol=0.15,
        tbp_10_f=100.0,
        tbp_50_f=200.0,
        tbp_90_f=300.0,
        note="SR light naphtha → gasoline / isomerate",
    ),
    "cdu_naphtha_heavy": StreamComposition(
        name="cdu_naphtha_heavy",
        family=FAMILY_NAPHTHA,
        api=55.0,
        specific_gravity=api_to_sg(55.0),
        sulfur_wt=0.01,
        ccr_wt=0.0,
        ron=58.0,
        rvp_psi=4.0,
        paraffins_vol=0.45,
        naphthenes_vol=0.35,
        aromatics_vol=0.20,
        tbp_10_f=220.0,
        tbp_50_f=300.0,
        tbp_90_f=380.0,
        note="SR heavy naphtha → reformer preferred",
    ),
    "cdu_distillate": StreamComposition(
        name="cdu_distillate",
        family=FAMILY_DISTILLATE,
        api=38.0,
        specific_gravity=api_to_sg(38.0),
        sulfur_wt=0.12,
        ccr_wt=0.05,
        ron=0.0,
        paraffins_vol=0.40,
        naphthenes_vol=0.30,
        aromatics_vol=0.30,
        tbp_10_f=380.0,
        tbp_50_f=500.0,
        tbp_90_f=620.0,
        note="SR kero/diesel",
    ),
    "cdu_gasoil": StreamComposition(
        name="cdu_gasoil",
        family=FAMILY_GASOIL,
        api=22.0,
        specific_gravity=api_to_sg(22.0),
        sulfur_wt=0.45,
        ccr_wt=0.4,
        nitrogen_ppm=800.0,
        metals_ni_v_ppm=1.0,
        basic_nitrogen_ppm=200.0,
        paraffins_vol=0.30,
        naphthenes_vol=0.30,
        aromatics_vol=0.40,
        tbp_10_f=650.0,
        tbp_50_f=850.0,
        tbp_90_f=1050.0,
        note="VGO / atmospheric gasoil → FCC preferred",
    ),
    "cdu_resid": StreamComposition(
        name="cdu_resid",
        family=FAMILY_RESID,
        api=12.0,
        specific_gravity=api_to_sg(12.0),
        sulfur_wt=2.5,
        ccr_wt=8.0,
        metals_ni_v_ppm=40.0,
        viscosity_cst_210f=80.0,
        tbp_10_f=950.0,
        tbp_50_f=1100.0,
        tbp_90_f=1300.0,
        note="atmospheric resid → coker / FO",
    ),
    "fcc_dry_gas": StreamComposition(
        name="fcc_dry_gas",
        family=FAMILY_LIGHT_ENDS,
        api=140.0,
        specific_gravity=0.35,
        sulfur_wt=0.0,
        hydrogen_wt=0.22,
        tbp_10_f=-150.0,
        tbp_50_f=-50.0,
        tbp_90_f=50.0,
        note="H2–C2 fuel gas",
    ),
    "fcc_lpg": StreamComposition(
        name="fcc_lpg",
        family=FAMILY_LIGHT_ENDS,
        api=110.0,
        specific_gravity=0.55,
        sulfur_wt=0.0,
        olefins_vol=0.45,
        ron=95.0,
        rvp_psi=120.0,
        tbp_10_f=-40.0,
        tbp_50_f=20.0,
        tbp_90_f=80.0,
        note="C3/C4 incl. olefins → LPG / alkylation",
    ),
    "fcc_naphtha": StreamComposition(
        name="fcc_naphtha",
        family=FAMILY_NAPHTHA,
        api=55.0,
        specific_gravity=api_to_sg(55.0),
        sulfur_wt=0.05,
        ron=93.0,
        rvp_psi=6.0,
        olefins_vol=0.30,
        paraffins_vol=0.25,
        naphthenes_vol=0.25,
        aromatics_vol=0.50,
        tbp_10_f=100.0,
        tbp_50_f=250.0,
        tbp_90_f=400.0,
        note="cat gasoline — pool / soft HDT; NOT reformer default",
    ),
    "fcc_lco": StreamComposition(
        name="fcc_lco",
        family=FAMILY_DISTILLATE,
        api=22.0,
        specific_gravity=api_to_sg(22.0),
        sulfur_wt=0.35,
        ccr_wt=0.2,
        aromatics_vol=0.60,
        tbp_10_f=400.0,
        tbp_50_f=550.0,
        tbp_90_f=650.0,
        note="LCO → diesel / FO",
    ),
    "fcc_slurry": StreamComposition(
        name="fcc_slurry",
        family=FAMILY_RESID,
        api=8.0,
        specific_gravity=api_to_sg(8.0),
        sulfur_wt=0.8,
        ccr_wt=4.0,
        metals_ni_v_ppm=5.0,
        tbp_10_f=650.0,
        tbp_50_f=850.0,
        tbp_90_f=1050.0,
        note="slurry oil → FO / carbon black",
    ),
    "fcc_coke": StreamComposition(
        name="fcc_coke",
        family=FAMILY_SOLID,
        api=0.0,
        specific_gravity=1.4,
        sulfur_wt=1.5,
        ccr_wt=100.0,
        hydrogen_wt=0.05,
        note="coke burned in regenerator (heat credit)",
    ),
    "coker_dry_gas": StreamComposition(
        name="coker_dry_gas",
        family=FAMILY_LIGHT_ENDS,
        api=130.0,
        specific_gravity=0.40,
        sulfur_wt=0.02,
        hydrogen_wt=0.18,
        tbp_10_f=-120.0,
        tbp_50_f=-20.0,
        tbp_90_f=40.0,
        note="coker dry gas → fuel",
    ),
    "coker_lpg": StreamComposition(
        name="coker_lpg",
        family=FAMILY_LIGHT_ENDS,
        api=105.0,
        specific_gravity=0.56,
        sulfur_wt=0.05,
        olefins_vol=0.35,
        rvp_psi=100.0,
        tbp_10_f=-30.0,
        tbp_50_f=30.0,
        tbp_90_f=90.0,
        note="coker C3/C4 → LPG",
    ),
    "coker_naphtha": StreamComposition(
        name="coker_naphtha",
        family=FAMILY_NAPHTHA,
        api=52.0,
        specific_gravity=api_to_sg(52.0),
        sulfur_wt=0.25,
        ron=72.0,
        rvp_psi=7.0,
        olefins_vol=0.40,
        paraffins_vol=0.35,
        naphthenes_vol=0.25,
        aromatics_vol=0.40,
        tbp_10_f=100.0,
        tbp_50_f=260.0,
        tbp_90_f=400.0,
        note="olefinic high-S coker naphtha → HDT / FO; NOT reformer",
    ),
    "coker_gasoil": StreamComposition(
        name="coker_gasoil",
        family=FAMILY_GASOIL,
        api=18.0,
        specific_gravity=api_to_sg(18.0),
        sulfur_wt=0.40,
        ccr_wt=0.8,
        nitrogen_ppm=1200.0,
        metals_ni_v_ppm=3.0,
        aromatics_vol=0.45,
        tbp_10_f=450.0,
        tbp_50_f=700.0,
        tbp_90_f=950.0,
        note="coker GO → diesel/FO (optional FCC later)",
    ),
    "coker_coke": StreamComposition(
        name="coker_coke",
        family=FAMILY_SOLID,
        api=0.0,
        specific_gravity=1.5,
        sulfur_wt=3.0,
        ccr_wt=100.0,
        hydrogen_wt=0.04,
        note="petcoke product / sales",
    ),
}


def get_stream(name: str, overrides: Optional[Mapping[str, Any]] = None) -> StreamComposition:
    base = STREAM_LIBRARY.get(name)
    if base is None:
        comp = StreamComposition(name=name, note="unknown stream — generic defaults")
    else:
        comp = StreamComposition(**asdict(base))
    if overrides:
        for k, v in overrides.items():
            if hasattr(comp, k) and k != "extras":
                setattr(comp, k, v)
            else:
                try:
                    comp.extras[str(k)] = float(v)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    pass
    if "specific_gravity" not in (overrides or {}) and overrides and "api" in overrides:
        comp.specific_gravity = api_to_sg(float(overrides["api"]))
    return comp


def library_names() -> List[str]:
    return sorted(STREAM_LIBRARY.keys())


def compositions_for_streams(names: Iterable[str]) -> Dict[str, StreamComposition]:
    return {n: get_stream(n) for n in names}
