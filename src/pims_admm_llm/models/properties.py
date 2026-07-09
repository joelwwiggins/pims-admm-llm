"""Feed property vectors that drive unit yields (PIMS-style assay attributes)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class FeedProperties:
    """Basic properties used by yield response surfaces."""

    name: str = "feed"
    api: float = 30.0
    sulfur_wt: float = 1.0
    ccr_wt: float = 2.0
    nitrogen_ppm: float = 1000.0
    paraffins_vol: float = 0.33
    naphthenes_vol: float = 0.33
    aromatics_vol: float = 0.34
    extras: Dict[str, float] = field(default_factory=dict)

    @property
    def pna_sum(self) -> float:
        return self.paraffins_vol + self.naphthenes_vol + self.aromatics_vol

    @property
    def n_plus_a(self) -> float:
        """Naphthenes + aromatics — reformer severity / yield driver."""
        return self.naphthenes_vol + self.aromatics_vol

    def blend(self, other: "FeedProperties", w_self: float, w_other: float) -> "FeedProperties":
        """Volume-weighted blend of two property vectors."""
        t = max(w_self + w_other, 1e-12)
        a, b = w_self / t, w_other / t
        return FeedProperties(
            name=f"blend({self.name}+{other.name})",
            api=a * self.api + b * other.api,
            sulfur_wt=a * self.sulfur_wt + b * other.sulfur_wt,
            ccr_wt=a * self.ccr_wt + b * other.ccr_wt,
            nitrogen_ppm=a * self.nitrogen_ppm + b * other.nitrogen_ppm,
            paraffins_vol=a * self.paraffins_vol + b * other.paraffins_vol,
            naphthenes_vol=a * self.naphthenes_vol + b * other.naphthenes_vol,
            aromatics_vol=a * self.aromatics_vol + b * other.aromatics_vol,
        )


def crude_to_props(crude: dict) -> FeedProperties:
    return FeedProperties(
        name=str(crude.get("name", "crude")),
        api=float(crude.get("api", 30.0)),
        sulfur_wt=float(crude.get("sulfur_wt", crude.get("sulfur_wt_pct", 1.0))),
        ccr_wt=float(crude.get("ccr_wt", 2.0)),
        nitrogen_ppm=float(crude.get("nitrogen_ppm", 1000.0)),
        paraffins_vol=float(crude.get("paraffins_vol", 0.33)),
        naphthenes_vol=float(crude.get("naphthenes_vol", 0.33)),
        aromatics_vol=float(crude.get("aromatics_vol", 0.34)),
    )
