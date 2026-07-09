"""Refinery models: legacy toy blocks + wave2 full plant (assays/FCC/coker/reformer/tanks)."""

from .data import (
    CrudeAssay,
    InventorySpec,
    ProductSpec,
    RefineryData,
    UtilitySpec,
    default_data_path,
    load_crude_data,
    validate_refinery_data,
)
from .blocks import (
    BlockNames,
    MonolithicResult,
    SubproblemResult,
    build_blender_subproblem,
    build_cdu_subproblem,
    build_inventory_subproblem,
    build_monolithic_lp,
    build_utilities_subproblem,
    describe_block_angular_structure,
    extract_monolithic_solution,
    solve_all_subproblems,
    solve_blender_subproblem,
    solve_cdu_subproblem,
    solve_inventory_subproblem,
    solve_monolithic,
    solve_utilities_subproblem,
)
from .assay_loader import (
    assays_to_refinery_data,
    crude_properties_list,
    default_assays_path,
    default_intermediates_path,
    default_routing_path,
    intermediate_properties_list,
    is_assay_shaped,
    load_assays_excel,
    load_assays_json,
    load_intermediates_json,
    load_routing,
    write_template_excel,
)
from .properties import FeedProperties, crude_to_props
from .quality_blender import (
    GasolineQualityConfig,
    QualityComponent,
    blend_quality_closed_form,
    load_component_qualities,
    ron_blending_index,
    ron_from_blending_index,
)
from .yields import (
    cdu_yields_from_assay,
    coker_yields,
    fcc_yields,
    reformer_yields,
)

# Optional wave2 plant modules (may land while W1/W2 run in parallel)
try:
    from .full_plant import FullPlantResult, admm_price_directed_plant, solve_full_plant
except ImportError:  # pragma: no cover
    FullPlantResult = None  # type: ignore
    admm_price_directed_plant = None  # type: ignore
    solve_full_plant = None  # type: ignore

try:
    from .plant_blocks import solve_all_plant_blocks
except ImportError:  # pragma: no cover
    solve_all_plant_blocks = None  # type: ignore

try:
    from .multi_period import MultiPeriodResult, solve_multi_period
except ImportError:  # pragma: no cover
    MultiPeriodResult = None  # type: ignore
    solve_multi_period = None  # type: ignore

__all__ = [
    "CrudeAssay",
    "InventorySpec",
    "ProductSpec",
    "RefineryData",
    "UtilitySpec",
    "default_data_path",
    "load_crude_data",
    "validate_refinery_data",
    "BlockNames",
    "MonolithicResult",
    "SubproblemResult",
    "build_monolithic_lp",
    "extract_monolithic_solution",
    "solve_monolithic",
    "build_cdu_subproblem",
    "build_inventory_subproblem",
    "build_blender_subproblem",
    "build_utilities_subproblem",
    "solve_cdu_subproblem",
    "solve_inventory_subproblem",
    "solve_blender_subproblem",
    "solve_utilities_subproblem",
    "solve_all_subproblems",
    "describe_block_angular_structure",
    "FeedProperties",
    "crude_to_props",
    "GasolineQualityConfig",
    "QualityComponent",
    "blend_quality_closed_form",
    "load_component_qualities",
    "ron_blending_index",
    "ron_from_blending_index",
    "assays_to_refinery_data",
    "crude_properties_list",
    "default_assays_path",
    "default_intermediates_path",
    "default_routing_path",
    "intermediate_properties_list",
    "is_assay_shaped",
    "load_assays_excel",
    "load_assays_json",
    "load_intermediates_json",
    "load_routing",
    "write_template_excel",
    "cdu_yields_from_assay",
    "coker_yields",
    "fcc_yields",
    "reformer_yields",
    "FullPlantResult",
    "admm_price_directed_plant",
    "solve_full_plant",
    "solve_all_plant_blocks",
    "MultiPeriodResult",
    "solve_multi_period",
]
