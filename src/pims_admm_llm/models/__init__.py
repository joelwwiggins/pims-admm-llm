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
from .unit_specs import (
    default_process_conditions,
    unit_catalog,
    unit_yield_stream_names,
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

try:
    from .process_pool import (
        ProcessPoolResult,
        build_process_pool_yield_library,
        process_pool_once,
        solve_process_pool_mip,
    )
except ImportError:  # pragma: no cover
    ProcessPoolResult = None  # type: ignore
    build_process_pool_yield_library = None  # type: ignore
    process_pool_once = None  # type: ignore
    solve_process_pool_mip = None  # type: ignore

# W2B recursive quality (standalone; keep package import resilient)
try:
    from .quality_recursive import (
        evaluate_recursive_quality,
        resolve_gasoline_components,
        solve_full_plant_with_recursive_quality,
    )

    # alias used by docs / thin hooks
    apply_recursive_quality = evaluate_recursive_quality
except ImportError:  # pragma: no cover
    apply_recursive_quality = None  # type: ignore
    evaluate_recursive_quality = None  # type: ignore
    resolve_gasoline_components = None  # type: ignore
    solve_full_plant_with_recursive_quality = None  # type: ignore

# Base-delta CDU/FCC submodels + auto-route (stream compositions first)
try:
    from .base_delta import (
        auto_wire_edges_for_units,
        build_cdu_base_delta,
        build_coker_base_delta,
        build_fcc_base_delta,
        process_modes_cdu,
        process_modes_coker,
        process_modes_fcc,
        unit_submodels_cdu_fcc,
    )
    from .cdu_fcc import CduFccResult, solve_cdu_fcc, solve_cdu_fcc_coker
    from .auto_route import best_route, complete_missing_edges, guess_route
    from .stream_composition import StreamComposition, get_stream
except ImportError:  # pragma: no cover
    auto_wire_edges_for_units = None  # type: ignore
    build_cdu_base_delta = None  # type: ignore
    build_coker_base_delta = None  # type: ignore
    build_fcc_base_delta = None  # type: ignore
    process_modes_cdu = None  # type: ignore
    process_modes_coker = None  # type: ignore
    process_modes_fcc = None  # type: ignore
    unit_submodels_cdu_fcc = None  # type: ignore
    CduFccResult = None  # type: ignore
    solve_cdu_fcc = None  # type: ignore
    solve_cdu_fcc_coker = None  # type: ignore
    best_route = None  # type: ignore
    complete_missing_edges = None  # type: ignore
    guess_route = None  # type: ignore
    StreamComposition = None  # type: ignore
    get_stream = None  # type: ignore


try:
    from .assay_swing import (
        import_crude_from_assays_package,
        import_detailed_assay_json,
        list_importable_assays,
        solve_cdu_swing_cuts,
        allocate_cut_by_cut_points,
        cdu_cut_point_modes,
        normalize_cut_points,
        solve_cdu_from_cut_points,
        cdu_yields_and_props_from_assay,
        build_heart_swing_library,
    )
except ImportError:  # pragma: no cover
    import_crude_from_assays_package = None  # type: ignore
    import_detailed_assay_json = None  # type: ignore
    list_importable_assays = None  # type: ignore
    solve_cdu_swing_cuts = None  # type: ignore
    cdu_yields_and_props_from_assay = None  # type: ignore
    build_heart_swing_library = None  # type: ignore

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
    "apply_recursive_quality",
    "evaluate_recursive_quality",
    "resolve_gasoline_components",
    "solve_full_plant_with_recursive_quality",
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
    "default_process_conditions",
    "unit_catalog",
    "unit_yield_stream_names",
    "FullPlantResult",
    "admm_price_directed_plant",
    "solve_full_plant",
    "solve_all_plant_blocks",
    "MultiPeriodResult",
    "solve_multi_period",
    "ProcessPoolResult",
    "build_process_pool_yield_library",
    "process_pool_once",
    "solve_process_pool_mip",
    "auto_wire_edges_for_units",
    "build_cdu_base_delta",
    "build_coker_base_delta",
    "build_fcc_base_delta",
    "process_modes_cdu",
    "process_modes_coker",
    "process_modes_fcc",
    "unit_submodels_cdu_fcc",
    "CduFccResult",
    "solve_cdu_fcc",
    "solve_cdu_fcc_coker",
    "best_route",
    "complete_missing_edges",
    "guess_route",
    "StreamComposition",
    "import_crude_from_assays_package",
    "import_detailed_assay_json",
    "list_importable_assays",
    "solve_cdu_swing_cuts",
    "allocate_cut_by_cut_points",
    "cdu_cut_point_modes",
    "normalize_cut_points",
    "solve_cdu_from_cut_points",
    "cdu_yields_and_props_from_assay",
    "build_heart_swing_library",
    "get_stream",
]
