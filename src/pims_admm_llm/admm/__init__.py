"""ADMM coordinator for block-angular refinery LPs.

Dual variables λ are the economic shadow prices on linking (intermediate)
balances. At convergence they match the monolithic LP duals (up to solver
tolerance and sign convention of the maximize-form balance constraints).
"""

from .coordinator import (
    ADMMConfig,
    ADMMResult,
    ADMMState,
    ADMMCoordinator,
    run_admm,
)
from .subproblems import (
    BlockSolution,
    solve_cdu_block,
    solve_blender_block,
    solve_cdu_block_qp,
    solve_blender_block_qp,
    solve_blocks_pulp,
    solve_cdu_consensus,
    solve_blender_consensus,
)
from .residuals import primal_residual, dual_residual, converged

__all__ = [
    "ADMMConfig",
    "ADMMResult",
    "ADMMState",
    "ADMMCoordinator",
    "run_admm",
    "BlockSolution",
    "solve_cdu_block",
    "solve_blender_block",
    "solve_cdu_block_qp",
    "solve_blender_block_qp",
    "solve_blocks_pulp",
    "solve_cdu_consensus",
    "solve_blender_consensus",
    "primal_residual",
    "dual_residual",
    "converged",
]
