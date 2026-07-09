"""pims-admm-llm: block-angular ADMM + multi-agent LP demo for refinery planning."""

__version__ = "0.1.0"

from pims_admm_llm.admm import ADMMConfig, ADMMCoordinator, ADMMResult, run_admm

__all__ = [
    "__version__",
    "ADMMConfig",
    "ADMMCoordinator",
    "ADMMResult",
    "run_admm",
]
