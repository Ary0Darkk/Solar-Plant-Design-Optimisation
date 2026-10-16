"""
Custom logger module.
Backward compatibility wrapper delegating to solar_opt.tracking.logger.
"""

from solar_opt.tracking.logger import setup_rank_logger as setup_custom_logger

__all__ = ["setup_custom_logger"]
