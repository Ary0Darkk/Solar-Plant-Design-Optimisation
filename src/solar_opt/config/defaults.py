"""
Default settings and parameters for the CSP optimization pipeline.
"""

from pathlib import Path
from typing import Dict, List, Tuple

from .models import (
    GAParameters,
    OptimizationConfig,
    SimulationSettings,
    TariffSettings,
    VariableBounds,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# -------------------------------------------------------------------------
# Seasonal representative days (Month, Day)
# -------------------------------------------------------------------------
# Each season contains 7 representative days selected across the year:
DEFAULT_SEASONS: Dict[str, List[Tuple[int, int]]] = {
    "winter": [(1, 8), (1, 13), (1, 16), (1, 28), (2, 21), (2, 22), (12, 12)],
    "summer": [(3, 9), (3, 16), (4, 12), (4, 17), (5, 7), (5, 16), (5, 21)],
    "monsoon": [(6, 13), (7, 6), (7, 15), (7, 22), (7, 23), (8, 4), (8, 11)],
    "post": [(9, 5), (9, 20), (10, 3), (10, 12), (11, 3), (11, 17), (11, 23)],
}

# -------------------------------------------------------------------------
# Design Variables & Physical Bounds (Season-wide, 5 variables)
# -------------------------------------------------------------------------
DEFAULT_DESIGN_BOUNDS = VariableBounds(
    names=(
        "specified_total_aperture",  # total aperture area [m²] (baseline: 8175)
        "Row_Distance",              # pitch between rows [m] (baseline: 12.5)
        "ColperSCA",                 # modules per SCA (baseline: 10)
        "W_aperture",                # mirror aperture width [m] (baseline: 5.75)
        "L_SCA",                     # length of collector assembly [m] (baseline: 120)
    ),
    lb=(8000, 10, 2, 4, 50),
    ub=(12000, 25, 10, 8, 200),
    types=(int, int, int, int, int),
)

# -------------------------------------------------------------------------
# Operational Variables & Physical Bounds (Daily, 3 variables)
# -------------------------------------------------------------------------
DEFAULT_OPERATIONAL_BOUNDS = VariableBounds(
    names=(
        "m_dot",       # mass-flow rate limit [kg/s] (baseline range: 1 - 12)
        "T_startup",   # power block startup temperature [°C] (baseline: 325.34)
        "T_shutdown",  # power block shutdown temperature threshold [°C] (baseline: 325.34)
    ),
    lb=(2, 300, 250),
    ub=(12, 380, 300),
    types=(int, int, int),
)


def get_default_config() -> OptimizationConfig:
    """Create a new master OptimizationConfig with validated production defaults."""
    return OptimizationConfig(
        seasons=dict(DEFAULT_SEASONS),
        design=DEFAULT_DESIGN_BOUNDS,
        operational=DEFAULT_OPERATIONAL_BOUNDS,
        ga=GAParameters(),
        tariff=TariffSettings(),
        simulation=SimulationSettings(),
    )
