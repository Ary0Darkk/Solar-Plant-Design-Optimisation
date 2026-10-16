"""
Data models and types for PySAM simulations.
"""

from typing import List, Sequence, TypedDict, Union


class SimulationOutputs(TypedDict):
    """Structured outputs returned from a successful PySAM simulation."""

    hourly_energy: Union[Sequence[float], List[float]]
    pc_htf_pump_power: Union[Sequence[float], List[float]]
    field_htf_pump_power: Union[Sequence[float], List[float]]
    field_collector_tracking_power: Union[Sequence[float], List[float]]
    pc_startup_thermal_power: Union[Sequence[float], List[float]]
    field_piping_thermal_loss: Union[Sequence[float], List[float]]
    receiver_thermal_loss: Union[Sequence[float], List[float]]
    annual_energy: float
    gross_annual_energy: float
    land_area: float
    land_cost: float
    total_installed_cost: float
