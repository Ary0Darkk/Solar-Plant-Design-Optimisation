"""
Economic Objective Function & Daily Fitness Evaluation.
"""

from typing import Optional

import numpy as np

from .simulation import DailySimulationSlice


class ObjectiveResult(float):
    """Float subclass that allows tuple unpacking for (net_profit, gross_rev, par_cost, loss_cost)."""

    def __new__(
        cls,
        net_profit: float,
        gross_rev: float = 0.0,
        par_cost: float = 0.0,
        loss_cost: float = 0.0,
    ):
        obj = super().__new__(cls, net_profit)
        obj.net_profit = float(net_profit)
        obj.gross_revenue = float(gross_rev)
        obj.parasitic_cost = float(par_cost)
        obj.loss_cost = float(loss_cost)
        return obj

    def __iter__(self):
        yield self.net_profit
        yield self.gross_revenue
        yield self.parasitic_cost
        yield self.loss_cost

    def __len__(self):
        return 4

    def __getitem__(self, index: int) -> float:
        return [self.net_profit, self.gross_revenue, self.parasitic_cost, self.loss_cost][index]


def compute_daily_objective_value(
    sim_slice: Optional[DailySimulationSlice],
    tariff_slice: np.ndarray,
    land_area: float = 10000.0,
    eta_loss: float = 0.40,
) -> ObjectiveResult:
    """
    Vectorized computation of 24-hour daily net revenue in INR.
    Formula:
        Revenue_t = Energy_t * Tariff_t
        ParCost_t = Parasitic_t * Tariff_t
        LossCost_t = eta_loss * HeatLoss_t * Tariff_t
        Net_t = Revenue_t - ParCost_t - LossCost_t

    Smart Standby Logic:
        If Energy_t == 0 and Net_t < 0, Net_t is clamped to 0.0 INR.
    """
    if sim_slice is None:
        return ObjectiveResult(-1e6, 0.0, 0.0, 0.0)

    energy = np.asarray(sim_slice.hourly_energy, dtype=np.float32)
    parasitic = np.asarray(sim_slice.hourly_parasitic, dtype=np.float32)
    heat_loss = np.asarray(sim_slice.hourly_heat_loss, dtype=np.float32)
    tariff = np.asarray(tariff_slice, dtype=np.float32)

    rev_hourly = energy * tariff
    par_hourly = parasitic * tariff
    loss_hourly = eta_loss * heat_loss * tariff

    net_hourly = rev_hourly - par_hourly - loss_hourly

    # Standby logic: clamp cloudy/night negative generation to 0
    zero_gen_mask = (energy <= 1e-5) & (net_hourly < 0.0)
    net_hourly = np.where(zero_gen_mask, 0.0, net_hourly)

    gross_revenue = float(np.sum(rev_hourly))
    parasitic_cost = float(np.sum(par_hourly))
    loss_cost = float(np.sum(loss_hourly))
    total_net = float(np.sum(net_hourly))

    return ObjectiveResult(total_net, gross_revenue, parasitic_cost, loss_cost)
