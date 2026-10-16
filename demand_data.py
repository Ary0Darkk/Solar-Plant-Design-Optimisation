"""
Electricity demand and dynamic price module.
Backward compatibility wrapper delegating to solar_opt.economic.
"""

from solar_opt.economic.pricing import (
    compute_dynamic_tariff as calc_dynamic_price,
    get_cached_dynamic_prices,
    load_formatted_demand_data as format_data,
)


def get_dynamic_price(min_price=3.0, max_price=8.0):
    formatted = format_data()
    tariff = calc_dynamic_price(formatted["Hourly Demand Met (in MW)"], min_price, max_price)
    formatted["dynamic_price"] = tariff
    return formatted


__all__ = ["format_data", "calc_dynamic_price", "get_dynamic_price", "get_cached_dynamic_prices"]
