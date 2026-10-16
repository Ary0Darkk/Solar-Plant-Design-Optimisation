# config.py
"""
Global configuration file for the Concentrated Solar Power (CSP) optimization project.
Modify values here — all simulation, indexing, and optimization scripts use these settings.
"""

# NOTE: In PySAM Outputs, "Number of SCAs per loop" is accessible via model.Outputs.nSCA (and total loops via model.Outputs.nLoops).

CONFIG = {
    # -------------------------------------------------------------------------
    # Data file paths
    # -------------------------------------------------------------------------
    "demand_file_path": "electricity_data/Yearly Demand Profile Haryana-Himanchal.xlsx",
    "show_demand_plot": False,
    "show_price_plot": False,

    # -------------------------------------------------------------------------
    # Electricity Tariff Settings (INR / kWh)
    # -------------------------------------------------------------------------
    # Scaled linearly against Haryana state hourly grid demand.
    # Default range: 3.0 to 8.0 INR/kWh (typical Indian CSP / solar PPA tariff).
    "min_price": 3.0,
    "max_price": 8.0,

    # -------------------------------------------------------------------------
    # PySAM Simulation Settings
    # -------------------------------------------------------------------------
    # Baseline plant configuration JSON
    "json_file": "simulation/Gurgaon_plant.json",
    # PySAM technology model
    "model": "PhysicalTroughNone",
    # Log individual simulation durations and outputs (set True for deep debugging)
    "log_each_simulation": False,
    # Penalty value assigned to failed or physically infeasible simulations
    "penalty": -1e13,

    # -------------------------------------------------------------------------
    # Seasonal representative days
    # -------------------------------------------------------------------------
    # Formatted as (month, day).
    # Uses standard non-leap reference year (365 days / 8,760 hours, Feb 29 omitted),
    # matching the 8,760-hour Himawari weather and Haryana electricity price profiles.
    # Each season contains 7 representative days:
    #   winter:  Jan, Feb, and Dec dates (spans calendar year boundary)
    #   summer:  Mar, Apr, May dates
    #   monsoon: Jun, Jul, Aug dates
    #   post:    Sep, Oct, Nov dates
    "SEASONS": {
        "winter": [(1, 8), (1, 13), (1, 16), (1, 28), (2, 21), (2, 22), (12, 12)],
        "summer": [(3, 9), (3, 16), (4, 12), (4, 17), (5, 7), (5, 16), (5, 21)],
        "monsoon": [(6, 13), (7, 6), (7, 15), (7, 22), (7, 23), (8, 4), (8, 11)],
        "post": [(9, 5), (9, 20), (10, 3), (10, 12), (11, 3), (11, 17), (11, 23)],
    },

    # -------------------------------------------------------------------------
    # Design Variables & Physical Bounds (Season-wide, 5 variables)
    # -------------------------------------------------------------------------
    # Overrides applied across all representative days of a season:
    #   specified_total_aperture: Total solar field reflective area [m²] (baseline: 8,175)
    #   Row_Distance: Pitch between parallel trough rows [m] (baseline: 12.5m)
    #   ColperSCA: Number of collector modules per SCA (baseline: 10)
    #   W_aperture: Mirror aperture width [m] (commercial troughs: 4-8m, baseline: 5.75m)
    #   L_SCA: Total length of single collector assembly [m] (baseline: 120m)
    #
    # Note on Bounds:
    #   Row_Distance lb (10) > W_aperture ub (8), guaranteeing Row_Distance > W_aperture
    #   at all times to prevent mechanical mirror collision penalties during GA search.
    "design": {
        "overrides": [
            "specified_total_aperture",  # total aperture area [m²]
            "Row_Distance",              # row spacing [m]
            "ColperSCA",                 # num of modules per SCA
            "W_aperture",                # width of SCA aperture [m]
            "L_SCA",                     # length of collector assembly [m]
        ],
        # Bounds: [specified_total_aperture, Row_Distance, ColperSCA, W_aperture, L_SCA]
        "lb": [8000, 10, 2, 4, 50],
        "ub": [12000, 25, 10, 8, 200],
        "types": [
            int,
            int,
            int,
            int,
            int,
        ],
    },

    # -------------------------------------------------------------------------
    # Operational Variables & Bounds (Daily, 3 variables per representative day)
    # -------------------------------------------------------------------------
    # Overrides optimized independently for each representative day (7 days x 3 vars = 21 vars):
    #   m_dot: HTF mass flow rate limit [kg/s] (baseline range: 1 - 12 kg/s)
    #   T_startup: Power block startup temperature setpoint [°C] (baseline: 325.34°C)
    #   T_shutdown: Power block shutdown temperature threshold [°C] (baseline: 325.34°C)
    #
    # Note on Bounds:
    #   T_shutdown upper bound (300°C) is <= T_startup lower bound (300°C), ensuring
    #   the power cycle does not immediately trip into shutdown upon startup.
    "operational": {
        "overrides": [
            "m_dot",       # mass-flow rate limit [kg/s]
            "T_startup",   # power block startup temperature [°C]
            "T_shutdown",  # power block shutdown temperature [°C]
        ],
        # Bounds: [m_dot, T_startup, T_shutdown]
        "lb": [2, 300, 250],
        "ub": [12, 380, 300],
        "types": [int, int, int],
    },

    # -------------------------------------------------------------------------
    # DEAP Genetic Algorithm Hyperparameters
    # -------------------------------------------------------------------------
    # Checkpointing and reproducibility
    "resume_from_checkpoint": True,
    "checkpoint_interval": 1,
    "random_seed": 41,

    # Population and generation sizing:
    #   HPC cluster profile: pop_size: 112 (1 individual per CPU core on 112-core nodes)
    #   Local desktop profile: override with CLI flags: --pop-size 4 --generations 2
    "pop_size": 112,
    "num_generations": 100,
    "tournament_size": 7,
    "hall_of_fame_size": 5,

    # Genetic operators
    "cxpb": 0.8,    # Crossover probability
    "mutpb": 0.2,   # Mutation probability
    "indpb": 0.1,   # Independent gene mutation probability (~1/num_genes)
    "verbose": True,
}
