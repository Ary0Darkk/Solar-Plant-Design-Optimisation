"""
Solar Plant Design & Operational Optimization (solar_opt).

A parallel simulation-optimization pipeline for Concentrated Solar Power (CSP)
parabolic-trough plants utilizing PySAM, DEAP genetic algorithms, MPI, and MLflow/DagsHub.
"""

from dotenv import load_dotenv

# Automatically load environment variables from .env file if present
load_dotenv()

__version__ = "0.2.0"
__all__ = ["__version__"]
