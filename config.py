"""
Global configuration file for the Concentrated Solar Power (CSP) optimization project.
Backed by typed solar_opt.config module with full backward compatibility.
"""

from solar_opt.config import get_default_config

CONFIG = get_default_config().to_legacy_dict()
