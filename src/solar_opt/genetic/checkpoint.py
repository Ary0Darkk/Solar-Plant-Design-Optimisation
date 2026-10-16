"""
Deterministic checkpoint hashing, atomic persistence, and state recovery.
"""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config.defaults import PROJECT_ROOT

logger = logging.getLogger("NYS_Optimisation")


def compute_checkpoint_key(
    season: str,
    design_vars: Sequence[str],
    operational_vars: Sequence[str],
    design_types: Sequence[type],
    operational_types: Sequence[type],
    design_lb: Sequence[int],
    design_ub: Sequence[int],
    operational_lb: Sequence[int],
    operational_ub: Sequence[int],
    pop_size: int,
    num_generations: int,
    cxpb: float,
    mutpb: float,
    indpb: float,
) -> str:
    """Compute a deterministic 12-character SHA256 hash identifying configuration signature."""
    payload = {
        "season": season,
        "vars": list(design_vars) + list(operational_vars),
        "types": [t.__name__ for t in list(design_types) + list(operational_types)],
        "lb": list(design_lb) + list(operational_lb),
        "ub": list(design_ub) + list(operational_ub),
        "pop": pop_size,
        "gens": num_generations,
        "cxpb": cxpb,
        "mutpb": mutpb,
        "indpb": indpb,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def safe_pickle_save(data: Any, file_path: Path) -> bool:
    """Write data to a temporary file and atomically replace target file."""
    tmp_path = file_path.with_suffix(".tmp")
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f)
        tmp_path.replace(file_path)
        return True
    except Exception as exc:
        logger.warning("Checkpoint atomic save failed for %s: %s", file_path, exc)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return False


def load_checkpoint(
    checkpoint_file: Path,
    expected_key: str,
    pop_size: int,
    chromosome_length: int,
) -> Optional[Dict[str, Any]]:
    """
    Load and validate checkpoint. Accepts exact key match or compatible chromosome length.
    """
    if not checkpoint_file.exists():
        return None

    try:
        with open(checkpoint_file, "rb") as f:
            cp = pickle.load(f)

        valid_key = cp.get("ckpt_key") == expected_key
        if not valid_key:
            pop_data = cp.get("pop", [])
            if len(pop_data) == pop_size and len(pop_data[0][0]) == chromosome_length:
                logger.info(
                    "Accepting compatible checkpoint (key: %s, gen: %d, pop: %d).",
                    cp.get("ckpt_key"),
                    cp.get("generation", 0),
                    len(pop_data),
                )
                valid_key = True

        if not valid_key:
            logger.warning(
                "Checkpoint key mismatch! Checkpoint: %s, Expected: %s. Starting fresh.",
                cp.get("ckpt_key"),
                expected_key,
            )
            return None

        return cp

    except Exception as exc:
        logger.error("Checkpoint corrupted at %s: %s", checkpoint_file, exc)
        return None
