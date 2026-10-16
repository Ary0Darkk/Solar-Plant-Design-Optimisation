"""
Gzip-compressed, atomic checkpoint persistence and state restoration.
"""

import gzip
import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger("NYS_Optimisation")


def compute_signature(
    season: str,
    design_names: Sequence[str],
    operational_names: Sequence[str],
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
    """Generate a deterministic 12-char SHA-256 hash identifying the GA run configuration."""
    data = {
        "season": season,
        "design_names": list(design_names),
        "operational_names": list(operational_names),
        "design_lb": list(design_lb),
        "design_ub": list(design_ub),
        "operational_lb": list(operational_lb),
        "operational_ub": list(operational_ub),
        "pop_size": pop_size,
        "num_generations": num_generations,
        "cxpb": cxpb,
        "mutpb": mutpb,
        "indpb": indpb,
    }
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def save_compressed_checkpoint(data: Any, file_path: Path) -> bool:
    """
    Atomically save data to a gzip-compressed pickle file (.pkl.gz).
    Writes to .tmp first and replaces target file atomically.
    """
    tmp_path = file_path.with_suffix(".tmp")
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(tmp_path, "wb", compresslevel=6) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(file_path)
        return True
    except Exception as exc:
        logger.warning("Failed to save compressed checkpoint to %s: %s", file_path, exc)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return False


def load_compressed_checkpoint(
    file_path: Path,
    expected_key: str,
    pop_size: int,
    chromosome_len: int,
) -> Optional[Dict[str, Any]]:
    """
    Load and validate a gzip-compressed checkpoint file.
    Accepts exact key match or compatible chromosome length.
    """
    if not file_path.exists():
        return None

    try:
        with gzip.open(file_path, "rb") as f:
            cp = pickle.load(f)

        key_matches = cp.get("ckpt_key") == expected_key
        if not key_matches:
            pop_data = cp.get("pop", [])
            if len(pop_data) == pop_size and len(pop_data[0][0]) == chromosome_len:
                logger.info(
                    "Accepting compatible checkpoint (key=%s, gen=%d, pop=%d)",
                    cp.get("ckpt_key"),
                    cp.get("generation", 0),
                    len(pop_data),
                )
                key_matches = True

        if not key_matches:
            logger.warning("Checkpoint signature mismatch in %s. Starting fresh run.", file_path)
            return None

        return cp

    except Exception as exc:
        logger.error("Failed to load compressed checkpoint %s: %s", file_path, exc)
        return None
