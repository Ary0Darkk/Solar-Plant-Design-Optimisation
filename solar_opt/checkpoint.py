"""
Gzip-compressed, atomic checkpoint persistence and state restoration.
"""

import gzip
import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

logger = logging.getLogger("Solar_Optimisation")


def compute_signature(
    season: str,
    design_names: Sequence[str],
    operational_names: Sequence[str],
    design_lb: Sequence[int],
    design_ub: Sequence[int],
    operational_lb: Sequence[int],
    operational_ub: Sequence[int],
    pop_size: int,
    num_generations: Optional[int] = None,
    cxpb: float = 0.8,
    mutpb: float = 0.2,
    indpb: float = 0.1,
    include_generations: bool = False,
) -> str:
    """
    Generate a deterministic 12-char SHA-256 hash identifying the GA problem configuration.
    num_generations is excluded by default so runs can be resumed and extended to higher limits.
    """
    data = {
        "season": season,
        "design_names": list(design_names),
        "operational_names": list(operational_names),
        "design_lb": list(design_lb),
        "design_ub": list(design_ub),
        "operational_lb": list(operational_lb),
        "operational_ub": list(operational_ub),
        "pop_size": pop_size,
        "cxpb": cxpb,
        "mutpb": mutpb,
        "indpb": indpb,
    }
    if include_generations and num_generations is not None:
        data["num_generations"] = num_generations

    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def save_compressed_checkpoint(data: Any, file_path: Path) -> bool:
    """
    Atomically save data to a gzip-compressed pickle file (.pkl.gz).
    Writes to a process-unique .tmp file first and replaces target file atomically
    with retry backoff to handle transient Windows file locks.
    """
    tmp_path = file_path.with_name(f"{file_path.name}.{os.getpid()}.tmp")
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(tmp_path, "wb", compresslevel=6) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        for attempt in range(5):
            try:
                os.replace(tmp_path, file_path)
                return True
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
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


def find_compatible_checkpoint(
    season_dir: Path,
    expected_key: str,
    pop_size: int,
    chromosome_len: int,
) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """
    Scan season checkpoint directories for the newest compatible checkpoint.
    Returns (checkpoint_path, checkpoint_data) or None if no compatible checkpoint exists.
    """
    if not season_dir.exists():
        return None

    # First check exact signature match
    exact_file = season_dir / expected_key / "checkpoint_latest.pkl.gz"
    if exact_file.exists():
        data = load_compressed_checkpoint(exact_file, expected_key, pop_size, chromosome_len)
        if data is not None:
            return exact_file, data

    # Fall back to searching other signature subdirectories
    candidate_files = list(season_dir.glob("*/checkpoint_latest.pkl.gz"))
    candidate_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for cf in candidate_files:
        data = load_compressed_checkpoint(cf, expected_key, pop_size, chromosome_len)
        if data is not None:
            logger.info("Discovered newest compatible checkpoint in %s", cf.parent.name)
            return cf, data

    return None
