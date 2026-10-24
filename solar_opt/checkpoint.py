"""
Gzip-Compressed State Persistence and Checkpoint Integrity.
"""

import gzip
import hashlib
import json
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def compute_config_signature(config: Any) -> str:
    """Compute 12-char SHA-256 fingerprint of optimization parameters."""
    cfg_data = {}
    if hasattr(config, "to_dict"):
        cfg_data = config.to_dict()
    elif isinstance(config, dict):
        cfg_data = config
    else:
        for attr in ["pop_size", "num_generations", "cxpb", "mutpb", "seed"]:
            if hasattr(config, attr):
                cfg_data[attr] = getattr(config, attr)

    serialized = json.dumps(cfg_data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def save_compressed_checkpoint(
    filepath: Path,
    population: List[Any],
    generation: int,
    hall_of_fame: Optional[List[Any]] = None,
    logbook: Optional[Any] = None,
    signature: str = "",
) -> Path:
    """Atomically save gzipped checkpoint to disk."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "generation": generation,
        "population": population,
        "hall_of_fame": hall_of_fame or [],
        "logbook": logbook,
        "signature": signature,
    }

    # Atomic write to temporary file in same directory then rename
    dir_path = filepath.parent
    with tempfile.NamedTemporaryFile("wb", dir=dir_path, delete=False, suffix=".tmp") as tmp:
        tmp_name = tmp.name
        with gzip.GzipFile(fileobj=tmp, mode="wb", compresslevel=6) as gz:
            pickle.dump(data, gz, protocol=pickle.HIGHEST_PROTOCOL)

    Path(tmp_name).replace(filepath)
    return filepath


def load_compressed_checkpoint(filepath: Path) -> Dict[str, Any]:
    """Load and decompress checkpoint file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    with gzip.open(filepath, "rb") as gz:
        return pickle.load(gz)
