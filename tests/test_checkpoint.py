"""
Unit tests for compressed checkpoint persistence and restoration.
"""

from pathlib import Path

from optimized_pipeline.checkpoint import (
    compute_signature,
    load_compressed_checkpoint,
    save_compressed_checkpoint,
)
from optimized_pipeline.config import DESIGN_BOUNDS, OPERATIONAL_BOUNDS


def test_save_and_load_compressed_checkpoint(tmp_path: Path):
    """Verify that checkpoint state can be saved and restored atomically with gzip compression."""
    file_path = tmp_path / "test_cp.pkl.gz"
    ckpt_key = compute_signature(
        season="winter",
        design_names=DESIGN_BOUNDS.names,
        operational_names=OPERATIONAL_BOUNDS.names,
        design_lb=DESIGN_BOUNDS.lb,
        design_ub=DESIGN_BOUNDS.ub,
        operational_lb=OPERATIONAL_BOUNDS.lb,
        operational_ub=OPERATIONAL_BOUNDS.ub,
        pop_size=4,
        num_generations=2,
        cxpb=0.8,
        mutpb=0.2,
        indpb=0.1,
    )

    data = {
        "generation": 1,
        "ckpt_key": ckpt_key,
        "pop": [([1, 2, 3], (100.0,)), ([4, 5, 6], (200.0,))],
        "total_successful": 2,
        "total_penalized": 0,
        "best_fitness_ever": 200.0,
    }

    # Save
    success = save_compressed_checkpoint(data, file_path)
    assert success is True
    assert file_path.exists()

    # Load with matching key
    loaded = load_compressed_checkpoint(
        file_path, expected_key=ckpt_key, pop_size=2, chromosome_len=3
    )
    assert loaded is not None
    assert loaded["generation"] == 1
    assert loaded["best_fitness_ever"] == 200.0
    assert len(loaded["pop"]) == 2


def test_load_nonexistent_checkpoint(tmp_path: Path):
    """Verify that loading a non-existent file returns None without raising."""
    file_path = tmp_path / "does_not_exist.pkl.gz"
    loaded = load_compressed_checkpoint(
        file_path, expected_key="abc", pop_size=4, chromosome_len=10
    )
    assert loaded is None


def test_load_signature_mismatch(tmp_path: Path):
    """Verify that mismatched key and incompatible dimensions returns None."""
    file_path = tmp_path / "mismatch.pkl.gz"
    data = {
        "generation": 0,
        "ckpt_key": "correct_key",
        "pop": [([1, 2], (50.0,))],
    }
    save_compressed_checkpoint(data, file_path)

    # Incompatible pop size and chromosome length
    loaded = load_compressed_checkpoint(
        file_path, expected_key="wrong_key", pop_size=10, chromosome_len=20
    )
    assert loaded is None
