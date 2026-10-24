"""
Tests for checkpoint saving, loading, compression, and signature verification.
"""

from pathlib import Path

from solar_opt.checkpoint import (
    compute_config_signature,
    load_compressed_checkpoint,
    save_compressed_checkpoint,
)
from solar_opt.config import PipelineConfig


def test_compute_signature_deterministic():
    cfg1 = PipelineConfig()
    cfg2 = PipelineConfig()
    sig1 = compute_config_signature(cfg1)
    sig2 = compute_config_signature(cfg2)
    assert sig1 == sig2
    assert len(sig1) == 12

    cfg2.ga.pop_size = 999
    sig3 = compute_config_signature(cfg2)
    assert sig1 != sig3


def test_save_and_load_compressed_checkpoint(tmp_path: Path):
    pop = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    ckpt_file = tmp_path / "checkpoint_test.pkl.gz"
    sig = "abc123def456"

    saved_path = save_compressed_checkpoint(
        ckpt_file,
        population=pop,
        generation=5,
        hall_of_fame=[pop[0]],
        logbook=[{"gen": 5, "avg": 100.0}],
        signature=sig,
    )
    assert saved_path.exists()

    loaded = load_compressed_checkpoint(saved_path)
    assert loaded["generation"] == 5
    assert loaded["population"] == pop
    assert loaded["signature"] == sig
    assert len(loaded["hall_of_fame"]) == 1
