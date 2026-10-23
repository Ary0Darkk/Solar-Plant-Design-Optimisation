"""
Tests for atomic gzip checkpoint persistence and recovery.
"""

from pathlib import Path

from solar_opt.checkpoint import (
    compute_signature,
    load_compressed_checkpoint,
    save_compressed_checkpoint,
)


def test_compute_signature_deterministic():
    sig1 = compute_signature("summer", ["d1"], ["o1"], [10], [20], [1], [5], pop_size=100)
    sig2 = compute_signature("summer", ["d1"], ["o1"], [10], [20], [1], [5], pop_size=100)
    assert sig1 == sig2
    assert len(sig1) == 12


def test_save_and_load_compressed_checkpoint(tmp_path: Path):
    ckpt_file = tmp_path / "checkpoint_test.pkl.gz"
    payload = {
        "season": "summer",
        "ckpt_key": "test_sig_123",
        "generation": 5,
        "pop": [([1, 2, 3], (100.0,))],
    }

    assert save_compressed_checkpoint(payload, ckpt_file) is True
    assert ckpt_file.exists()

    loaded = load_compressed_checkpoint(
        ckpt_file,
        expected_key="test_sig_123",
        pop_size=1,
        chromosome_len=3,
    )
    assert loaded is not None
    assert loaded["season"] == "summer"
    assert loaded["generation"] == 5
