"""
End-to-end integration test running full GA on minimal generation/pop size.
"""

import pytest

from solar_opt.config import PipelineConfig
from solar_opt.engine import run_optimization


@pytest.mark.e2e
def test_e2e_micro_optimization(tmp_path):
    cfg = PipelineConfig()
    cfg.ga.num_generations = 2
    cfg.ga.pop_size = 6
    cfg.tracking.enabled = False

    artifacts = run_optimization(
        season="winter",
        config=cfg,
        workers=2,
        checkpoint_dir=tmp_path,
    )
    assert artifacts["season"] == "winter"
    assert "best_fitness" in artifacts
    assert artifacts["png"].exists()
    assert artifacts["pdf"].exists()
    assert artifacts["excel"].exists()
    assert artifacts["report"].exists()
    assert artifacts["json"].exists()
