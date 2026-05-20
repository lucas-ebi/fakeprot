"""Tests for simulation.py helpers."""

import numpy as np
import pytest

from fakeprot.config import SimulationConfig
from fakeprot.simulation import _dup_edge_t, _jitter_branch_length


@pytest.fixture
def cfg() -> SimulationConfig:
    return SimulationConfig(size=10, length=20, seed=1,
                            dup_boost_factor=2.0, dup_boost_decay=3.0)


class TestDupEdgeT:
    def test_decreases_with_distance(self, cfg: SimulationConfig):
        t0 = _dup_edge_t(0, cfg)   # parent at d=0 → child at d=1
        t2 = _dup_edge_t(2, cfg)   # child at d=3
        t9 = _dup_edge_t(9, cfg)   # child at d=10, nearly baseline
        assert t0 > t2 > t9

    def test_approaches_baseline_at_large_distance(self, cfg: SimulationConfig):
        t9 = _dup_edge_t(9, cfg)
        assert t9 < cfg.branch_length * 1.05

    def test_significantly_elevated_at_small_distance(self, cfg: SimulationConfig):
        t0 = _dup_edge_t(0, cfg)
        assert t0 > cfg.branch_length * 1.3

    def test_zero_decay_returns_baseline(self):
        cfg_flat = SimulationConfig(size=10, length=20, seed=1,
                                    dup_boost_factor=2.0, dup_boost_decay=0.0)
        assert _dup_edge_t(0, cfg_flat) == cfg_flat.branch_length
        assert _dup_edge_t(5, cfg_flat) == cfg_flat.branch_length

    def test_none_returns_baseline(self, cfg: SimulationConfig):
        assert _dup_edge_t(None, cfg) == cfg.branch_length

    def test_at_d_equals_tau_decays_by_one_efold(self, cfg: SimulationConfig):
        import math
        # parent at d=tau-1, child at d=tau → m = 1 + (k-1)*exp(-1) ≈ 1.368
        tau = int(cfg.dup_boost_decay)
        t = _dup_edge_t(tau - 1, cfg)
        expected = cfg.branch_length * (1.0 + (cfg.dup_boost_factor - 1.0) * math.exp(-1.0))
        assert abs(t - expected) < 1e-10


class TestJitterBranchLength:
    def test_cv_zero_returns_exact_value(self):
        assert _jitter_branch_length(0.05, 0.0) == pytest.approx(0.05)

    def test_output_is_positive(self):
        np.random.seed(0)
        for _ in range(100):
            assert _jitter_branch_length(0.05, 0.5) > 0.0

    def test_mean_is_approximately_preserved(self):
        np.random.seed(1)
        samples = [_jitter_branch_length(0.05, 0.3) for _ in range(5000)]
        assert np.mean(samples) == pytest.approx(0.05, rel=0.05)

    def test_cv_is_approximately_correct(self):
        np.random.seed(2)
        samples = [_jitter_branch_length(1.0, 0.3) for _ in range(5000)]
        assert np.std(samples) / np.mean(samples) == pytest.approx(0.3, rel=0.1)
