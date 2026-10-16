import numpy as np
import pytest

from src.metrics import smoothness_ratio, time_order_correlation


@pytest.fixture
def traj():
    rng = np.random.default_rng(0)
    return np.cumsum(rng.normal(size=(50, 16)), axis=0)  # random walk, (T, D)


def test_scale_and_translation_invariant(traj):
    moved = 37.0 * traj + 5.0
    assert smoothness_ratio(moved) == pytest.approx(smoothness_ratio(traj))
    assert time_order_correlation(moved) == pytest.approx(time_order_correlation(traj))


def test_straight_line_is_perfectly_ordered():
    line = np.arange(30.0)[
        :, None
    ]  # integer spacing keeps equal-|dt| distances exactly tied
    assert time_order_correlation(line) == pytest.approx(1.0)
    # evenly spaced points on a segment: step = L/(T-1), mean pairwise = L(T+1)/(3(T-1))
    assert smoothness_ratio(line) == pytest.approx(3 / 31)


def test_shuffling_breaks_structure(traj):
    shuffled = traj[np.random.default_rng(1).permutation(len(traj))]
    assert smoothness_ratio(shuffled) > smoothness_ratio(traj)
    assert time_order_correlation(shuffled) < time_order_correlation(traj)
