"""Internal consistency of the leading-order theory (docs/THEORY.md)."""

import pytest

from esfc import cavity
from esfc.closure import r_star_inf


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
def test_variance_budget_adds_up(p):
    v = cavity.variance_budget(p)
    assert abs(v["frozen"] + v["rotating"] - v["instantaneous"]) < 1e-6
    assert 0 < v["rotating_filtered"] < v["rotating"]
    assert abs(v["instantaneous"] - (1 - r_star_inf(p) ** 2) / 2) < 1e-12


@pytest.mark.parametrize("p", [0.5, 1.0, 1.5, 2.0, 3.0])
def test_freezing_all_scatter_recovers_the_rician_closure(p):
    """The same second-order formula with every fluctuation frozen must give (p+2)/2 exactly."""
    assert abs(cavity.frozen_all_coefficient(p) - (p + 2) / 2) < 1e-9


@pytest.mark.parametrize("p", [1.0, 2.0])
def test_ordering_of_the_three_treatments(p):
    assert cavity.frozen_only_coefficient(p) < cavity.dynamic_coefficient(p) < (p + 2) / 2


def test_transverse_response_of_neighbours_is_exactly_one():
    """beta L / mu = 1: rotating the field rotates the population response by the same angle."""
    for p in (0.5, 1.0, 2.0):
        c = cavity._fold_constants(p)
        assert abs(c["beta"] * c["L"] / c["mu"] - 1) < 1e-12
        assert abs(c["alpha"] / c["mu"] - c["dz0_dh"]) < 1e-12


@pytest.mark.slow
def test_reaction_raises_the_fold():
    locked, drifting = cavity.reaction_parts(1.0)
    assert locked < 0 and abs(drifting) < 0.1 * abs(locked) and cavity.reaction_coefficient(1.0) > 0
