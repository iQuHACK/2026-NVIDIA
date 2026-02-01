import pytest
import numpy as np
from unittest.mock import patch

import sys
from pathlib import Path
# Ensure repo root is importable when running `pytest` from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Point at the optimised module.  Adjust this path if your layout differs.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

import quantum.qe_mts as qe_mts


def test_get_interactions_small_N():
    G2, G4 = qe_mts.get_interactions(5)

    # Expected from Eq. 15 logic
    assert all(len(pair) == 2 for pair in G2)
    assert all(len(quad) == 4 for quad in G4)

    # Indices must be in range
    for pair in G2:
        assert 0 <= pair[0] < 5
        assert 0 <= pair[1] < 5
        assert pair[0] < pair[1]

    for quad in G4:
        assert quad == sorted(quad)
        assert quad[-1] < 5


def test_interactions_monotonic_growth():
    G2_6, G4_6 = qe_mts.get_interactions(6)
    G2_7, G4_7 = qe_mts.get_interactions(7)

    assert len(G2_7) >= len(G2_6)
    assert len(G4_7) >= len(G4_6)


def test_bitstring_convert_basic():
    bitstring = "1010"
    spins = qe_mts.bitstring_convert(bitstring)

    assert isinstance(spins, np.ndarray)
    assert np.array_equal(spins, np.array([1, -1, 1, -1]))


def test_bitstring_convert_only_pm1():
    bitstring = "111000"
    spins = qe_mts.bitstring_convert(bitstring)

    assert set(spins.tolist()) == {1, -1}


@pytest.fixture
def fake_samples():
    return {
        "000": 3,
        "111": 2,
        "010": 1
    }


@patch("quantum.qe_mts.cudaq.sample")
def test_quantum_population_shape(mock_sample, fake_samples):
    mock_sample.return_value = fake_samples

    pop = qe_mts.quantum_population(popsize=4, N=3)

    assert len(pop) == 4
    for individual in pop:
        assert individual.shape == (3,)
        assert set(individual.tolist()) <= {1, -1}


@patch("quantum.qe_mts.cudaq.sample")
def test_population_respects_popsize(mock_sample):
    mock_sample.return_value = {"0": 100}

    pop = qe_mts.quantum_population(popsize=10, N=1)
    assert len(pop) == 10


def test_qemts_runs_with_valid_population():
    population = [
        np.array([1, -1, 1]),
        np.array([-1, 1, -1]),
        np.array([1, 1, -1])
    ]

    result = qe_mts.qe_mts(population)

    # MTS returns a tuple; verify structure
    assert isinstance(result, tuple)
    assert len(result) >= 3


def test_spin_flip_symmetry():
    s = np.array([1, -1, 1, -1])
    flipped = -s

    assert np.all(np.abs(s) == 1)
    assert np.all(np.abs(flipped) == 1)

