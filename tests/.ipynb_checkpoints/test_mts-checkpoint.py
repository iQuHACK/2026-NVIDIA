import numpy as np


import sys
from pathlib import Path

# Ensure repo root is importable when running `pytest` from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Point at the optimised module.  Adjust this path if your layout differs.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import classical.mts as mts


def _flip(s: np.ndarray, j: int) -> np.ndarray:
    s2 = np.asarray(s, dtype=np.int8).copy()
    s2[j] *= -1
    return s2


def _alternating_inversion(s: np.ndarray) -> np.ndarray:
    """Gauge symmetry: s_i -> (-1)^i s_i (with i starting at 1).

    With 0-based indexing, this flips indices 0,2,4,... (since they correspond
    to i=1,3,5,... in 1-based indexing).
    """
    s2 = np.asarray(s, dtype=np.int8).copy()
    s2[::2] *= -1
    return s2


def _all_bitstrings_pm1(N: int) -> np.ndarray:
    """All {-1,+1} sequences of length N, shape (2**N, N). Intended for small N."""
    if N <= 0:
        raise ValueError("N must be >= 1")
    if N > 16:
        raise ValueError("N too large for exhaustive enumeration in unit tests")

    ints = np.arange(2**N, dtype=np.uint16)
    bits = np.unpackbits(ints[:, None].view(np.uint8), axis=1)
    bits = bits[:, -N:]  # take least-significant N bits
    # Map {0,1} -> {+1,-1}
    return (1 - 2 * bits).astype(np.int8)


# ===========================================================================
# energy / generate — signatures unchanged
# ===========================================================================


def test_generate_bitstrings_shape_and_values():
    pop = mts.generate_bitstrings(k=7, N=13, seed=123)
    assert pop.shape == (7, 13)
    assert pop.dtype == np.int8
    assert set(np.unique(pop)).issubset({-1, 1})


def test_energy_single_vs_batch_consistency():
    rng = np.random.default_rng(0)
    pop = rng.choice(np.array([-1, 1], dtype=np.int8), size=(5, 9))
    Es = mts.energy(pop)
    assert Es.shape == (5,)
    for i in range(pop.shape[0]):
        assert int(mts.energy(pop[i])) == int(Es[i])


def test_energy_invariants_global_flip_and_reverse():
    rng = np.random.default_rng(1)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(17,))

    e = int(mts.energy(s))
    assert int(mts.energy(-s)) == e  # global spin flip
    assert int(mts.energy(s[::-1])) == e  # time reversal / sequence reversal


def test_energy_invariant_alternating_inversion():
    rng = np.random.default_rng(11)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(19,))
    assert int(mts.energy(_alternating_inversion(s))) == int(mts.energy(s))


def test_energy_invariants_exhaustive_small_N():
    # Exhaustively verify the PRD's gauge symmetries for a tiny N.
    pop = _all_bitstrings_pm1(8)  # 256 sequences
    E = mts.energy(pop)
    assert np.array_equal(mts.energy(-pop), E)  # inversion
    assert np.array_equal(mts.energy(pop[:, ::-1]), E)  # reversal
    assert np.array_equal(
        mts.energy(_alternating_inversion(pop)), E
    )  # alternating inversion


def test_energy_known_small_case():
    # s = [+1, +1, +1, +1]
    # C1=3, C2=2, C3=1 -> E=9+4+1=14
    s = np.ones(4, dtype=np.int8)
    assert int(mts.energy(s)) == 14


def test_energy_N1_is_zero():
    assert int(mts.energy(np.array([1], dtype=np.int8))) == 0
    assert int(mts.energy(np.array([-1], dtype=np.int8))) == 0


def test_energy_rejects_invalid_rank():
    s = np.ones((2, 3, 4), dtype=np.int8)
    try:
        _ = mts.energy(s)
        assert False, "expected ValueError for invalid ndim"
    except ValueError:
        pass


# ===========================================================================
# combine — now writes into a pre-allocated `out` buffer, returns None
# ===========================================================================


def test_combine_produces_prefix_suffix_from_parents():
    rng = np.random.default_rng(42)
    p1 = np.array([1, 1, 1, 1, 1, 1], dtype=np.int8)
    p2 = np.array([-1, -1, -1, -1, -1, -1], dtype=np.int8)

    child = np.empty_like(p1)  # caller-owned buffer
    mts.combine(p1, p2, rng, child)  # writes result into child

    # Find the crossover point by locating first index where it differs from p1.
    # Because parents are all-ones vs all-minus-ones, this is unambiguous.
    cut = int(np.argmax(child != p1))
    # If cut==0 then it was all from p2, which should never happen (cut in [1..N-1])
    assert 1 <= cut <= len(p1) - 1
    assert np.all(child[:cut] == p1[:cut])
    assert np.all(child[cut:] == p2[cut:])


def test_combine_does_not_mutate_parents():
    """combine writes only into `out`; the two parent arrays must be untouched."""
    rng = np.random.default_rng(77)
    N = 20
    p1 = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
    p2 = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
    p1_before, p2_before = p1.copy(), p2.copy()

    out = np.empty(N, dtype=np.int8)
    mts.combine(p1, p2, rng, out)

    assert np.array_equal(p1, p1_before)
    assert np.array_equal(p2, p2_before)


# ===========================================================================
# mutate_inplace — mutates the array it receives; caller must copy first
# ===========================================================================


def test_mutate_inplace_probability_extremes():
    rng = np.random.default_rng(0)
    s = np.array([1, -1, 1, -1, 1], dtype=np.int8)

    # p_mut=0 -> nothing changes
    work = s.copy()
    mts.mutate_inplace(work, p_mut=0.0, rng=rng)
    assert np.array_equal(work, s)

    # p_mut=1 -> every bit flipped
    work = s.copy()
    mts.mutate_inplace(work, p_mut=1.0, rng=rng)
    assert np.array_equal(work, -s)


def test_mutate_inplace_only_touches_its_argument():
    """A separate copy of the original must remain untouched after the call."""
    rng = np.random.default_rng(1234)
    s_orig = rng.choice(np.array([-1, 1], dtype=np.int8), size=(25,))
    s_orig_snapshot = s_orig.copy()  # independent snapshot

    work = s_orig.copy()  # the copy we hand to mutate_inplace
    mts.mutate_inplace(work, p_mut=0.5, rng=rng)

    # The snapshot (standing in for "the original") is untouched.
    assert np.array_equal(s_orig, s_orig_snapshot)
    # And work was actually modified (with p=0.5 and N=25 this is
    # astronomically unlikely to be a no-op).
    assert not np.array_equal(work, s_orig)


# ===========================================================================
# _all_deltas — replaces the old per-j _delta_energy_for_flip
# ===========================================================================


def test_all_deltas_matches_full_recompute_for_every_flip():
    """For every bit j, delta from _all_deltas must equal energy(flipped) - energy(original)."""
    rng = np.random.default_rng(3)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(14,))
    N = s.shape[0]

    C = mts._autocorr_vector(s)
    E = int(np.sum(C[1:] * C[1:], dtype=np.int64))
    assert E == int(mts.energy(s))

    s64 = s.astype(np.int64)
    D = np.empty((N - 1, N), dtype=np.int64)
    deltas = mts._all_deltas(s64, C, D)  # (N,) — one call, all flips

    for j in range(N):
        s_flipped = _flip(s, j)
        expected_delta = int(mts.energy(s_flipped)) - E
        assert (
            int(deltas[j]) == expected_delta
        ), f"delta mismatch at j={j}: got {deltas[j]}, expected {expected_delta}"


def test_all_deltas_does_not_mutate_inputs():
    """_all_deltas must not modify s64 or C (D is scratch and may be overwritten)."""
    rng = np.random.default_rng(88)
    N = 16
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
    C = mts._autocorr_vector(s)
    s64 = s.astype(np.int64)

    s64_before = s64.copy()
    C_before = C.copy()
    D = np.empty((N - 1, N), dtype=np.int64)

    mts._all_deltas(s64, C, D)

    assert np.array_equal(s64, s64_before)
    assert np.array_equal(C, C_before)


# ===========================================================================
# _apply_flip_in_place — now takes (s, s64, C, j)
# ===========================================================================


def test_apply_flip_in_place_consistent_with_recompute():
    """After flipping index j, both s/s64 and C must match a full recompute."""
    rng = np.random.default_rng(4)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(11,))
    C = mts._autocorr_vector(s)
    s64 = s.astype(np.int64)

    for j in [0, 3, 7, 10]:
        # Ground truth from scratch
        s_expected = _flip(s, j)
        C_expected = mts._autocorr_vector(s_expected)
        s64_expected = s_expected.astype(np.int64)

        # In-place update
        mts._apply_flip_in_place(s, s64, C, j)

        assert np.array_equal(s, s_expected), f"s  mismatch after flip j={j}"
        assert np.array_equal(s64, s64_expected), f"s64 mismatch after flip j={j}"
        assert np.array_equal(C, C_expected), f"C  mismatch after flip j={j}"


def test_apply_flip_in_place_round_trip():
    """Flipping the same index twice is a no-op on (s, s64, C)."""
    rng = np.random.default_rng(99)
    N = 18
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
    C = mts._autocorr_vector(s)
    s64 = s.astype(np.int64)

    s_orig = s.copy()
    s64_orig = s64.copy()
    C_orig = C.copy()

    j = 5
    mts._apply_flip_in_place(s, s64, C, j)
    mts._apply_flip_in_place(s, s64, C, j)  # flip back

    assert np.array_equal(s, s_orig)
    assert np.array_equal(s64, s64_orig)
    assert np.array_equal(C, C_orig)


# ===========================================================================
# tabu_search — public signature unchanged
# ===========================================================================


def test_tabu_search_returns_nonworse_best():
    rng = np.random.default_rng(5)
    s0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=(18,))
    e0 = int(mts.energy(s0))

    best_s, best_E = mts.tabu_search(s0, max_steps=50, seed=0)
    assert int(best_E) <= e0
    assert int(mts.energy(best_s)) == int(best_E)


def test_tabu_search_does_not_mutate_input():
    rng = np.random.default_rng(55)
    s0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=(21,))
    s0_before = s0.copy()

    _best_s, _best_E = mts.tabu_search(s0, max_steps=30, seed=0)
    assert np.array_equal(s0, s0_before)


# ===========================================================================
# MTS — public signature unchanged
# ===========================================================================


def test_mts_deterministic_with_seed():
    out1 = mts.MTS(k=12, N=16, max_iter=60, seed=123)
    out2 = mts.MTS(k=12, N=16, max_iter=60, seed=123)
    # Compare best energy and best string deterministically.
    assert int(out1[1]) == int(out2[1])
    assert np.array_equal(out1[0], out2[0])


def test_mts_best_history_monotone_nonincreasing():
    _best_s, _best_E, _pop, _Es, hist = mts.MTS(k=10, N=18, max_iter=80, seed=321)
    assert len(hist) >= 1
    assert all(hist[i] <= hist[i - 1] for i in range(1, len(hist)))


def test_mts_population0_used_when_max_iter_zero():
    rng = np.random.default_rng(7)
    pop0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=(9, 10))
    Es0 = mts.energy(pop0)
    expected_best = int(Es0.min())

    best_s, best_E, pop, Es, hist = mts.MTS(
        k=999, N=999, population0=pop0, max_iter=0, seed=0
    )
    assert int(best_E) == expected_best
    assert pop.shape == pop0.shape
    assert np.array_equal(pop, pop0.astype(np.int8))
    assert np.array_equal(Es, Es0.astype(np.int64))
    assert hist == [expected_best]


def test_mts_population0_all_sequences_matches_exhaustive_best():
    # Deterministic ground truth check: feed *all* sequences as population0,
    # and MTS(max_iter=0) must return the true exhaustive optimum energy.
    pop0 = _all_bitstrings_pm1(10)  # 1024 sequences, still fast
    Es0 = mts.energy(pop0)
    expected_best = int(Es0.min())

    _best_s, best_E, pop, Es, hist = mts.MTS(
        k=999, N=999, population0=pop0, max_iter=0, seed=0
    )
    assert pop.shape == pop0.shape
    assert np.array_equal(pop, pop0.astype(np.int8))
    assert np.array_equal(Es, Es0.astype(np.int64))
    assert int(best_E) == expected_best
    assert hist == [expected_best]


def test_mts_rejects_population0_not_2d():
    pop0 = np.ones((10,), dtype=np.int8)
    try:
        _ = mts.MTS(k=5, N=10, population0=pop0, max_iter=0, seed=0)
        assert False, "expected ValueError for non-2D population0"
    except ValueError:
        pass
