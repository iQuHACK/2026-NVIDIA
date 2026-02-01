"""

Tests cover:
- Symmetry invariance (negation, reversal)
- Known optimal values for small N
- Bitstring conversion correctness
- Canonicalization idempotence and equivalence-class correctness
- Hamiltonian encoding matches classical energy
- CVaR computation
"""

from labs_core import (
    compute_energy,
    random_sequence,
    bitstring_to_sequence,
    sequence_to_bitstring,
    canonicalize_sequence,
    get_weighted_interactions,
    compute_energy_from_hamiltonian,
    compute_cvar,
)


def test_symmetry_invariance():
    """Test that energy is invariant under LABS symmetries."""
    print("Test: Symmetry Invariance")
    
    for N in [5, 7, 10]:
        for _ in range(10):
            seq = random_sequence(N)
            e_orig = compute_energy(seq)
            e_neg = compute_energy([-s for s in seq])
            e_rev = compute_energy(seq[::-1])
            e_both = compute_energy([-s for s in seq[::-1]])
            
            assert e_orig == e_neg, f"Negation symmetry failed: {e_orig} != {e_neg}"
            assert e_orig == e_rev, f"Reversal symmetry failed: {e_orig} != {e_rev}"
            assert e_orig == e_both, f"Combined symmetry failed: {e_orig} != {e_both}"
    
    print("  PASSED")


def test_known_optima():
    """Test known optimal values for small N (brute force verification)."""
    print("Test: Known Optima")
    
    known_optima = {3: 1, 4: 2}
    
    for N, expected in known_optima.items():
        min_energy = float('inf')
        for i in range(2**N):
            bitstring = format(i, f'0{N}b')
            seq = bitstring_to_sequence(bitstring)
            energy = compute_energy(seq)
            min_energy = min(min_energy, energy)
        
        assert min_energy == expected, f"N={N}: got {min_energy}, expected {expected}"
        print(f"  N={N}: brute-force min={min_energy}, expected={expected}")
    
    print("  PASSED")


def test_bitstring_conversion():
    """Test bitstring <-> sequence conversion correctness."""
    print("Test: Bitstring Conversion")
    
    test_cases = [
        ('00000', [1, 1, 1, 1, 1]),
        ('11111', [-1, -1, -1, -1, -1]),
        ('01010', [1, -1, 1, -1, 1]),
        ('10101', [-1, 1, -1, 1, -1]),
    ]
    
    for bs, expected_seq in test_cases:
        result = bitstring_to_sequence(bs)
        roundtrip = sequence_to_bitstring(result)
        
        assert result == expected_seq, f"{bs} -> {result}, expected {expected_seq}"
        assert roundtrip == bs, f"Roundtrip failed: {bs} -> {result} -> {roundtrip}"
    
    print("  PASSED")


def test_canonicalization_idempotence():
    """Test that canonicalization is idempotent."""
    print("Test: Canonicalization Idempotence")
    
    for _ in range(20):
        seq = random_sequence(8)
        canon1, bs1 = canonicalize_sequence(seq)
        canon2, bs2 = canonicalize_sequence(canon1)
        
        assert bs1 == bs2, f"Not idempotent: {bs1} != {bs2}"
    
    print("  PASSED")


def test_canonicalization_equivalence():
    """Test that all symmetric variants map to the same canonical form."""
    print("Test: Canonicalization Equivalence")
    
    for _ in range(20):
        seq = random_sequence(7)
        variants = [
            seq,
            [-s for s in seq],
            seq[::-1],
            [-s for s in seq[::-1]]
        ]
        
        canonicals = [canonicalize_sequence(v)[1] for v in variants]
        
        assert len(set(canonicals)) == 1, \
            f"Variants map to different canonicals: {canonicals}"
    
    print("  PASSED")


def test_hamiltonian_encoding():
    """Test that weighted Hamiltonian matches classical compute_energy."""
    print("Test: Hamiltonian Encoding")
    
    for N in [5, 8, 10, 12]:
        G2, G4, const = get_weighted_interactions(N)
        
        for _ in range(20):
            seq = random_sequence(N)
            classical = compute_energy(seq)
            hamiltonian = compute_energy_from_hamiltonian(seq, G2, G4, const)
            
            assert classical == hamiltonian, \
                f"N={N}: classical={classical}, hamiltonian={hamiltonian}"
        
        print(f"  N={N}: PASSED")


def test_cvar_computation():
    """Test CVaR computation with known values."""
    print("Test: CVaR Computation")
    
    energies = [10, 20, 30, 40, 50]
    counts = [1, 1, 1, 1, 1]
    
    cvar_20 = compute_cvar(energies, counts, alpha=0.2)
    cvar_40 = compute_cvar(energies, counts, alpha=0.4)
    
    assert abs(cvar_20 - 10.0) < 0.01, f"CVaR_0.2={cvar_20}, expected 10"
    assert abs(cvar_40 - 15.0) < 0.01, f"CVaR_0.4={cvar_40}, expected 15"
    
    print(f"  CVaR_0.2={cvar_20}, CVaR_0.4={cvar_40}")
    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 50)
    print("LABS QAOA-MTS Unit Tests")
    print("=" * 50)
    
    test_symmetry_invariance()
    test_known_optima()
    test_bitstring_conversion()
    test_canonicalization_idempotence()
    test_canonicalization_equivalence()
    test_hamiltonian_encoding()
    test_cvar_computation()
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
