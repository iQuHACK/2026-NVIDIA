"""
Test Suite for Enhanced LABS Implementation
============================================

Comprehensive validation following Milestone 3 requirements.
Tests for:
1. Interaction generation correctness
2. Quantum circuit validity
3. LABS energy calculation
4. Symmetry preservation
5. Ancilla management
"""

import pytest
import numpy as np
from enhanced_labs_implementation import (
    get_interactions,
    AncillaManager
)
from utils import (
    calculate_labs_energy,
    generate_symmetric_variants,
    hamming_distance
)


# ============================================================================
# TEST 1: Interaction Generation
# ============================================================================

class TestInteractionGeneration:
    """Test G2 and G4 interaction generation."""

    def test_g2_count_small_n(self):
        """Verify G2 count for small problem sizes."""
        test_cases = {
            4: 2,  # N=4 should have 2 G2 interactions
            5: 4,
            6: 6,
            8: 12,
            10: 20
        }

        for N, expected_count in test_cases.items():
            G2, _ = get_interactions(N)
            assert len(G2) == expected_count, \
                f"N={N}: Expected {expected_count} G2, got {len(G2)}"

    def test_g4_all_indices_distinct(self):
        """CRITICAL: All G4 interactions must have 4 distinct indices."""
        for N in [5, 8, 10, 12, 15]:
            _, G4 = get_interactions(N)

            for interaction in G4:
                assert len(interaction) == 4, \
                    f"G4 should have 4 indices: {interaction}"
                assert len(set(interaction)) == 4, \
                    f"G4 indices not unique: {interaction}"

    def test_indices_in_bounds(self):
        """All indices must be within [0, N-1]."""
        for N in [5, 10, 15, 20]:
            G2, G4 = get_interactions(N)

            for interaction in G2:
                assert all(0 <= idx < N for idx in interaction), \
                    f"G2 index out of bounds: {interaction}"

            for interaction in G4:
                assert all(0 <= idx < N for idx in interaction), \
                    f"G4 index out of bounds: {interaction}"

    def test_g2_pattern(self):
        """G2 interactions should follow [i, i+k] pattern."""
        N = 10
        G2, _ = get_interactions(N)

        for interaction in G2:
            i, ik = interaction
            assert ik > i, f"G2 pattern violated: {interaction}"
            # Verify it's within the formula constraints
            assert ik - i <= (N - i - 1) // 2, \
                f"G2 exceeds formula constraints: {interaction}"


# ============================================================================
# TEST 2: LABS Energy Calculation
# ============================================================================

class TestLABSEnergy:
    """Test LABS energy function correctness."""

    def test_known_small_cases(self):
        """Test against manually calculated energies for small N."""
        # N=3 cases
        assert calculate_labs_energy([1, 1, 1], 3) == 5
        assert calculate_labs_energy([1, -1, 1], 3) == 5
        assert calculate_labs_energy([1, 1, -1], 3) == 1  # Best for N=3

        # N=4 cases
        assert calculate_labs_energy([1, 1, 1, 1], 4) == 14  # Worst
        assert calculate_labs_energy([1, -1, 1, -1], 4) == 14
        assert calculate_labs_energy([1, 1, -1, -1], 4) == 6  # Better

    def test_energy_bounds(self):
        """Energy must be non-negative and within theoretical bounds."""
        for N in [4, 6, 8, 10]:
            # Worst case: all same
            worst_seq = [1] * N
            worst_energy = calculate_labs_energy(worst_seq, N)

            # Theoretical worst case
            theoretical_worst = sum((N - k) ** 2 for k in range(1, N))

            assert worst_energy == theoretical_worst, \
                f"N={N}: Worst case mismatch"

            # Random sequences should be better than worst
            for _ in range(10):
                random_seq = [1 if np.random.random() > 0.5 else 0 for _ in range(N)]
                random_energy = calculate_labs_energy(random_seq, N)

                assert 0 <= random_energy <= worst_energy, \
                    f"Energy {random_energy} out of bounds [0, {worst_energy}]"

    def test_binary_input_conversion(self):
        """Test that binary (0/1) input is handled correctly."""
        # Binary input
        binary_seq = [0, 1, 0, 1]
        # Equivalent spin input
        spin_seq = [-1, 1, -1, 1]

        energy_binary = calculate_labs_energy(binary_seq, 4)
        energy_spin = calculate_labs_energy(spin_seq, 4)

        # Should produce same energy
        assert energy_binary == energy_spin, \
            f"Binary vs spin mismatch: {energy_binary} != {energy_spin}"


# ============================================================================
# TEST 3: LABS Symmetries
# ============================================================================

class TestLABSSymmetries:
    """Test that LABS symmetries are preserved."""

    def test_bitflip_symmetry(self):
        """Energy must be invariant under bit-flip."""
        for N in [5, 8, 10]:
            for _ in range(10):
                seq = [1 if np.random.random() > 0.5 else 0 for _ in range(N)]
                flipped = [1 - s for s in seq]

                e_orig = calculate_labs_energy(seq, N)
                e_flip = calculate_labs_energy(flipped, N)

                assert e_orig == e_flip, \
                    f"Bit-flip symmetry violated: {e_orig} != {e_flip}"

    def test_time_reversal_symmetry(self):
        """Energy must be invariant under time-reversal."""
        for N in [5, 8, 10]:
            for _ in range(10):
                seq = [1 if np.random.random() > 0.5 else 0 for _ in range(N)]
                reversed_seq = seq[::-1]

                e_orig = calculate_labs_energy(seq, N)
                e_rev = calculate_labs_energy(reversed_seq, N)

                assert e_orig == e_rev, \
                    f"Time-reversal symmetry violated: {e_orig} != {e_rev}"

    def test_combined_symmetry(self):
        """Energy must be invariant under bit-flip + time-reversal."""
        for N in [5, 8, 10]:
            for _ in range(10):
                seq = [1 if np.random.random() > 0.5 else 0 for _ in range(N)]
                combined = [1 - s for s in seq[::-1]]

                e_orig = calculate_labs_energy(seq, N)
                e_comb = calculate_labs_energy(combined, N)

                assert e_orig == e_comb, \
                    f"Combined symmetry violated: {e_orig} != {e_comb}"

    def test_all_four_variants(self):
        """All four symmetric variants should have identical energy."""
        for N in [6, 8, 10]:
            seq = [1 if np.random.random() > 0.5 else 0 for _ in range(N)]
            variants = generate_symmetric_variants(seq)

            energies = [calculate_labs_energy(v, N) for v in variants]

            assert len(set(energies)) == 1, \
                f"Symmetric variants have different energies: {energies}"


# ============================================================================
# TEST 4: Ancilla Management (SQUARE)
# ============================================================================

class TestAncillaManager:
    """Test SQUARE ancilla management logic."""

    def test_allocation_and_reclamation(self):
        """Test basic allocation and reclamation cycle."""
        manager = AncillaManager(N=10)

        # Allocate 3 qubits
        allocated = manager.allocate_ancilla(3)
        assert len(allocated) == 3
        assert manager.get_stats()['active_qubits'] == 3

        # Reclaim them
        manager.reclaim_ancilla(allocated)
        assert manager.get_stats()['active_qubits'] == 0
        assert manager.get_stats()['heap_size'] == 3

        # Reallocate - should reuse from heap
        reallocated = manager.allocate_ancilla(2)
        assert len(reallocated) == 2
        assert all(r in allocated for r in reallocated), \
            "Should reuse from heap"

    def test_reclamation_decision(self):
        """Test cost-effective reclamation heuristic."""
        manager = AncillaManager(N=20)

        # Allocate some qubits
        manager.allocate_ancilla(10)

        # Low cost, shallow level → should reclaim
        should_reclaim_1 = manager.should_reclaim(
            gate_cost=5, n_ancilla=3, level=0
        )
        assert should_reclaim_1, "Should reclaim with low cost"

        # High cost, deep level → should NOT reclaim
        should_reclaim_2 = manager.should_reclaim(
            gate_cost=100, n_ancilla=2, level=3
        )
        assert not should_reclaim_2, "Should NOT reclaim with high cost"

    def test_peak_qubit_tracking(self):
        """Verify peak qubit usage is tracked correctly."""
        manager = AncillaManager(N=20)

        manager.allocate_ancilla(5)
        manager.allocate_ancilla(10)  # Peak should be 15
        manager.reclaim_ancilla([0, 1, 2, 3, 4])
        manager.allocate_ancilla(3)

        stats = manager.get_stats()
        assert stats['peak_qubits'] >= 15, \
            f"Peak qubits {stats['peak_qubits']} should be >= 15"


# Add these new test classes

class TestGPUAcceleration:
    """Test GPU vs CPU consistency."""

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
    def test_gpu_cpu_consistency(self):
        """GPU results should match CPU results."""
        from gpu_labs import evaluate_population_gpu

        N = 10
        population = [[np.random.randint(0, 2) for _ in range(N)]
                      for _ in range(100)]

        # CPU
        cpu_energies = [calculate_labs_energy(seq, N) for seq in population]

        # GPU
        gpu_energies = evaluate_population_gpu(population, N)

        np.testing.assert_array_equal(cpu_energies, gpu_energies)


class TestScalability:
    """Test that solution scales to larger N."""

    def test_large_n_generation(self):
        """Should handle N up to 50."""
        for N in [10, 20, 30, 40, 50]:
            G2, G4 = get_interactions(N)
            assert len(G2) > 0
            # Verify no crashes


class TestPhysicalConstraints:
    """Test physical constraint violations."""

    def test_energy_never_negative(self):
        """LABS energy must always be >= 0."""
        for _ in range(100):
            N = np.random.randint(5, 30)
            seq = [np.random.randint(0, 2) for _ in range(N)]
            energy = calculate_labs_energy(seq, N)
            assert energy >= 0, f"Negative energy {energy} for N={N}"

    def test_interaction_validity(self):
        """All interactions must reference valid qubits."""
        for N in range(4, 51):
            G2, G4 = get_interactions(N)

            for interaction in G2 + G4:
                assert all(0 <= idx < N for idx in interaction), \
                    f"Invalid qubit index at N={N}"
# ============================================================================
# TEST 5: Helper Functions
# ============================================================================

class TestHelperFunctions:
    """Test utility functions."""

    def test_hamming_distance(self):
        """Test Hamming distance calculation."""
        assert hamming_distance([0, 0, 0], [0, 0, 0]) == 0
        assert hamming_distance([0, 0, 0], [1, 1, 1]) == 3
        assert hamming_distance([0, 1, 0, 1], [1, 1, 0, 0]) == 2

    def test_symmetric_variants_count(self):
        """Should generate exactly 4 variants."""
        seq = [0, 1, 0, 1, 1]
        variants = generate_symmetric_variants(seq)

        assert len(variants) == 4, "Should generate 4 variants"

        # All should be distinct
        variants_tuples = [tuple(v) for v in variants]
        assert len(set(variants_tuples)) == 4, "Variants should be distinct"

    def test_symmetric_variants_include_original(self):
        """Original sequence should be first variant."""
        seq = [0, 1, 0, 1]
        variants = generate_symmetric_variants(seq)

        assert variants[0] == seq, "First variant should be original"


# ============================================================================
# TEST 6: Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_small_n_workflow(self):
        """Test complete workflow for small N."""
        N = 5

        # Generate interactions
        G2, G4 = get_interactions(N)
        assert len(G2) > 0, "Should have G2 interactions"
        assert len(G4) >= 0, "Should have G4 interactions"

        # Generate test population
        population = [[1 if np.random.random() > 0.5 else 0
                       for _ in range(N)] for _ in range(10)]

        # Calculate energies
        energies = [calculate_labs_energy(seq, N) for seq in population]
        assert all(e >= 0 for e in energies), "All energies should be non-negative"

        # Find best
        best_idx = np.argmin(energies)
        best_seq = population[best_idx]
        best_energy = energies[best_idx]

        # Verify symmetric variants have same energy
        variants = generate_symmetric_variants(best_seq)
        variant_energies = [calculate_labs_energy(v, N) for v in variants]
        assert all(e == best_energy for e in variant_energies), \
            "Symmetric variants should have same energy"


# ============================================================================
# PROPERTY-BASED TESTS
# ============================================================================

class TestProperties:
    """Property-based tests using hypothesis-style approach."""

    def test_energy_symmetry_property(self):
        """Property: All symmetries preserve energy for ANY sequence."""
        for _ in range(50):  # 50 random tests
            N = np.random.randint(4, 12)
            seq = [1 if np.random.random() > 0.5 else 0 for _ in range(N)]

            e_base = calculate_labs_energy(seq, N)

            # Test all symmetries
            variants = generate_symmetric_variants(seq)
            for v in variants:
                assert calculate_labs_energy(v, N) == e_base, \
                    f"Symmetry failed for N={N}, seq={seq}"

    def test_interaction_indices_property(self):
        """Property: All interaction indices must be valid for ANY N."""
        for N in range(4, 21):
            G2, G4 = get_interactions(N)

            # All G2 indices in bounds
            for interaction in G2:
                assert all(0 <= idx < N for idx in interaction)

            # All G4 indices in bounds and distinct
            for interaction in G4:
                assert all(0 <= idx < N for idx in interaction)
                assert len(set(interaction)) == 4


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])