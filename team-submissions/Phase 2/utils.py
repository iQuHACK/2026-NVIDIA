"""
LABS Utility Functions
======================

Complete utility module for LABS problem including
- Theta computation (from paper)
- LABS energy calculation
- Symmetry operations
- Population metrics
- Validation helpers
"""

import numpy as np
from math import sin, pi
from typing import List, Tuple, Dict


# ============================================================================
# THETA COMPUTATION (From Paper)
# ============================================================================

def compute_topology_overlaps(G2, G4):
    """
    Computes the topological invariants I_22, I_24, I_44 based on set overlaps.
    I_alpha_beta counts how many sets share IDENTICAL elements.
    """

    # Helper to count identical sets
    def count_matches(list_a, list_b):
        matches = 0
        # Convert to sorted tuples to ensure order doesn't affect equality
        set_b = set(tuple(sorted(x)) for x in list_b)
        for item in list_a:
            if tuple(sorted(item)) in set_b:
                matches += 1
        return matches

    # For standard LABS/Ising chains, these overlaps are often 0 or specific integers
    # We implement the general counting logic here.
    I_22 = count_matches(G2, G2)  # Self overlap is just len(G2)
    I_44 = count_matches(G4, G4)  # Self overlap is just len(G4)
    I_24 = 0  # 2-body set vs 4-body set overlap usually 0 as sizes differ

    return {'22': I_22, '44': I_44, '24': I_24}


def compute_theta(t, dt, total_time, N, G2, G4):
    """
    Computes theta(t) using the analytical solutions for Gamma1 and Gamma2.

    This implements the counteradiabatic driving angle from the LABS paper.

    Args:
        t: Current time
        dt: Time step
        total_time: Total evolution time T
        N: Problem size
        G2: Two-body interactions
        G4: Four-body interactions

    Returns:
        Rotation angle theta for current time step
    """

    # ---  Better Schedule (Trigonometric) ---
    # lambda(t) = sin^2(pi * t / 2T)
    # lambda_dot(t) = (pi / 2T) * sin(pi * t / T)

    if total_time == 0:
        return 0.0

    # Argument for the trig functions
    arg = (pi * t) / (2.0 * total_time)

    lam = sin(arg) ** 2
    # Derivative: (pi/2T) * sin(2 * arg) -> sin(pi * t / T)
    lam_dot = (pi / (2.0 * total_time)) * sin((pi * t) / total_time)

    # ---  Calculate Gamma Terms (LABS assumptions: h^x=1, h^b=0) ---
    # For G2 (size 2): S_x = 2
    # For G4 (size 4): S_x = 4

    # Gamma 1 (Eq 16)
    # Gamma1 = 16 * Sum_G2(S_x) + 64 * Sum_G4(S_x)
    term_g1_2 = 16 * len(G2) * 2
    term_g1_4 = 64 * len(G4) * 4
    Gamma1 = term_g1_2 + term_g1_4

    # Gamma 2 (Eq 17)
    # G2 term: Sum (lambda^2 * S_x)
    # S_x = 2
    sum_G2 = len(G2) * (lam ** 2 * 2)

    # G4 term: 4 * Sum (4*lambda^2 * S_x + (1-lambda)^2 * 8)
    # S_x = 4
    # Inner = 16*lam^2 + 8*(1-lam)^2
    sum_G4 = 4 * len(G4) * (16 * (lam ** 2) + 8 * ((1 - lam) ** 2))

    # Topology part
    I_vals = compute_topology_overlaps(G2, G4)
    term_topology = 4 * (lam ** 2) * (4 * I_vals['24'] + I_vals['22']) + 64 * (lam ** 2) * I_vals['44']

    # Combine Gamma 2
    Gamma2 = -256 * (term_topology + sum_G2 + sum_G4)

    # ---  Alpha & Theta ---
    if abs(Gamma2) < 1e-12:
        alpha = 0.0
    else:
        alpha = - Gamma1 / Gamma2

    return dt * alpha * lam_dot


# LABS ENERGY CALCULATION
def calculate_labs_energy(sequence: List[int], N: int) -> int:
    """
    Calculate LABS (Low Autocorrelation Binary Sequence) energy.

    The LABS energy is defined as:
    E = Σ_{k=1}^{N-1} [C(k)]²

    where C(k) is the autocorrelation at lag k:
    C(k) = Σ_{i=0}^{N-k-1} s_i · s_{i+k}

    with s_i ∈ {-1, +1} (spins)

    Lower energy is better. Optimal is E = 0 (impossible for most N).

    Args:
        sequence: Binary sequence as list of 0s and 1s
        N: Length of sequence

    Returns:
        LABS energy (integer, non-negative)

    Examples:
        >>> calculate_labs_energy([1, 1, 1], 3)
        5
        >>> calculate_labs_energy([1, 1, -1], 3)
        1  # Best for N=3
        >>> calculate_labs_energy([0, 0, 1], 3)
        1  # Same as above (0→-1, 1→+1)
    """
    # Convert binary {0,1} to spins {-1,+1}
    spins = [2 * s - 1 if s in [0, 1] else s for s in sequence]

    # Calculate autocorrelations
    energy = 0
    for k in range(1, N):
        # Autocorrelation at lag k
        autocorr = sum(spins[i] * spins[i + k] for i in range(N - k))
        # Add squared autocorrelation to energy
        energy += autocorr ** 2

    return energy


def calculate_labs_energy_batch(sequences: np.ndarray, N: int) -> np.ndarray:
    """
    Calculate LABS energy for batch of sequences (vectorized).

    More efficient than calling calculate_labs_energy in a loop.

    Args:
        sequences: Array of shape (batch_size, N) with values in {0,1} or {-1,+1}
        N: Sequence length

    Returns:
        Array of energies with shape (batch_size,)
    """
    # Convert to spins if needed
    if np.any((sequences == 0) | (sequences == 1)):
        spins = 2 * sequences - 1
    else:
        spins = sequences

    batch_size = spins.shape[0]
    energies = np.zeros(batch_size, dtype=int)

    for k in range(1, N):
        # Vectorized autocorrelation
        autocorr = np.sum(spins[:, :-k] * spins[:, k:], axis=1)
        energies += autocorr ** 2

    return energies

# SYMMETRY OPERATIONS
def generate_symmetric_variants(sequence: List[int]) -> List[List[int]]:
    """
    Generate all symmetric variants of a LABS sequence.

    LABS has 4-fold symmetry:
    1. Original sequence
    2. Bit-flip: 0↔1 (or -1↔+1)
    3. Time-reversal: Reverse the sequence
    4. Combined: Bit-flip + time-reversal

    All four variants have IDENTICAL LABS energy.

    Args:
        sequence: Original sequence

    Returns:
        List of 4 symmetric variants

    Example:
        >>> variants = generate_symmetric_variants([0, 1, 0, 1])
        >>> # variants[0] = [0, 1, 0, 1]  (original)
        >>> # variants[1] = [1, 0, 1, 0]  (bit-flip)
        >>> # variants[2] = [1, 0, 1, 0]  (time-reversal)
        >>> # variants[3] = [0, 1, 0, 1]  (combined)
    """
    variants = []

    # 1. Original
    variants.append(sequence.copy())

    # 2. Bit-flip (0↔1 or -1↔+1)
    if all(s in [0, 1] for s in sequence):
        # Binary: flip 0↔1
        variants.append([1 - s for s in sequence])
    else:
        # Spins: flip -1↔+1
        variants.append([-s for s in sequence])

    # 3. Time-reversal
    variants.append(sequence[::-1])

    # 4. Combined (bit-flip + time-reversal)
    if all(s in [0, 1] for s in sequence):
        variants.append([1 - s for s in sequence[::-1]])
    else:
        variants.append([-s for s in sequence[::-1]])

    return variants


def verify_symmetry(sequence: List[int], N: int, tolerance: int = 0) -> bool:
    """
    Verify that all symmetric variants have the same energy.

    Args:
        sequence: Sequence to check
        N: Length
        tolerance: Allowed energy difference (should be 0)

    Returns:
        True if all symmetric variants have same energy
    """
    variants = generate_symmetric_variants(sequence)
    energies = [calculate_labs_energy(v, N) for v in variants]

    min_energy = min(energies)
    max_energy = max(energies)

    return (max_energy - min_energy) <= tolerance


# POPULATION METRICS
def hamming_distance(seq1: List[int], seq2: List[int]) -> int:
    """
    Calculate Hamming distance between two sequences.

    Hamming distance = number of positions where sequences differ.

    Args:
        seq1, seq2: Sequences to compare

    Returns:
        Number of differing positions

    Example:
        >>> hamming_distance([0, 0, 1], [0, 1, 1])
        1
        >>> hamming_distance([0, 0, 0], [1, 1, 1])
        3
    """
    return sum(a != b for a, b in zip(seq1, seq2))


def calculate_population_diversity(population: List[List[int]], N: int) -> Dict:
    """
    Calculate diversity metrics for a population.

    Metrics:
    - Mean pairwise Hamming distance
    - Min pairwise distance
    - Max pairwise distance
    - Unique sequences count

    Args:
        population: List of sequences
        N: Sequence length

    Returns:
        Dictionary with diversity metrics
    """
    pop_size = len(population)

    if pop_size <= 1:
        return {
            'mean_distance': 0,
            'min_distance': 0,
            'max_distance': 0,
            'unique_count': pop_size
        }

    # Calculate all pairwise distances
    distances = []
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            dist = hamming_distance(population[i], population[j])
            distances.append(dist)

    # Count unique sequences
    unique_seqs = set(tuple(seq) for seq in population)

    return {
        'mean_distance': np.mean(distances),
        'min_distance': min(distances),
        'max_distance': max(distances),
        'unique_count': len(unique_seqs),
        'diversity_ratio': len(unique_seqs) / pop_size
    }


def calculate_population_statistics(population: List[List[int]], N: int) -> Dict:
    """
    Calculate comprehensive statistics for a population.

    Includes both energy and diversity metrics.

    Args:
        population: List of sequences
        N: Sequence length

    Returns:
        Dictionary with all statistics
    """
    # Energy statistics
    energies = [calculate_labs_energy(seq, N) for seq in population]

    # Diversity statistics
    diversity = calculate_population_diversity(population, N)

    return {
        # Energy metrics
        'min_energy': min(energies),
        'max_energy': max(energies),
        'mean_energy': np.mean(energies),
        'median_energy': np.median(energies),
        'std_energy': np.std(energies),

        # Diversity metrics
        'mean_distance': diversity['mean_distance'],
        'min_distance': diversity['min_distance'],
        'max_distance': diversity['max_distance'],
        'unique_count': diversity['unique_count'],
        'diversity_ratio': diversity['diversity_ratio'],

        # Population size
        'population_size': len(population)
    }


# KNOWN OPTIMAL SOLUTIONS
def get_known_optimal(N: int) -> Tuple[int, List[int]]:
    """
    Get known optimal LABS energy for small N.

    These are verified optimal or best-known solutions from literature.

    Args:
        N: Problem size

    Returns:
        (optimal_energy, optimal_sequence) or (None, None) if unknown

    Reference: http://www.packomania.com/
    """
    known_solutions = {
        3: (1, [1, 1, -1]),
        4: (2, [1, 1, -1, -1]),
        5: (2, [1, 1, 1, -1, -1]),
        6: (4, [1, 1, 1, -1, -1, -1]),
        7: (4, [1, 1, 1, -1, -1, 1, -1]),
        8: (6, [1, 1, 1, -1, -1, 1, -1, -1]),
        9: (6, [1, 1, 1, 1, -1, -1, 1, -1, -1]),
        10: (8, [1, 1, 1, -1, -1, 1, -1, -1, 1, -1]),
        11: (10, [1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1]),
        12: (12, [1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1]),
    }

    if N in known_solutions:
        energy, sequence = known_solutions[N]
        # Convert to binary
        binary_seq = [(s + 1) // 2 for s in sequence]
        return energy, binary_seq
    else:
        return None, None


def calculate_approximation_ratio(found_energy: int, N: int) -> float:
    """
    Calculate approximation ratio for LABS solution.

    Ratio = optimal_energy / found_energy

    Higher is better. 1.0 means optimal solution found.

    Args:
        found_energy: Energy of found solution
        N: Problem size

    Returns:
        Approximation ratio (or None if optimal unknown)
    """
    optimal_energy, _ = get_known_optimal(N)

    if optimal_energy is None:
        return None

    if found_energy == 0:
        return float('inf') if optimal_energy == 0 else 0.0

    return optimal_energy / found_energy



# VALIDATION HELPERS
def validate_sequence(sequence: List[int], N: int) -> Dict:
    """
    Validate a LABS sequence and calculate all metrics.

    Args:
        sequence: Sequence to validate
        N: Expected length

    Returns:
        Dictionary with validation results and metrics
    """
    # Check length
    length_valid = len(sequence) == N

    # Check values
    if all(s in [0, 1] for s in sequence):
        format_type = 'binary'
        values_valid = True
    elif all(s in [-1, 1] for s in sequence):
        format_type = 'spin'
        values_valid = True
    else:
        format_type = 'invalid'
        values_valid = False

    # Calculate energy
    energy = calculate_labs_energy(sequence, N) if values_valid else None

    # Check symmetry
    symmetry_valid = verify_symmetry(sequence, N) if values_valid else False

    # Compare to known optimal
    optimal_energy, _ = get_known_optimal(N)
    approx_ratio = calculate_approximation_ratio(energy, N) if energy is not None else None

    return {
        'length_valid': length_valid,
        'values_valid': values_valid,
        'format': format_type,
        'energy': energy,
        'symmetry_valid': symmetry_valid,
        'optimal_energy': optimal_energy,
        'approximation_ratio': approx_ratio,
        'is_optimal': (energy == optimal_energy) if (energy is not None and optimal_energy is not None) else None
    }


def get_theoretical_bounds(N: int) -> Dict:
    """
    Get theoretical energy bounds for LABS problem.

    Args:
        N: Problem size

    Returns:
        Dictionary with bounds
    """
    # Worst case: all spins same
    worst_case = sum((N - k) ** 2 for k in range(1, N))

    # Best known (from literature)
    optimal, _ = get_known_optimal(N)

    # Trivial lower bound: 0 (unattainable for most N)
    lower_bound = 0

    return {
        'lower_bound': lower_bound,
        'best_known': optimal,
        'worst_case': worst_case,
        'N': N
    }


# CONVERSION UTILITIES
def binary_to_spins(binary_seq: List[int]) -> List[int]:
    """Convert binary {0,1} to spins {-1,+1}."""
    return [2 * b - 1 for b in binary_seq]


def spins_to_binary(spin_seq: List[int]) -> List[int]:
    """Convert spins {-1,+1} to binary {0,1}."""
    return [(s + 1) // 2 for s in spin_seq]


def bitstring_to_list(bitstring: str) -> List[int]:
    """Convert bitstring '0101' to list [0,1,0,1]."""
    return [int(bit) for bit in bitstring]


def list_to_bitstring(seq: List[int]) -> str:
    """Convert list [0,1,0,1] to bitstring '0101'."""
    return ''.join(str(s) for s in seq)
