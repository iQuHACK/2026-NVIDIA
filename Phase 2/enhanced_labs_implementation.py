"""
Enhanced LABS Quantum-Enhanced Optimization
============================================

This implementation extends the baseline counteradiabatic approach with:
1. SQUARE ancilla reuse (strategic uncomputation)
2. Dynamic circuit optimization (measurement and reset)
3. Improved qubit allocation strategies

Based on:
- arXiv:2511.04553 (Counteradiabatic LABS)
- arXiv:2004.08539 (SQUARE)
- arXiv:2511.22712 (Dynamic Qubit Reuse)
"""

import cudaq
import numpy as np
import random
from math import pi
from typing import List, Tuple, Dict
import utils


# ============================================================================
# STEP 1: Enhanced Interaction Generation with Duplicate Filtering
# ============================================================================

def get_interactions(N: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Generate G2 and G4 interactions with proper duplicate filtering.

    Args:
        N: Problem size (sequence length)

    Returns:
        G2: Two-body interactions [[i, i+k], ...]
        G4: Four-body interactions [[i, i+j, i+k, i+k+j], ...] with distinct indices
    """
    G2 = []
    G4 = []

    # Two-body interactions: Y_i ⊗ Z_{i+k}
    for i in range(1, N - 1):
        for k in range(1, (N - i) // 2 + 1):
            idx_i = i - 1
            idx_i_k = i + k - 1
            G2.append([idx_i, idx_i_k])

    # Four-body interactions: Y_i ⊗ Z_{i+j} ⊗ Z_{i+k} ⊗ Z_{i+k+j}
    for i in range(1, N - 2):
        for j in range(1, N - i):
            for k in range(i + 1, N - i):
                idx = [i - 1, i + j - 1, i + k - 1, i + k + j - 1]
                # Only add if all indices are valid AND distinct
                if idx[3] < N and len(set(idx)) == 4:
                    G4.append(idx)

    return G2, G4


# ============================================================================
# STEP 2: Ancilla Management with SQUARE Principles
# ============================================================================

class AncillaManager:
    """
    Manages qubit allocation and reclamation using SQUARE heuristics.
    """

    def __init__(self, N: int):
        self.N = N
        self.ancilla_heap: List[int] = []
        self.active_qubits: set = set()
        self.allocation_history: List[Tuple] = []

    def allocate_ancilla(self, n_qubits: int) -> List[int]:
        """Allocate ancilla qubits with locality awareness."""
        allocated = []

        while len(allocated) < n_qubits and self.ancilla_heap:
            qubit_id = self.ancilla_heap.pop()
            allocated.append(qubit_id)
            self.active_qubits.add(qubit_id)

        while len(allocated) < n_qubits:
            new_id = max(self.active_qubits) + 1 if self.active_qubits else 0
            allocated.append(new_id)
            self.active_qubits.add(new_id)

        self.allocation_history.append(('allocate', allocated))
        return allocated

    def should_reclaim(self, gate_cost: int, n_ancilla: int, level: int) -> bool:
        """Cost-Effective Reclamation decision."""
        Nactive = len(self.active_qubits)
        Guncomp = gate_cost

        C1 = Nactive * Guncomp * (2 ** level)

        Gparent = 2 * Guncomp
        area_expansion = np.sqrt((Nactive + n_ancilla) / max(Nactive, 1))
        C0 = n_ancilla * Gparent * area_expansion

        return C1 <= C0

    def reclaim_ancilla(self, qubit_ids: List[int]):
        """Reclaim ancilla qubits and add to heap for reuse."""
        for qid in qubit_ids:
            if qid in self.active_qubits:
                self.active_qubits.remove(qid)
                self.ancilla_heap.append(qid)

        self.allocation_history.append(('reclaim', qubit_ids))

    def get_stats(self) -> Dict:
        """Get statistics about ancilla usage."""
        peak = len(self.active_qubits)
        for op, alloc in self.allocation_history:
            if op == 'allocate':
                peak = max(peak, len(alloc))

        return {
            'active_qubits': len(self.active_qubits),
            'heap_size': len(self.ancilla_heap),
            'total_allocated': len(self.active_qubits) + len(self.ancilla_heap),
            'peak_qubits': peak
        }

    def get_reuse_efficiency(self) -> Dict:
        """Calculate metrics for ancilla reuse efficiency."""
        total_allocations = sum(1 for op, _ in self.allocation_history if op == 'allocate')
        total_reclaims = sum(1 for op, _ in self.allocation_history if op == 'reclaim')

        if total_allocations == 0:
            reuse_ratio = 0.0
        else:
            reuse_ratio = len(self.ancilla_heap) / total_allocations

        return {
            'total_allocations': total_allocations,
            'total_reclaims': total_reclaims,
            'reuse_ratio': reuse_ratio,
            'peak_active': self.get_stats()['peak_qubits'],
            'current_heap_size': len(self.ancilla_heap)
        }


# ============================================================================
# STEP 3: Enhanced Trotterized Circuit with Flattened Interactions
# ============================================================================

@cudaq.kernel
def enhanced_trotterized_circuit(
        N: int,
        G2_flat: list[int],
        G2_count: int,
        G4_flat: list[int],
        G4_count: int,
        steps: int,
        thetas: list[float]
):
    """
    Enhanced counteradiabatic circuit with flattened interaction lists.
    """
    reg = cudaq.qvector(N)
    h(reg)

    for step in range(steps):
        theta = thetas[step]

        # Two-body interactions
        for g2_idx in range(G2_count):
            i = G2_flat[2 * g2_idx]
            k = G2_flat[2 * g2_idx + 1]
            angle = 4.0 * theta

            cx(reg[i], reg[k])
            ry(angle, reg[k])
            cx(reg[i], reg[k])

        # Four-body interactions
        for g4_idx in range(G4_count):
            i = G4_flat[4 * g4_idx]
            i1 = G4_flat[4 * g4_idx + 1]
            ik = G4_flat[4 * g4_idx + 2]
            iki = G4_flat[4 * g4_idx + 3]
            angle = 8.0 * theta

            cx(reg[i1], reg[iki])
            cx(reg[ik], reg[iki])
            cx(reg[i], reg[iki])
            ry(angle, reg[iki])
            cx(reg[i], reg[iki])
            cx(reg[ik], reg[iki])
            cx(reg[i1], reg[iki])


def flatten_interactions(G2: List[List[int]], G4: List[List[int]]) -> Tuple[List[int], int, List[int], int]:
    """Flatten nested interaction lists for CUDA-Q kernel compatibility."""
    G2_flat = []
    for interaction in G2:
        G2_flat.extend(interaction)

    G4_flat = []
    for interaction in G4:
        G4_flat.extend(interaction)

    return G2_flat, len(G2), G4_flat, len(G4)


def track_qubit_usage(N: int, G2: List[List[int]], G4: List[List[int]], n_steps: int) -> Dict:
    """Track which qubits are actively used at each step."""
    qubit_usage: Dict[int, List[int]] = {i: [] for i in range(N)}
    current_time = 0

    for _ in range(n_steps):
        for interaction in G2:
            i, k = interaction
            qubit_usage[i].append(current_time)
            qubit_usage[k].append(current_time)
            current_time += 1

        for interaction in G4:
            i, i1, ik, iki = interaction
            for q in [i, i1, ik, iki]:
                qubit_usage[q].append(current_time)
            current_time += 3

    qubit_active_time = {q: len(times) for q, times in qubit_usage.items()}
    total_active_time = sum(qubit_active_time.values())

    max_active = max(qubit_active_time.values()) if qubit_active_time else 0

    return {
        'qubit_usage': qubit_usage,
        'qubit_active_time': qubit_active_time,
        'total_active_time': total_active_time,
        'average_active_time': total_active_time / N if N > 0 else 0,
        'max_active_qubit': max_active
    }


def deduplicate_population(
    population: List[List[int]],
    strategy: str = 'hash'
) -> Tuple[List[List[int]], Dict]:
    """Remove duplicate sequences."""
    seen: set = set()
    unique_pop: List[List[int]] = []

    if strategy == 'hash':
        for seq in population:
            seq_tuple = tuple(seq)
            if seq_tuple not in seen:
                unique_pop.append(seq)
                seen.add(seq_tuple)

    elif strategy == 'symmetric':
        for seq in population:
            variants = utils.generate_symmetric_variants(seq)
            variant_tuples = [tuple(v) for v in variants]

            if not any(vt in seen for vt in variant_tuples):
                unique_pop.append(seq)
                seen.update(variant_tuples)
    else:
        unique_pop = list(population)

    stats = {
        'original_size': len(population),
        'deduplicated_size': len(unique_pop),
        'duplicates_removed': len(population) - len(unique_pop),
        'compression_ratio': len(unique_pop) / len(population) if population else 0
    }

    return unique_pop, stats


def mutate(sequence: List[int], p_mut: float = 0.1) -> List[int]:
    """Mutate a sequence with given probability per bit."""
    return [1 - s if random.random() < p_mut else s for s in sequence]


# ============================================================================
# STEP 4: Improved Quantum Population Sampling
# ============================================================================

def sample_quantum_population_enhanced(
        N: int,
        G2: List[List[int]],
        G4: List[List[int]],
        n_steps: int,
        dt: float,
        T: float,
        thetas: List[float],
        pop_size: int = 50,
        n_shots: int = 5000,
        strategy: str = 'hybrid',
        use_deduplication: bool = True
) -> Tuple[List[List[int]], Dict]:
    """
    Enhanced quantum population sampling with multiple strategies.
    """
    print(f"\n{'=' * 70}")
    print("ENHANCED QUANTUM SAMPLING")
    print(f"{'=' * 70}")
    print(f"Problem size: N={N}")
    print(f"Population size: {pop_size}")
    print(f"Quantum shots: {n_shots}")
    print(f"Strategy: {strategy}")
    print(f"Deduplication: {use_deduplication}")

    # Flatten interactions for CUDA-Q compatibility
    G2_flat, G2_count, G4_flat, G4_count = flatten_interactions(G2, G4)

    # Sample from quantum circuit
    result = cudaq.sample(
        enhanced_trotterized_circuit,
        N, G2_flat, G2_count, G4_flat, G4_count, n_steps, thetas,
        shots_count=n_shots
    )

    # Extract and analyze results
    counts = {state: result.count(state) for state in result}
    total_shots = sum(counts.values())

    print(f"\nQuantum sampling results:")
    print(f"  Unique states: {len(counts)}")
    print(f"  Total shots: {total_shots}")

    usage_stats = track_qubit_usage(N, G2, G4, n_steps)
    print(f"\nQubit usage statistics:")
    print(f"  Average active time: {usage_stats['average_active_time']:.2f}")
    print(f"  Max active qubit time: {usage_stats['max_active_qubit']}")

    # Convert to population based on strategy
    if strategy == 'amplitude':
        population = _amplitude_based_sampling(counts, N, pop_size)
    elif strategy == 'energy':
        population = _energy_based_sampling(counts, N, pop_size)
    elif strategy == 'diversity':
        population = _diversity_based_sampling(counts, N, pop_size)
    else:  # hybrid
        population = _hybrid_sampling(counts, N, pop_size)

    dedup_stats: Dict = {}
    if use_deduplication:
        population, dedup_stats = deduplicate_population(
            population,
            strategy='symmetric'
        )
        print(f"\nDeduplication results:")
        print(f"  Removed {dedup_stats['duplicates_removed']} duplicates")
        print(f"  Compression: {dedup_stats['compression_ratio']:.2%}")

        # Refill to desired size if needed
        existing = set(tuple(seq) for seq in population)
        while len(population) < pop_size:
            parent = random.choice(population[:max(1, len(population) // 2)])
            child = mutate(parent, p_mut=0.1)
            if tuple(child) not in existing:
                population.append(child)
                existing.add(tuple(child))

    # Calculate metrics
    energies = [utils.calculate_labs_energy(seq, N) for seq in population]

    max_active = usage_stats['max_active_qubit']
    qubit_efficiency = usage_stats['average_active_time'] / max_active if max_active > 0 else 0

    metrics = {
        'min_energy': min(energies),
        'max_energy': max(energies),
        'mean_energy': np.mean(energies),
        'std_energy': np.std(energies),
        'unique_states': len(counts),
        'population_size': len(population),
        'qubit_efficiency': qubit_efficiency,
        'deduplication_stats': dedup_stats if use_deduplication else None
    }

    print(f"\nPopulation metrics:")
    print(f"  Energy - Min: {metrics['min_energy']}, "
          f"Mean: {metrics['mean_energy']:.2f}, "
          f"Max: {metrics['max_energy']}")
    print(f"{'=' * 70}\n")

    return population, metrics


def _amplitude_based_sampling(counts: Dict, N: int, pop_size: int) -> List[List[int]]:
    """Sample weighted by quantum amplitude (measurement frequency)."""
    sorted_states = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    population = []

    for state, _ in sorted_states[:pop_size]:
        individual = [int(bit) for bit in state]
        if len(individual) < N:
            individual = [0] * (N - len(individual)) + individual
        population.append(individual)

    while len(population) < pop_size:
        parent = random.choice(population[:max(1, len(population) // 2)])
        child = mutate(parent, p_mut=0.1)
        if child not in population:
            population.append(child)

    return population[:pop_size]


def _energy_based_sampling(counts: Dict, N: int, pop_size: int) -> List[List[int]]:
    """Sample based on LABS energy (greedy approach)."""
    candidates = []

    for state in counts.keys():
        individual = [int(bit) for bit in state]
        if len(individual) < N:
            individual = [0] * (N - len(individual)) + individual
        energy = utils.calculate_labs_energy(individual, N)
        candidates.append((individual, energy))

    candidates.sort(key=lambda x: x[1])
    population = [seq for seq, _ in candidates[:pop_size]]

    while len(population) < pop_size:
        parent = random.choice(population[:max(1, len(population) // 4)])
        child = mutate(parent, p_mut=0.1)
        if child not in population:
            population.append(child)

    return population[:pop_size]


def _diversity_based_sampling(counts: Dict, N: int, pop_size: int) -> List[List[int]]:
    """Sample with diversity awareness."""
    candidates = []

    for state in counts.keys():
        individual = [int(bit) for bit in state]
        if len(individual) < N:
            individual = [0] * (N - len(individual)) + individual
        energy = utils.calculate_labs_energy(individual, N)
        candidates.append((individual, energy))

    candidates.sort(key=lambda x: x[1])

    population = [candidates[0][0]] if candidates else []

    for seq, energy in candidates[1:]:
        if len(population) >= pop_size:
            break

        min_dist = min(utils.hamming_distance(seq, existing) for existing in population)

        if min_dist > N // 4 or energy < candidates[0][1] * 1.1:
            population.append(seq)

    while len(population) < pop_size:
        parent = random.choice(population) if population else [0] * N
        child = mutate(parent, p_mut=0.15)
        if child not in population:
            population.append(child)

    return population[:pop_size]


def _hybrid_sampling(counts: Dict, N: int, pop_size: int) -> List[List[int]]:
    """Hybrid strategy combining amplitude, energy, and diversity."""
    candidates = []
    for state, count in counts.items():
        individual = [int(bit) for bit in state]
        if len(individual) < N:
            individual = [0] * (N - len(individual)) + individual
        energy = utils.calculate_labs_energy(individual, N)
        weight = np.sqrt(count)
        candidates.append((individual, energy, weight))

    candidates.sort(key=lambda x: x[1])
    cutoff = max(1, len(candidates) // 2)
    filtered = candidates[:cutoff]

    enriched = []
    top_k = max(1, len(filtered) // 4)

    for i, (seq, energy, _) in enumerate(filtered):
        if i < top_k:
            for variant in utils.generate_symmetric_variants(seq):
                variant_energy = utils.calculate_labs_energy(variant, N)
                enriched.append((variant, variant_energy))
        else:
            enriched.append((seq, energy))

    enriched.sort(key=lambda x: x[1])
    population = []
    seen: set = set()

    for seq, _ in enriched:
        if len(population) >= pop_size:
            break

        seq_tuple = tuple(seq)
        if seq_tuple not in seen:
            population.append(seq)
            seen.add(seq_tuple)

    while len(population) < pop_size:
        parent = random.choice(population[:max(1, len(population) // 2)]) if population else [0] * N
        child = mutate(parent, p_mut=0.1)
        if tuple(child) not in seen:
            population.append(child)
            seen.add(tuple(child))

    return population[:pop_size]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED LABS QUANTUM OPTIMIZATION")
    print("With SQUARE Ancilla Reuse + Dynamic Circuits")
    print("=" * 70)

    # Configuration
    N = 10
    n_steps = 1
    T = 1.0
    dt = T / n_steps
    pop_size = 30
    n_shots = 2000

    print(f"\nConfiguration:")
    print(f"  N = {N}")
    print(f"  Population size = {pop_size}")
    print(f"  Quantum shots = {n_shots}")

    # Generate interactions
    print("\nGenerating interactions...")
    G2, G4 = get_interactions(N)
    print(f"  G2 interactions: {len(G2)}")
    print(f"  G4 interactions: {len(G4)}")

    # Compute theta values
    print("\nComputing rotation angles...")
    thetas = []
    for step in range(1, n_steps + 1):
        t = step * dt
        theta_val = utils.compute_theta(t, dt, T, N, G2, G4)
        thetas.append(theta_val)
    print(f"  Computed {len(thetas)} angle(s)")

    # Generate quantum-enhanced population
    print("\n" + "=" * 70)
    quantum_pop, quantum_metrics = sample_quantum_population_enhanced(
        N=N,
        G2=G2,
        G4=G4,
        n_steps=n_steps,
        dt=dt,
        T=T,
        thetas=thetas,
        pop_size=pop_size,
        n_shots=n_shots,
        strategy='hybrid',
        use_deduplication=True
    )

    # Generate random population for comparison
    print("Generating random population for comparison...")
    random_pop = [[random.randint(0, 1) for _ in range(N)] for _ in range(pop_size)]
    random_energies = [utils.calculate_labs_energy(seq, N) for seq in random_pop]
    random_metrics = {
        'min_energy': min(random_energies),
        'max_energy': max(random_energies),
        'mean_energy': np.mean(random_energies),
        'std_energy': np.std(random_energies)
    }

    print("QUBIT REUSE ANALYSIS")
    if quantum_metrics.get('qubit_efficiency') is not None:
        print(f"Qubit efficiency: {quantum_metrics['qubit_efficiency']:.2%}")

    usage_stats = track_qubit_usage(N, G2, G4, n_steps)
    print(f"Most active qubit used {usage_stats['max_active_qubit']} times")
    print(f"Average qubit used {usage_stats['average_active_time']:.2f} times")

    if quantum_metrics.get('deduplication_stats'):
        dedup = quantum_metrics['deduplication_stats']
        print(f"\nMemory savings from deduplication:")
        print(f"  Compression ratio: {dedup['compression_ratio']:.2%}")
        print(f"  Space saved: {dedup['duplicates_removed']} sequences")

    # Comparison
    print(f"\n{'=' * 70}")
    print("QUANTUM vs RANDOM COMPARISON")
    print(f"{'=' * 70}")
    print(f"\n{'Metric':<20} {'Quantum':>15} {'Random':>15} {'Improvement':>15}")
    print(f"{'-' * 70}")
    print(f"{'Min Energy':<20} {quantum_metrics['min_energy']:>15} "
          f"{random_metrics['min_energy']:>15} "
          f"{random_metrics['min_energy'] - quantum_metrics['min_energy']:>15}")
    print(f"{'Mean Energy':<20} {quantum_metrics['mean_energy']:>15.2f} "
          f"{random_metrics['mean_energy']:>15.2f} "
          f"{random_metrics['mean_energy'] - quantum_metrics['mean_energy']:>15.2f}")

    improvement_pct = ((random_metrics['mean_energy'] - quantum_metrics['mean_energy'])
                       / random_metrics['mean_energy'] * 100)
    print(f"\nQuantum advantage: {improvement_pct:.1f}% better mean energy")
    print(f"{'=' * 70}\n")
