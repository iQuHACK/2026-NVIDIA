"""Memetic Tabu Search (MTS) for the LABS problem.

This module implements a classical Memetic Tabu Search (MTS) heuristic for the
Low Autocorrelation Binary Sequences (LABS) optimization problem.

Representation
--------------
- A candidate solution is a length-N array `s` with entries in {-1, +1}.
- A population is an array of shape (k, N).

Objective
---------
Given `s`, define autocorrelation values:
    C_k(s) = sum_{i=0..N-k-1} s[i] * s[i+k],   for k=1..N-1
and the LABS energy:
    E(s) = sum_{k=1..N-1} C_k(s)^2

Algorithm sketch (MTS)
----------------------
1) Initialize population of k random bitstrings.
2) Track best (lowest-energy) solution in population.
3) Repeat until max_iter or target achieved:
   - Sample an existing solution OR combine two parents (one-point crossover)
   - Mutate each bit with probability p_mutate
   - Run tabu search (local improvement) starting from the child
   - Update global best if improved
   - Replace a random population member if the new solution is better

Testing
-------
Run unit tests with:
    `pytest`

Notes for performance work (next step)
--------------------------------------
The functions `_delta_energy_for_flip` and `_apply_flip_in_place` are written to
avoid recomputing the full LABS objective for each 1-bit flip. They’re good
targets for CuPy/Numba acceleration later.
"""

from typing import Optional, Tuple, List

import numpy as np


def generate_bitstrings(k: int, N: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate k random {-1,+1} bitstrings of length N.

    Returns:
        population: np.ndarray of shape (k, N)
    """
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(k, N))


def energy(s: np.ndarray) -> np.ndarray:
    """LABS energy.

    E(s) = sum_{k=1..N-1} C_k(s)^2, where C_k = sum_{i=1..N-k} s_i s_{i+k}

    Supports:
      - s shape (N,)  -> returns scalar np.int64
      - s shape (k,N) -> returns (k,) energies
    """
    s = np.asarray(s)

    if s.ndim == 1:
        N = s.shape[0]
        e = np.int64(0)
        for shift in range(1, N):
            ck = int(np.dot(s[: N - shift], s[shift:]))
            e += np.int64(ck * ck)
        return e

    if s.ndim == 2:
        k_pop, N = s.shape
        e = np.zeros(k_pop, dtype=np.int64)
        for shift in range(1, N):
            ck = (s[:, : N - shift] * s[:, shift:]).sum(axis=1, dtype=np.int64)
            e += ck * ck
        return e

    raise ValueError("s must be 1D or 2D")


def _autocorr_vector(s: np.ndarray) -> np.ndarray:
    """Return C_k for k=0..N-1 (C_0 unused, set to 0)."""
    N = s.shape[0]
    C = np.zeros(N, dtype=np.int64)
    for k in range(1, N):
        C[k] = int(np.dot(s[: N - k], s[k:]))
    return C


def _delta_energy_for_flip(s: np.ndarray, C: np.ndarray, j: int) -> int:
    """Compute delta E if we flip s[j] (does not mutate s or C)."""
    N = s.shape[0]
    sj = int(s[j])
    delta = 0
    for k in range(1, N):
        dCk = 0
        if j < N - k:
            dCk += -2 * sj * int(s[j + k])
        if j >= k:
            dCk += -2 * int(s[j - k]) * sj

        if dCk:
            ck = int(C[k])
            delta += 2 * ck * dCk + dCk * dCk
    return int(delta)


def _apply_flip_in_place(s: np.ndarray, C: np.ndarray, j: int) -> None:
    """Flip s[j] and update C_k in place."""
    N = s.shape[0]
    sj_old = int(s[j])

    for k in range(1, N):
        if j < N - k:
            C[k] += -2 * sj_old * int(s[j + k])
        if j >= k:
            C[k] += -2 * int(s[j - k]) * sj_old

    s[j] = -s[j]


def combine(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One-point crossover (Algorithm 3: Combine)."""
    N = p1.shape[0]
    cut = int(rng.integers(1, N))  # {1, ..., N-1}
    child = np.empty_like(p1)
    child[:cut] = p1[:cut]
    child[cut:] = p2[cut:]
    return child


def mutate(s: np.ndarray, p_mut: float, rng: np.random.Generator) -> np.ndarray:
    """Bit-flip mutation (Algorithm 3: Mutate)."""
    if p_mut <= 0:
        return s
    flips = rng.random(s.shape[0]) < p_mut
    out = s.copy()
    out[flips] *= -1
    return out


def tabu_search(
    s0: np.ndarray,
    target: int = 0,
    max_steps: int = 250,
    tabu_tenure: int = 10,
    stall_limit: int = 75,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Tabu search over 1-bit flips with aspiration criterion.

    - Tabu list stores recently flipped indices.
    - Aspiration: allow tabu move if it improves the best-so-far energy.
    """
    rng = np.random.default_rng(seed)

    s = np.asarray(s0, dtype=np.int8).copy()
    N = s.shape[0]

    C = _autocorr_vector(s)
    E = int(np.sum(C[1:] * C[1:], dtype=np.int64))

    best_s = s.copy()
    best_E = E

    # tabu_expire[j] = step index when tabu expires (0 means not tabu)
    tabu_expire = np.zeros(N, dtype=np.int64)
    stall = 0

    for step in range(1, max_steps + 1):
        if best_E <= target:
            break

        best_move_j = None
        best_move_E = None

        # Choose best admissible move among all 1-bit flips.
        for j in range(N):
            cand_E = E + _delta_energy_for_flip(s, C, j)

            is_tabu = tabu_expire[j] > step
            admissible = (not is_tabu) or (cand_E < best_E)  # aspiration
            if not admissible:
                continue

            if best_move_E is None or cand_E < best_move_E:
                best_move_E = cand_E
                best_move_j = j

        if best_move_j is None:
            break

        # Apply chosen move.
        _apply_flip_in_place(s, C, best_move_j)
        E = int(np.sum(C[1:] * C[1:], dtype=np.int64))

        # Set tabu tenure (small randomization helps avoid cycles).
        jitter = int(rng.integers(0, max(1, tabu_tenure // 3 + 1)))
        tabu_expire[best_move_j] = step + tabu_tenure + jitter

        if E < best_E:
            best_E = E
            best_s = s.copy()
            stall = 0
        else:
            stall += 1
            if stall >= stall_limit:
                break

    return best_s, int(best_E)


def MTS(
    k: int,
    N: int,
    target: int = 0,
    max_iter: int = 500,
    p_sample: float = 0.5,
    p_mutate: float = 0.02,
    tabu_steps: int = 250,
    tabu_tenure: int = 10,
    population0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
):
    """Memetic Tabu Search (MTS) as depicted in the Exercise 2 figure.

    Returns:
        best_s: best bitstring found (N,)
        best_E: best energy found (int)
        population: final population (k,N)
        energies: energies for final population (k,)
        best_history: list of best energies over iterations
    """
    rng = np.random.default_rng(seed)

    # 1) generate initial population (or use a provided seed population)
    if population0 is None:
        population = rng.choice(np.array([-1, 1], dtype=np.int8), size=(k, N))
    else:
        population = np.asarray(population0, dtype=np.int8)
        if population.ndim != 2:
            raise ValueError("population0 must have shape (k, N)")
        k, N = population.shape

    energies = energy(population).astype(np.int64)

    # 3) find best in initial population
    best_idx = int(np.argmin(energies))
    best_s = population[best_idx].copy()
    best_E = int(energies[best_idx])

    best_history: List[int] = [best_E]

    for it in range(max_iter):
        if best_E <= target:
            break

        # (a) sample or combine
        if rng.random() < p_sample:
            child = population[int(rng.integers(0, k))].copy()
        else:
            i1, i2 = rng.choice(k, size=2, replace=False)
            child = combine(population[int(i1)], population[int(i2)], rng)

        # (b) mutate based on probability
        child = mutate(child, p_mutate, rng)

        # (c) tabu search (local improvement)
        local_s, local_E = tabu_search(
            child,
            target=target,
            max_steps=tabu_steps,
            tabu_tenure=tabu_tenure,
            seed=None if seed is None else (seed + 1000 + it),
        )

        # (d) update if lower energy
        if local_E < best_E:
            best_E = int(local_E)
            best_s = local_s.copy()

        # (e) randomly replace population member if better
        r = int(rng.integers(0, k))
        if local_E < energies[r]:
            population[r] = local_s
            energies[r] = int(local_E)

        best_history.append(best_E)

    return best_s, best_E, population, energies, best_history


def plot_population_energy_distribution(
    energies: np.ndarray, best_history: Optional[List[int]] = None
):
    """Visualize histogram of population energies (+ optional best-energy trace)."""
    import matplotlib.pyplot as plt

    energies = np.asarray(energies)

    fig, ax = plt.subplots(1, 2 if best_history is not None else 1, figsize=(10, 3.5))
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    bins = min(30, max(5, int(np.sqrt(len(energies)))))
    ax[0].hist(energies, bins=bins, edgecolor="black")
    ax[0].set_title("Final population energy distribution")
    ax[0].set_xlabel("Energy")
    ax[0].set_ylabel("Count")

    if best_history is not None:
        ax[1].plot(best_history)
        ax[1].set_title("Best energy over MTS iterations")
        ax[1].set_xlabel("Iteration")
        ax[1].set_ylabel("Best energy")
        ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


__all__ = [
    "generate_bitstrings",
    "energy",
    "combine",
    "mutate",
    "tabu_search",
    "MTS",
    "plot_population_energy_distribution",
]
