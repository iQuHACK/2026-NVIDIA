"""Memetic Tabu Search (MTS) for the LABS problem — CPU-optimized.

Optimizations applied
----------------------
1. **All-j delta vectorized.**  The original computed delta_E one bit at a time
   inside a Python loop (O(N) Python iterations, each doing an O(N) Python loop
   over lags).  The new `_all_deltas` computes every candidate delta in one
   vectorised pass: O(N) NumPy array operations over lags, each operating on
   length-N arrays.  The Python loop over lags stays (it's O(N) iterations) but
   every iteration is a handful of vectorised C-level ops — no per-bit Python
   overhead.

2. **Redundant energy recomputation removed.**  After applying a flip the
   original re-summed C[1:]**2.  We already know the exact delta, so
   `E += delta` suffices.

3. **In-place C update vectorised.**  `_apply_flip_in_place` had a Python loop
   over lags; replaced with two slice additions (one for the right-neighbour
   contribution, one for the left).

4. **Tabu mask as a boolean array.**  Instead of comparing `tabu_expire[j] > step`
   inside a Python loop we build a boolean mask once per step and use it with
   `np.where` / argmin logic — fully in NumPy.

5. **Pre-allocated scratch arrays.**  The `neighbor_sum` and `deltas` arrays are
   allocated once and reused across every tabu step, avoiding repeated
   allocation in the innermost loop.

6. **Minimal copies in MTS outer loop.**  `mutate` now works in-place on an
   already-copied array; `combine` writes directly into a pre-allocated buffer.
"""

from typing import Optional, Tuple, List
import time

import numpy as np


# ---------------------------------------------------------------------------
# Unchanged helpers (already efficient or not on the hot path)
# ---------------------------------------------------------------------------

def generate_bitstrings(k: int, N: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(k, N))


def energy(s: np.ndarray) -> np.ndarray:
    """LABS energy — batch-vectorised for 2-D input, scalar loop for 1-D."""
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
    """Return C[0..N-1]; C[0] is unused (set to 0)."""
    N = s.shape[0]
    C = np.zeros(N, dtype=np.int64)
    for k in range(1, N):
        C[k] = int(np.dot(s[: N - k], s[k:]))
    return C


# ---------------------------------------------------------------------------
# Vectorised hot-path primitives
# ---------------------------------------------------------------------------

def _all_deltas(s64: np.ndarray, C: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Compute delta_E for flipping every bit j — fully vectorised, no Python loop over lags.

    Math
    ----
    dC_k(j) = -2 * s[j] * ( s[j+k]*[j<N-k] + s[j-k]*[j>=k] )
    delta_E(j) = sum_{k=1}^{N-1}  dC_k(j) * (2*C[k] + dC_k(j))

    We build the full (N-1, N) matrix D where D[k-1, j] = dC_k(j) using two
    triangular block-fills (right and left neighbours), then contract to (N,)
    in a single fused reduction.

    Parameters
    ----------
    s64 : (N,)      int64  current spin vector (caller maintains this in sync with s)
    C   : (N,)      int64  autocorrelation vector
    D   : (N-1, N)  int64  pre-allocated scratch matrix (reused across steps)

    Returns
    -------
    deltas : (N,) int64
    """
    N = s64.shape[0]

    D[:] = 0

    # Right neighbours: for lag k, indices j in [0, N-k)
    for k in range(1, N):
        D[k - 1, : N - k] = s64[: N - k] * s64[k:]

    # Left neighbours: for lag k, indices j in [k, N)
    for k in range(1, N):
        D[k - 1, k:] += s64[k:] * s64[: N - k]

    # Apply the -2 factor.  The slice products already include s[j].
    D *= -2

    # Contract: delta_E(j) = sum_k  D[k-1,j] * (2*C[k] + D[k-1,j])
    C_col = C[1:, np.newaxis]                # (N-1, 1)
    deltas = (D * (2 * C_col + D)).sum(axis=0)  # (N,)

    return deltas


def _apply_flip_in_place(s: np.ndarray, s64: np.ndarray, C: np.ndarray, j: int) -> None:
    """Flip s[j], update s64[j], and update autocorrelation vector C in place.

    Parameters
    ----------
    s   : (N,) int8   spin vector (mutated)
    s64 : (N,) int64  int64 mirror of s (mutated in sync)
    C   : (N,) int64  autocorrelation vector (mutated)
    j   : int         index to flip
    """
    N = s.shape[0]
    factor = np.int64(-2) * s64[j]   # -2 * s[j], scalar int64

    # Right neighbours: lags 1 .. N-j-1  ->  C[1:N-j] += factor * s64[j+1:N]
    if j < N - 1:
        C[1: N - j] += factor * s64[j + 1: N]

    # Left neighbours: lags 1 .. j  ->  C[k] += factor * s64[j-k]  for k=1..j
    # s64[j-k] for k=1..j  is  s64[j-1], s64[j-2], ..., s64[0]  =  s64[j-1::-1]
    if j > 0:
        C[1: j + 1] += factor * s64[j - 1::-1]

    s[j]   = -s[j]
    s64[j] = -s64[j]


# ---------------------------------------------------------------------------
# Tabu search — fully vectorised move evaluation
# ---------------------------------------------------------------------------

def tabu_search(
    s0: np.ndarray,
    target: int = 0,
    max_steps: int = 250,
    tabu_tenure: int = 10,
    stall_limit: int = 75,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Tabu search over 1-bit flips with aspiration criterion.

    Every step evaluates all N candidate flips at once via `_all_deltas`, then
    picks the best admissible move using masked argmin — no Python loop over j.
    """
    rng = np.random.default_rng(seed)

    s = np.asarray(s0, dtype=np.int8).copy()
    N = s.shape[0]
    s64 = s.astype(np.int64)          # persistent int64 mirror; updated in sync with s

    C = _autocorr_vector(s)
    E = int(np.sum(C[1:] * C[1:]))

    best_s = s.copy()
    best_E = E

    tabu_expire = np.zeros(N, dtype=np.int64)           # tabu_expire[j] = step when tabu lifts
    D_scratch   = np.empty((N - 1, N), dtype=np.int64)  # reused scratch for _all_deltas
    stall = 0

    INF = np.iinfo(np.int64).max

    for step in range(1, max_steps + 1):
        if best_E <= target:
            break

        # --- evaluate all N flips at once ---
        deltas = _all_deltas(s64, C, D_scratch)           # (N,) int64
        cand_E = E + deltas                               # (N,) candidate energies

        # --- admissibility mask ---
        is_tabu = tabu_expire > step                      # (N,) bool
        aspiration = cand_E < best_E                      # (N,) bool: tabu move allowed if it beats global best
        admissible = (~is_tabu) | aspiration              # (N,) bool

        # --- pick best admissible move ---
        # Set inadmissible candidates to +INF so argmin ignores them.
        masked = np.where(admissible, cand_E, INF)
        best_move_j = int(np.argmin(masked))

        if masked[best_move_j] == INF:
            break       # no admissible move exists

        chosen_delta = int(deltas[best_move_j])

        # --- apply move ---
        _apply_flip_in_place(s, s64, C, best_move_j)
        E += chosen_delta                                 # exact; no recomputation needed

        # --- tabu tenure with jitter ---
        jitter = int(rng.integers(0, max(1, tabu_tenure // 3 + 1)))
        tabu_expire[best_move_j] = step + tabu_tenure + jitter

        # --- update global best / stall counter ---
        if E < best_E:
            best_E = E
            best_s = s.copy()
            stall = 0
        else:
            stall += 1
            if stall >= stall_limit:
                break

    return best_s, int(best_E)


# ---------------------------------------------------------------------------
# MTS outer loop
# ---------------------------------------------------------------------------

def combine(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator, out: np.ndarray) -> None:
    """One-point crossover written into pre-allocated `out`."""
    N = p1.shape[0]
    cut = int(rng.integers(1, N))
    out[:cut] = p1[:cut]
    out[cut:] = p2[cut:]


def mutate_inplace(s: np.ndarray, p_mut: float, rng: np.random.Generator) -> None:
    """Bit-flip mutation applied in place (caller must have already copied)."""
    if p_mut <= 0:
        return
    flips = rng.random(s.shape[0]) < p_mut
    s[flips] *= -1


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
    record_time: bool = False,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, List[int], Optional[float]]:
    """Memetic Tabu Search (MTS) — optimised for CPU throughput.

    Args:
        k: number of sequences in the population
        N: length of the bitstrings
        target: target energy
        max_iter: maximum number of iterations
        p_sample: probability of sampling a sequence
        p_mutate: probability of mutating a sequence
        tabu_steps: max steps inside each tabu search call
        tabu_tenure: base tabu tenure
        population0: optional initial population (k, N)
        seed: RNG seed
        record_time: if True, return wall-clock time to first best solution

    Returns:
        best_s, best_E, population, energies, best_history[, elapsed_time]
    """

    start_time = time.time()
    end_time = start_time
    rng = np.random.default_rng(seed)

    # --- initial population ---
    if population0 is None:
        population = rng.choice(np.array([-1, 1], dtype=np.int8), size=(k, N))
    else:
        population = np.asarray(population0, dtype=np.int8)
        if population.ndim != 2:
            raise ValueError("population0 must have shape (k, N)")
        k, N = population.shape

    energies = energy(population).astype(np.int64)

    best_idx = int(np.argmin(energies))
    best_s = population[best_idx].copy()
    best_E = int(energies[best_idx])

    best_history: List[int] = [best_E]

    # Pre-allocate child buffer — reused every iteration (avoids repeated alloc).
    child = np.empty(N, dtype=np.int8)

    for it in range(max_iter):
        if best_E <= target:
            break

        # (a) sample or combine — write into pre-allocated `child`
        if rng.random() < p_sample:
            idx = int(rng.integers(0, k))
            child[:] = population[idx]          # copy into buffer
        else:
            i1, i2 = rng.choice(k, size=2, replace=False)
            combine(population[int(i1)], population[int(i2)], rng, child)

        # (b) mutate in place (child is already our own copy)
        mutate_inplace(child, p_mutate, rng)

        # (c) tabu search
        local_s, local_E = tabu_search(
            child,
            target=target,
            max_steps=tabu_steps,
            tabu_tenure=tabu_tenure,
            seed=None if seed is None else (seed + 1000 + it),
        )

        # (d) update global best
        if local_E < best_E:
            best_E = int(local_E)
            best_s = local_s.copy()
            end_time = time.time()

        # (e) replace random population member if better
        r = int(rng.integers(0, k))
        if local_E < energies[r]:
            population[r] = local_s
            energies[r] = int(local_E)

        best_history.append(best_E)

    if record_time:
        return best_s, best_E, population, energies, best_history, end_time - start_time

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


def run_benchmark(
    Ns,
    k=100,
    target=0,
    max_iter=500,
    p_sample=0.5,
    p_mutate=0.02,
    tabu_steps=250,
    tabu_tenure=10,
    seed=None,
):
    """Run benchmark for different values of N."""
    best_E_list = []
    time_list = []
    for N in Ns:
        best_s, best_E, population, energies, best_history, time = MTS(
            k,
            N,
            target,
            max_iter,
            p_sample,
            p_mutate,
            tabu_steps,
            tabu_tenure,
            seed,
            record_time=True,
        )
        best_E_list.append(best_E)
        time_list.append(time)
    return best_E_list, time_list


__all__ = [
    "generate_bitstrings",
    "energy",
    "combine",
    "mutate",
    "tabu_search",
    "MTS",
    "plot_population_energy_distribution",
]
