import cupy as cp
from typing import List
import numpy as np


def calculate_labs_energy_gpu(sequences: cp.ndarray, N: int) -> cp.ndarray:
    """
    GPU-accelerated LABS energy calculation.

    Args:
        sequences: CuPy array of shape (batch_size, N) with values in {0,1} or {-1,+1}
    """
    # Convert to spins if needed
    if cp.any((sequences == 0) | (sequences == 1)):
        spins = 2 * sequences - 1
    else:
        spins = sequences

    batch_size = spins.shape[0]
    energies = cp.zeros(batch_size, dtype=cp.int32)

    # Vectorized autocorrelation
    for k in range(1, N):
        autocorr = cp.sum(spins[:, :-k] * spins[:, k:], axis=1)
        energies += autocorr ** 2

    return energies


def evaluate_population_gpu(population: List[List[int]], N: int) -> np.ndarray:
    """Batch evaluate population on GPU."""
    # Convert to CuPy array
    pop_array = cp.array(population, dtype=cp.int32)

    # Calculate energies on GPU
    energies_gpu = calculate_labs_energy_gpu(pop_array, N)

    # Return to CPU
    return cp.asnumpy(energies_gpu)