"""
Check that all algorithms agree with results from CPU benchmarking
(with classical MTS) for N=1 to N=14
"""
import sys
from pathlib import Path

# Ensure repo root is importable when running `pytest` from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Point at the optimised module.  Adjust this path if your layout differs.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
import classical.mts as mts
import quantum.qe_mts as qe_mts
from cpu_benchmark import BenchmarkParams
from quantum.bfdcqo import quantum_enhanced_mts

@pytest.fixture
def params():
    return BenchmarkParams

# found from CPU benchmarking with classical MTS
VALIDATION_RESULTS = [0, 1, 1, 2, 2, 7, 3, 8, 12, 13, 5, 10, 6, 19]

def test_classical_results(params: BenchmarkParams):
    """test results for classical MTS"""
    energies = []
    for n in range(1, 15):
        _, best_energy, _, _, _ = mts.MTS(
            k=params.k,
            N=n,
            target=0,
            max_iter=params.max_iter,
            p_sample=params.p_sample,
            p_mutate=params.p_mutate,
            tabu_steps=params.tabu_steps,
            tabu_tenure=params.tabu_tenure,
            population0=None,
        )
        energies.append(best_energy)

    assert energies == VALIDATION_RESULTS
        

def test_qemts_results(params: BenchmarkParams):
    """test results for quantum enhanced MTS"""
    energies = []
    for n in range(1, 15):
        # sample a quantum enhanced population
        population = qe_mts.quantum_population(
            popsize = params.k,
            N=n
        )
        
        _, best_energy, _, _, _ = mts.MTS(
            k=len(population),
            N=n,
            target=0,
            max_iter=params.max_iter,
            p_sample=params.p_sample,
            p_mutate=params.p_mutate,
            tabu_steps=params.tabu_steps,
            tabu_tenure=params.tabu_tenure,
            population0=population,
        )
        energies.append(best_energy)
    
    assert energies == VALIDATION_RESULTS


def test_bfdcqo_results():
    energies = []

    theta_cutoff = 0.06
    bf_dcqo_iter = 3
    pop_size = 100
    mts_iter = 1000
    alpha = 0.01
    kappa = 5
    n_iter = 11
    T = 1.0
    
    for n in range(1, 15):
        results = quantum_enhanced_mts(N=n, pop_size=pop_size, bf_dcqo_iter=bf_dcqo_iter, 
                                      mts_iter=mts_iter, alpha=alpha, kappa=kappa, 
                                      T=T, theta_cutoff=theta_cutoff, quantum_shots=1000)
        energies.append(results['solution']['energy'])
        
    assert energies == VALIDATION_RESULTS