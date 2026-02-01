import time
import cudaq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classical.mts import MTS
import tutorial_notebook.auxiliary_files.labs_utils as utils

### CudaQ kernels for 2q and 4q terms ###
@cudaq.kernel
def r_zz(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
    cx(q0, q1)
    rz(theta, q1)
    cx(q0, q1)

@cudaq.kernel
def r_yz(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
    # Y → Z on q0
    h(q0)
    s(q0)

    r_zz(q0, q1, 2.0 * theta)

    # Undo basis change
    s.adj(q0)
    h(q0)

@cudaq.kernel
def r_zy(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
    # Y → Z on q1
    h(q1)
    s(q1)

    r_zz(q0, q1, 2.0 * theta)

    # Undo basis change
    s.adj(q1)
    h(q1)


@cudaq.kernel
def r_zzzz(q0: cudaq.qubit,
          q1: cudaq.qubit,
          q2: cudaq.qubit,
          q3: cudaq.qubit,
          theta: float):
    cx(q0, q1)
    cx(q1, q2)
    cx(q2, q3)

    rz(2.0 * theta, q3)

    cx(q2, q3)
    cx(q1, q2)
    cx(q0, q1)


@cudaq.kernel
def r_yzzz(q0: cudaq.qubit,
           q1: cudaq.qubit,
           q2: cudaq.qubit,
           q3: cudaq.qubit,
           theta: float):
    s(q0)
    h(q0)

    r_zzzz(q0, q1, q2, q3, theta)

    s.adj(q0)
    h(q0)


@cudaq.kernel
def r_zyzz(q0: cudaq.qubit,
           q1: cudaq.qubit,
           q2: cudaq.qubit,
           q3: cudaq.qubit,
           theta: float):
    s(q1)
    h(q1)

    r_zzzz(q0, q1, q2, q3, theta)

    s.adj(q1)
    h(q1)


@cudaq.kernel
def r_zzyz(q0: cudaq.qubit,
           q1: cudaq.qubit,
           q2: cudaq.qubit,
           q3: cudaq.qubit,
           theta: float):
    s(q2)
    h(q2)

    r_zzzz(q0, q1, q2, q3, theta)

    s.adj(q2)
    h(q2)


@cudaq.kernel
def r_zzzy(q0: cudaq.qubit,
           q1: cudaq.qubit,
           q2: cudaq.qubit,
           q3: cudaq.qubit,
           theta: float):
    s(q3)
    h(q3)
    r_zzzz(q0, q1, q2, q3, theta)
    s.adj(q3)
    h(q3)


def get_interactions(N: int) -> (list, list):
    """
    Generates the interaction sets G2 and G4 based on the loop limits in Eq. 15.
    Returns standard 0-based indices as lists of lists of ints.
   
    Args:
        N (int): Sequence length.
       
    Returns:
        G2: List of lists containing two body term indices
        G4: List of lists containing four body term indices
    """
   
    G2 = []
    G4 = []

    # --- Two-body terms ---
    for i in range(N - 2):
        max_k = (N - i) // 2          # depends on i
        for k in range(1, max_k + 1): # starts at 1 to avoid (i,i)
            G2.append([i, i + k])

    # --- Four-body terms ---
    for i in range(N - 3):
        max_t = (N - i - 1) // 2
        for t in range(1, max_t + 1):
            for k in range(t + 1, N - i - t):
                quad = [i, i + t, i + k, i + k + t]
                if quad[3] < N:
                    G4.append(quad)

    return G2, G4


@cudaq.kernel
def trotterized_circuit(N: int, G2: list[list[int]], G4: list[list[int]], steps: int, dt: float, T: float, thetas: list[float]):
    """Equation B3 to create a trotterized circuit
    
    N:      the size of the bit strings
    G2:     list of lists containing two body term indices
    G4:     list of lists containing four body term indices
    steps:  number of trotter steps
    dt:     time step to approximate time evolution
    T:      total time
    thetas: thetas used in Equation B3
    """
    reg = cudaq.qvector(N)
    h(reg)

    for s in range(steps):
        theta = thetas[s]

        # 2-body block
        for pair in G2:
            i = pair[0]
            j = pair[1]

            r_yz(reg[i], reg[j], -4.0 * theta)
            r_zy(reg[i], reg[j], -4.0 * theta)


        # 4-body block
        for quad in G4:
            a, b, c, d = quad[0], quad[1], quad[2], quad[3]

            r_yzzz(reg[a], reg[b], reg[c], reg[d], -8.0 * theta)
            r_zyzz(reg[a], reg[b], reg[c], reg[d], -8.0 * theta)
            r_zzyz(reg[a], reg[b], reg[c], reg[d], -8.0 * theta)
            r_zzzy(reg[a], reg[b], reg[c], reg[d], -8.0 * theta)


# LABS: generates spin {-1, 1} for bitstrings, not {0, 1}
def bitstring_convert(bitstring: str) -> np.ndarray:
    """
    Convert cudaq.sample output to {-1, +1} bitstrings
    
    bitstring: the bitstring from a cudaq sample to convert
    """
    return np.array([1 if b == '1' else -1 for b in bitstring])


def quantum_population(
    popsize: int = 100,
    T: int = 1,
    n_steps: int = 3,
    N: int = 11,
    shots_count: int = 1000,
) -> list:
    """
    Generate the quantum enhanced population

    popsize: the population size
    T:       total time
    n_steps: number of trotterization steps
    N:       length of bitstrings
    """
    # initialize parameters for cudaq samples
    dt = T / n_steps
    G2, G4 = get_interactions(N)
    thetas =[]
    for step in range(1, n_steps + 1):
        t = step * dt
        theta_val = utils.compute_theta(t, dt, T, N, G2, G4)
        thetas.append(theta_val)
        
    # generate samples
    samples = cudaq.sample(
        trotterized_circuit,
        N,
        G2,
        G4,
        n_steps,
        dt,
        T,
        thetas,
        shots_count=int(shots_count),
    )

    population = []
    for bitstring, count in samples.items():
       for _ in range(count):
           population.append(bitstring_convert(bitstring))
           if len(population) >= popsize:
               return population
               
    return population


def qe_mts(population):
    """
    Run MTS with a quantum enhanced population

    population: the quantum enhanced population
    """
    N = len(population[0])

    # MTS results
    return MTS(len(population), N, population0 = population)


def generate_qemts_data(Ns, popsize=100, T=1, n_steps=3, shots_count: int = 1000):
    """
    Generate data for best energies and time to convergence for
    different values of N

    Ns:      an array of values for N
    popsize: population size
    T:       total time
    n_steps: number of trotterization steps
    """
    # list of best energies
    best_E_q_list = []

    # list of times to solution
    time_q_list = []

    for n in Ns:
        # run QE-MTS
        quantum_init = quantum_population(popsize, T, n_steps, n, shots_count=shots_count)
        quantum_final = MTS(len(quantum_init), n, population0=quantum_init, record_time=True)

        _, best_E_q, _, _, _, convergence_time = quantum_final
        best_E_q_list.append(best_E_q)
        time_q_list.append(convergence_time)

    data = pd.DataFrame({
        "N": Ns,
        "best_E_q": best_E_q_list,
        "time_q": time_q_list
    })
    
    return data