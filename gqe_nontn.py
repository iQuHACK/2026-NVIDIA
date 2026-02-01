import argparse
import numpy as np
import cudaq
import os

# ==========================================
# 1. Argument Parsing and Environment Setup
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument("--mpi", action="store_true", help="Enable MPI distribution")
args = parser.parse_args()

# Initialize CUDA-Q target based on MPI flag
if args.mpi:
    try:
        cudaq.set_target("nvidia", option="mqpu")
        cudaq.mpi.initialize()
    except RuntimeError:
        print(
            "Warning: NVIDIA GPUs or MPI not available, unable to use CUDA-Q MQPU. Skipping..."
        )
        exit(0)
else:
    try:
        cudaq.set_target("nvidia", option="fp64")
    except RuntimeError:
        # Fallback to CPU if NVIDIA target is not available
        cudaq.set_target("qpp-cpu")

# ==========================================
# 2. Imports and Reproducibility Configuration
# ==========================================

import cudaq_solvers as solvers
import torch
from cudaq import spin
from lightning.fabric.loggers import CSVLogger
from src.GQEMTS.gqe import get_default_config

# Set deterministic behavior for reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.manual_seed(3047)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==========================================
# 3. Hamiltonian Definition (LABS Problem)
# ==========================================

def labs_spin_op(N: int):
    """
    Constructs the Hamiltonian for the Low Autocorrelation Binary Sequence (LABS) problem.
    """
    H = 0.0

    # ---- 2-body interaction terms ----
    for i in range(N - 2):
        max_k = (N - i) // 2
        for k in range(1, max_k + 1):
            H += 2.0 * spin.z(i) * spin.z(i + k)

    # ---- 4-body interaction terms ----
    for i in range(N - 3):
        max_t = (N - i - 1) // 2
        for t in range(1, max_t + 1):
            for k in range(t + 1, N - i - t):
                H += (
                    4.0
                    * spin.z(i)
                    * spin.z(i + t)
                    * spin.z(i + k)
                    * spin.z(i + k + t)
                )

    return H

# Initialize problem parameters
N = 12
spin_ham = labs_spin_op(N)
n_qubits = N

# ==========================================
# 4. Operator Pool Construction
# ==========================================

# Coefficients for the rotation gates
params = [
    0.05,
    -0.05,
    0.1,
    -0.1,
]

def pool(params, n_qubits):
    """
    Generates a pool of variational operators (Ansatz pool).
    Includes 2-qubit and 4-qubit rotation terms.
    """
    ops = []

    # 2-qubit rotations (Nearest Neighbor)
    for i in range(n_qubits - 1):
        ops.append(cudaq.SpinOperator(spin.y(i) * spin.z(i + 1)))
        ops.append(cudaq.SpinOperator(spin.z(i) * spin.y(i + 1)))

    # 4-qubit rotations
    for i in range(n_qubits - 3):
        ops.append(cudaq.SpinOperator(spin.y(i) * spin.z(i + 1) * spin.z(i + 2) * spin.z(i + 3)))
        ops.append(cudaq.SpinOperator(spin.z(i) * spin.y(i + 1) * spin.z(i + 2) * spin.z(i + 3)))
        ops.append(cudaq.SpinOperator(spin.z(i) * spin.z(i + 1) * spin.y(i + 2) * spin.z(i + 3)))
        ops.append(cudaq.SpinOperator(spin.z(i) * spin.z(i + 1) * spin.z(i + 2) * spin.y(i + 3)))
    
    pool_ops = []

    # Combine base operators with parameters
    for c in params:
        for op in ops:
            pool_ops.append(c * op)

    return pool_ops

# Create the operator pool
op_pool = pool(params, n_qubits)
print("Number of operators in pool:", len(op_pool))

# ==========================================
# 5. Helper Functions & Kernels
# ==========================================

def term_coefficients(op: cudaq.SpinOperator) -> list[complex]:
    """Extract coefficients from a SpinOperator."""
    return [term.evaluate_coefficient() for term in op]


def term_words(op: cudaq.SpinOperator) -> list[cudaq.pauli_word]:
    """Extract Pauli words from a SpinOperator."""
    return [term.get_pauli_word(n_qubits) for term in op]


@cudaq.kernel
def kernel(coeffs: list[float], words: list[cudaq.pauli_word]):
    """
    Quantum Kernel for energy estimation.
    Applies Hadamard gates followed by the parameterized Pauli exponentials.
    """
    q = cudaq.qvector(n_qubits)

    # Start from superposition state |+>^N
    for i in range(n_qubits):
        h(q[i])

    # Apply parameterized ansatz
    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])


def cost(sampled_ops: list[cudaq.SpinOperator], **kwargs):
    """
    Cost function to evaluate the expectation value of the Hamiltonian.
    Handles both MPI (async) and standard execution.
    """
    full_coeffs = []
    full_words = []

    # Flatten the operators into coefficients and words for the kernel
    for op in sampled_ops:
        full_coeffs += [c.real for c in term_coefficients(op)]
        full_words += term_words(op)

    if args.mpi:
        handle = cudaq.observe_async(
            kernel,
            spin_ham,
            full_coeffs,
            full_words,
            qpu_id=kwargs["qpu_id"],
        )
        # Return handle and a lambda to retrieve the result later
        return handle, lambda res: res.get().expectation()
    else:
        return cudaq.observe(
            kernel, spin_ham, full_coeffs, full_words
        ).expectation()


@cudaq.kernel
def sample_optimized(coeffs: list[float], words: list[cudaq.pauli_word]):
    """
    Quantum Kernel for sampling the final state.
    Similar to 'kernel' but includes measurement (mz).
    """
    q = cudaq.qvector(n_qubits)

    # Start from |+>^N
    for i in range(n_qubits):
        h(q[i])

    # Apply the optimized evolution
    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])

    # Measure all qubits in the Z-basis
    for qubit in q:
        mz(qubit)


def labs_energy(x):
    """
    Calculates the classical LABS energy for a given bitstring configuration x.
    Used to verify the quantum result.
    """
    s = 2 * np.asarray(x) - 1  # Convert 0/1 to -1/+1
    N = len(s)
    E = 0
    for k in range(1, N):
        Ck = np.sum(s[:N - k] * s[k:])
        E += Ck**2
    return E

# ==========================================
# 6. Main Execution (GQE Solver)
# ==========================================

# Configure GQE (Genetic Quantum Eigensolver) settings
cfg = get_default_config()
cfg.use_fabric_logging = False
logger = CSVLogger("gqe_h2_logs/gqe.csv")
cfg.fabric_logger = logger
cfg.save_trajectory = False
cfg.verbose = True

# Run the GQE solver
minE, best_ops = solvers.gqe(cost, op_pool, max_iters=5, ngates=3, config=cfg)

# ==========================================
# 7. Results and Post-Processing
# ==========================================

# Only print results from the root rank if using MPI
if not args.mpi or cudaq.mpi.rank() == 0:
    print(f"Ground Energy = {minE}")
    print("Ansatz Ops")
    
    opt_coeffs = []
    opt_words = []
    
    # Process the best operators found by GQE
    for idx in best_ops:
        # Print the first term of the operator for inspection
        term = next(iter(op_pool[idx]))
        print(term.evaluate_coefficient().real, term.get_pauli_word(n_qubits))
        
        # Collect all terms for the final sampling circuit
        op = op_pool[idx]
        for term in op:
            opt_coeffs.append(term.evaluate_coefficient().real)
            opt_words.append(term.get_pauli_word(n_qubits))

    shots = 1000

    # Execute the optimized circuit to get samples
    samples = cudaq.sample(sample_optimized, opt_coeffs, opt_words, shots_count=shots)

    # Process sample results
    from collections import Counter
    counts = Counter(dict(samples.items()))

    print("10 most frequent bitstrings:")
    print(counts.most_common(10))

    # Calculate exact classical energies for the sampled bitstrings
    sample_energies = [labs_energy([int(c) for c in b]) for b in counts.keys()]
    print("Minimum sampled LABS energy:", min(sample_energies))

# Finalize MPI if necessary
if args.mpi:
    cudaq.mpi.finalize()