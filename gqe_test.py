# GQE is an optional component of the CUDA-QX Solvers Library. To install its
# dependencies, run:
# pip install -e .

# Run this script with
# python3 gqe_h2.py
#
# In order to leverage CUDA-Q MQPU and distribute the work across
# multiple QPUs (thereby observing a speed-up), run with:
#
# mpiexec -np N and vary N to see the speedup...
# e.g. PMIX_MCA_gds=hash mpiexec -np 2 python3 gqe_h2.py --mpi

import argparse

import cudaq

parser = argparse.ArgumentParser()
parser.add_argument("--mpi", action="store_true")
args = parser.parse_args()

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
        cudaq.set_target("qpp-cpu")

# Set deterministic seed and environment variables for deterministic behavior
# Disable this section for non-deterministic behavior
import os


import torch
from cudaq import spin
from lightning.fabric.loggers import CSVLogger

from src.GQEMTS.gqe import get_default_config

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.manual_seed(3047)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create the molecular hamiltonian
# geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7474))]
# molecule = solvers.create_molecule(geometry, "sto-3g", 0, 0, casci=True)

# spin_ham = molecule.hamiltonian

from cudaq import spin

def labs_spin_op(N: int):
    H = 0.0

    # ---- 2-body terms ----
    for i in range(N - 2):
        max_k = (N - i) // 2
        for k in range(1, max_k + 1):
            H += 2.0 * spin.z(i) * spin.z(i + k)

    # ---- 4-body terms ----
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

N=5
spin_ham = labs_spin_op(N)
n_qubits = N


params = [ 0.05,
    -0.05,
    0.1,
    -0.1,
]

def pool(params, n_qubits):
    ops = []

    # --- 2-qubit patterns ---
    for i in range(n_qubits - 1):
        ops.append(cudaq.SpinOperator(spin.x(i) * spin.x(i+1)))
        ops.append(cudaq.SpinOperator(spin.z(i) * spin.z(i+1)))

    # --- 4-qubit patterns ---
    for i in range(n_qubits - 3):
        ops.append(cudaq.SpinOperator(spin.x(i) * spin.x(i+1) * spin.y(i+2) * spin.y(i+3)))
        ops.append(cudaq.SpinOperator(spin.z(i) * spin.z(i+1) * spin.z(i+2) * spin.z(i+3)))

    pool_ops = []
    
    
    for c in params:
        for op in ops:
            pool_ops.append(c * op)

    return pool_ops

op_pool = pool(params, n_qubits)
print("Number of operators in pool:", len(op_pool))




# op_pool = pool(params)


def term_coefficients(op: cudaq.SpinOperator) -> list[complex]:
    return [term.evaluate_coefficient() for term in op]


def term_words(op: cudaq.SpinOperator) -> list[cudaq.pauli_word]:
    return [term.get_pauli_word(n_qubits) for term in op]


# Kernel that applies the selected operators
@cudaq.kernel
def kernel(coeffs: list[float], words: list[cudaq.pauli_word]):
    q = cudaq.qvector(n_qubits)

    # Start from |+>^N
    for i in range(n_qubits):
        h(q[i])

    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])



def cost(sampled_ops: list[cudaq.SpinOperator], **kwargs):
    full_coeffs = []
    full_words = []

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
        return handle, lambda res: res.get().expectation()
    else:
        return cudaq.observe(
            kernel, spin_ham, full_coeffs, full_words
        ).expectation()


# Configure GQE
cfg = get_default_config()
cfg.use_fabric_logging = False
logger = CSVLogger("gqe_h2_logs/gqe.csv")
cfg.fabric_logger = logger
cfg.save_trajectory = False
cfg.verbose = True

# Run GQE
minE, best_ops = src.GQEMTS.gqe(cost, op_pool, max_iters=5, ngates=3, config=cfg)

# Only print results from rank 0 when using MPI
if not args.mpi or cudaq.mpi.rank() == 0:
    print(f"Ground Energy = {minE}")
    print("Ansatz Ops")
    for idx in best_ops:
        # Get the first (and only) term since these are simple operators
        term = next(iter(op_pool[idx]))
        print(term.evaluate_coefficient().real, term.get_pauli_word(n_qubits))

if args.mpi:
    cudaq.mpi.finalize()
