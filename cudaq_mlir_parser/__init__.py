#!/usr/bin/env python3
"""
CUDA-Q MLIR Parser v2.0

Simple, clean API for extracting quantum circuit topology from CUDA-Q kernels
and simulating them using tensor network contraction.

Basic Usage:
    ```python
    import cudaq
    from cudaq_mlir_parser import parse, simulate

    @cudaq.kernel
    def bell_state():
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])

    # Parse circuit
    circuit = parse(bell_state)
    print(f"Circuit has {circuit.num_qubits} qubits and {len(circuit.gates)} gates")

    # Simulate and get state vector
    state_vector = simulate(circuit)
    print(f"Final state: {state_vector}")
    ```
"""

from .ir_parser import MLIRParser, Circuit, Gate, GateType
from .simulator import simulate
from .converter import to_tensors, contract, to_einsum, update_parameters

__version__ = "2.0.0"

__all__ = [
    "parse",
    "simulate",
    "to_tensors",
    "contract",
    "to_einsum",
    "update_parameters",
    "Circuit",
    "Gate",
    "GateType",
]

# Global parser instance
_parser = MLIRParser()


def parse(kernel) -> Circuit:
    """
    Parse a CUDA-Q kernel into a Circuit object.

    Args:
        kernel: CUDA-Q kernel function (decorated with @cudaq.kernel)

    Returns:
        Circuit object containing topology information

    Example:
        ```python
        @cudaq.kernel
        def my_circuit():
            q = cudaq.qvector(3)
            h(q[0])
            cx(q[0], q[1])
            rx(1.57, q[2])

        circuit = parse(my_circuit)
        print(circuit)  # Circuit(num_qubits=3, gates=3)
        ```
    """
    return _parser.parse(kernel)
