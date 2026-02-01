#!/usr/bin/env python3
"""
Quantum circuit simulator using tensor network contraction.
"""

import torch
import numpy as np
from typing import Optional, Union
from .ir_parser import Circuit, Gate, GateType


# Standard gate matrices
GATES = {
    # Single-qubit gates
    GateType.H: (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128),
    GateType.X: np.array([[0, 1], [1, 0]], dtype=np.complex128),
    GateType.Y: np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    GateType.Z: np.array([[1, 0], [0, -1]], dtype=np.complex128),
    GateType.S: np.array([[1, 0], [0, 1j]], dtype=np.complex128),
    GateType.T: np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=np.complex128),
    GateType.SDG: np.array([[1, 0], [0, -1j]], dtype=np.complex128),
    GateType.TDG: np.array([[1, 0], [0, np.exp(-1j*np.pi/4)]], dtype=np.complex128),

    # Two-qubit gates
    GateType.SWAP: np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]], dtype=np.complex128),

    # Controlled single-qubit gates (controlled on |1⟩)
    GateType.CH: (1/np.sqrt(2)) * np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 1],
                                             [0, 0, 1, -1]], dtype=np.complex128),
    GateType.CS: np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1j]], dtype=np.complex128),
    GateType.CT: np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, np.exp(1j*np.pi/4)]], dtype=np.complex128),
    GateType.CSDG: np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, -1j]], dtype=np.complex128),
    GateType.CTDG: np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, np.exp(-1j*np.pi/4)]], dtype=np.complex128),
    GateType.CX: np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]], dtype=np.complex128),
    GateType.CY: np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, -1j],
                            [0, 0, 1j, 0]], dtype=np.complex128),
    GateType.CZ: np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, -1]], dtype=np.complex128),

    # Multi-controlled gates
    GateType.CCX: np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.complex128),
}


def rx_matrix(theta: float) -> np.ndarray:
    """RX rotation gate."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex128)


def ry_matrix(theta: float) -> np.ndarray:
    """RY rotation gate."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz_matrix(theta: float) -> np.ndarray:
    """RZ rotation gate."""
    return np.array([[np.exp(-1j*theta/2), 0],
                      [0, np.exp(1j*theta/2)]], dtype=np.complex128)


def r1_matrix(theta: float) -> np.ndarray:
    """R1 (phase) gate."""
    return np.array([[1, 0], [0, np.exp(1j*theta)]], dtype=np.complex128)


def crx_matrix(theta: float) -> np.ndarray:
    """Controlled-RX rotation gate."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, c, -1j*s],
                      [0, 0, -1j*s, c]], dtype=np.complex128)


def cry_matrix(theta: float) -> np.ndarray:
    """Controlled-RY rotation gate."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, c, -s],
                      [0, 0, s, c]], dtype=np.complex128)


def crz_matrix(theta: float) -> np.ndarray:
    """Controlled-RZ rotation gate."""
    return np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, np.exp(-1j*theta/2), 0],
                      [0, 0, 0, np.exp(1j*theta/2)]], dtype=np.complex128)


def cr1_matrix(theta: float) -> np.ndarray:
    """Controlled-R1 (controlled phase) gate."""
    return np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, np.exp(1j*theta)]], dtype=np.complex128)


def get_gate_matrix(gate: Gate) -> np.ndarray:
    """Get the matrix representation of a gate."""
    # Rotation gates
    if gate.gate_type == GateType.RX:
        return rx_matrix(gate.params[0])
    elif gate.gate_type == GateType.RY:
        return ry_matrix(gate.params[0])
    elif gate.gate_type == GateType.RZ:
        return rz_matrix(gate.params[0])
    elif gate.gate_type == GateType.R1:
        return r1_matrix(gate.params[0])
    # Controlled rotation gates
    elif gate.gate_type == GateType.CRX:
        return crx_matrix(gate.params[0])
    elif gate.gate_type == GateType.CRY:
        return cry_matrix(gate.params[0])
    elif gate.gate_type == GateType.CRZ:
        return crz_matrix(gate.params[0])
    elif gate.gate_type == GateType.CR1:
        return cr1_matrix(gate.params[0])
    else:
        return GATES[gate.gate_type]


def simulate(
    circuit: Circuit,
    initial_state: Optional[np.ndarray] = None,
    backend: str = 'numpy'
) -> np.ndarray:
    """
    Simulate a quantum circuit using tensor network contraction.

    Args:
        circuit: Circuit to simulate
        initial_state: Initial state vector (defaults to |0...0⟩)
        backend: Backend to use ('numpy' or 'torch')

    Returns:
        Final state vector as numpy array

    Example:
        ```python
        circuit = parse(my_kernel)
        state = simulate(circuit)
        print(f"Amplitude of |00⟩: {state[0]}")
        ```
    """
    if backend == 'torch':
        return _simulate_torch(circuit, initial_state)
    else:
        return _simulate_numpy(circuit, initial_state)


def _simulate_numpy(
    circuit: Circuit,
    initial_state: Optional[np.ndarray] = None
) -> np.ndarray:
    """Simulate using numpy (statevector method)."""
    n = circuit.num_qubits

    # Initialize state
    if initial_state is None:
        state = np.zeros(2**n, dtype=np.complex128)
        state[0] = 1.0  # |0...0⟩
    else:
        state = initial_state.copy()

    # Apply gates sequentially
    for gate in circuit.gates:
        state = _apply_gate(state, gate, n)

    return state


def _apply_gate(state: np.ndarray, gate: Gate, num_qubits: int) -> np.ndarray:
    """Apply a gate to the state vector."""
    gate_matrix = get_gate_matrix(gate)

    # Determine which qubits the gate acts on
    if gate.is_controlled:
        qubits = gate.controls + gate.targets
    else:
        qubits = gate.targets

    # Reshape state into tensor form
    state_tensor = state.reshape([2] * num_qubits)

    # Apply gate using tensor contraction
    # Convert gate matrix to appropriate tensor shape
    if len(qubits) == 1:
        # Single qubit gate
        gate_tensor = gate_matrix.reshape(2, 2)
        state_tensor = np.tensordot(gate_tensor, state_tensor, axes=([1], [qubits[0]]))
        # Move axis back to original position
        state_tensor = np.moveaxis(state_tensor, 0, qubits[0])

    elif len(qubits) == 2:
        # Two qubit gate
        gate_tensor = gate_matrix.reshape(2, 2, 2, 2)
        # Contract with both qubits
        state_tensor = np.tensordot(gate_tensor, state_tensor, axes=([2, 3], [qubits[0], qubits[1]]))
        # Rearrange axes
        axes = list(range(len(state_tensor.shape)))
        new_axes = axes[2:]
        new_axes.insert(qubits[0], 0)
        new_axes.insert(qubits[1], 1)
        state_tensor = np.transpose(state_tensor, new_axes)

    elif len(qubits) == 3:
        # Three qubit gate (Toffoli)
        gate_tensor = gate_matrix.reshape([2] * 6)
        state_tensor = np.tensordot(gate_tensor, state_tensor, axes=([3, 4, 5], qubits))
        # Rearrange axes
        axes = list(range(len(state_tensor.shape)))
        new_axes = axes[3:]
        for i, q in enumerate(qubits):
            new_axes.insert(q, i)
        state_tensor = np.transpose(state_tensor, new_axes)

    # Flatten back to state vector
    return state_tensor.reshape(-1)


def _simulate_torch(
    circuit: Circuit,
    initial_state: Optional[np.ndarray] = None
) -> np.ndarray:
    """Simulate using PyTorch (for GPU acceleration)."""
    # TODO: Implement torch-based simulation
    # For now, fall back to numpy
    return _simulate_numpy(circuit, initial_state)
