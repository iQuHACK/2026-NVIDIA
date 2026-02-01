#!/usr/bin/env python3
"""
Converter for PyTorch tensor network representation.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import opt_einsum
from .ir_parser import Circuit, Gate, GateType
from .simulator import get_gate_matrix


def to_tensors(circuit: Circuit) -> Tuple[List[torch.Tensor], List[List[int]]]:
    """
    Convert circuit to list of tensors and their qubit indices.

    Args:
        circuit: Circuit to convert

    Returns:
        Tuple of (tensors, indices) where:
        - tensors: List of gate tensors
        - indices: List of qubit indices for each tensor

    Example:
        ```python
        circuit = parse(my_kernel)
        tensors, indices = to_tensors(circuit)
        ```
    """
    tensors = []
    qubit_indices = []

    for gate in circuit.gates:
        # Get gate matrix
        matrix = get_gate_matrix(gate)

        # Convert to torch tensor
        tensor = torch.from_numpy(matrix).to(torch.complex128)

        # Determine qubit indices
        if gate.is_controlled:
            qubits = gate.controls + gate.targets
        else:
            qubits = gate.targets

        # Reshape tensor based on number of qubits
        if len(qubits) == 1:
            tensor = tensor.reshape(2, 2)
        elif len(qubits) == 2:
            tensor = tensor.reshape(2, 2, 2, 2)
        elif len(qubits) == 3:
            tensor = tensor.reshape([2] * 6)

        tensors.append(tensor)
        qubit_indices.append(qubits)

    return tensors, qubit_indices


def contract(
    circuit: Circuit,
    initial_state: Optional[torch.Tensor] = None,
    optimize: str = 'auto'
) -> torch.Tensor:
    """
    Contract the tensor network for a circuit.

    Args:
        circuit: Circuit to contract
        initial_state: Initial state tensor (defaults to |0...0⟩)
        optimize: Optimization strategy for contraction ('auto', 'greedy', 'optimal')

    Returns:
        Final state vector as torch tensor

    Example:
        ```python
        circuit = parse(my_kernel)
        state = contract(circuit)
        amplitudes = state.abs() ** 2
        ```
    """
    n = circuit.num_qubits

    # Initialize state
    if initial_state is None:
        state = torch.zeros(2**n, dtype=torch.complex128)
        state[0] = 1.0
    else:
        state = initial_state.clone()

    # Reshape to tensor form
    state = state.reshape([2] * n)

    # Get tensors and indices
    tensors, indices = to_tensors(circuit)

    # Build einsum expression
    # This is a simplified version - full implementation would use opt_einsum
    for tensor, qubits in zip(tensors, indices):
        if len(qubits) == 1:
            q = qubits[0]
            # Contract single qubit
            state = torch.tensordot(tensor, state, dims=([1], [q]))
            state = torch.moveaxis(state, 0, q)
        elif len(qubits) == 2:
            # Two qubit gate - simplified contraction
            q0, q1 = qubits
            state = torch.tensordot(tensor, state, dims=([2, 3], [q0, q1]))
            # Rearrange axes
            axes = list(range(len(state.shape)))
            new_axes = axes[2:]
            new_axes.insert(q0, 0)
            new_axes.insert(q1, 1)
            state = state.permute(new_axes)

    # Flatten to vector
    return state.reshape(-1)


def to_einsum(
    circuit: Circuit,
    backend: str = 'numpy'
) -> Tuple[str, List, Dict]:
    """
    Convert circuit to einsum expression for use with cuQuantum/opt_einsum.

    This generates a complete einsum expression string that represents the entire
    circuit, along with operands (gate tensors). Parameterized gates use placeholder
    tensors that can be updated later using update_parameters().

    Args:
        circuit: Parsed CUDA-Q circuit
        backend: 'numpy' or 'torch' for tensor type

    Returns:
        Tuple of (expression, operands, metadata) where:
        - expression: Einsum string (e.g., "ab,bc,cd->ad")
        - operands: List of gate tensors (placeholders for parameterized gates)
        - metadata: Dictionary containing:
            - 'num_qubits': Number of qubits
            - 'parameter_gates': List of (operand_idx, gate, param_info)
            - 'qubit_indices': Final qubit index labels
            - 'gate_info': List of (gate_type, qubits, position)

    Example:
        ```python
        circuit = parse(my_kernel)
        expr, ops, meta = to_einsum(circuit)

        # Use with cuQuantum
        from cuquantum import Network
        network = Network(expr, *ops)
        result = network.contract()

        # Or use with opt_einsum
        import opt_einsum
        result = opt_einsum.contract(expr, *ops, initial_state)
        ```
    """
    # Step 1: Initialize index tracking using opt_einsum.get_symbol()
    qubit_indices = [opt_einsum.get_symbol(i) for i in range(circuit.num_qubits)]
    symbol_counter = circuit.num_qubits

    gate_expressions = []
    operands = []
    parameter_gates = []
    gate_info = []

    # Step 2: Build expression for each gate
    for gate_idx, gate in enumerate(circuit.gates):
        # Get qubits involved (controls + targets)
        all_qubits = gate.controls + gate.targets
        num_gate_qubits = len(all_qubits)

        # Build gate expression based on number of qubits
        if num_gate_qubits == 1:
            # Single-qubit gate: "da" (out_idx + in_idx)
            q = all_qubits[0]
            in_idx = qubit_indices[q]
            out_idx = opt_einsum.get_symbol(symbol_counter)
            gate_expr = f"{out_idx}{in_idx}"
            qubit_indices[q] = out_idx
            symbol_counter += 1

        elif num_gate_qubits == 2:
            # Two-qubit gate: "dcab" (out_q0 + out_q1 + in_q0 + in_q1)
            q0, q1 = all_qubits
            in_q0 = qubit_indices[q0]
            in_q1 = qubit_indices[q1]
            out_q0 = opt_einsum.get_symbol(symbol_counter)
            out_q1 = opt_einsum.get_symbol(symbol_counter + 1)
            gate_expr = f"{out_q0}{out_q1}{in_q0}{in_q1}"
            qubit_indices[q0] = out_q0
            qubit_indices[q1] = out_q1
            symbol_counter += 2

        elif num_gate_qubits == 3:
            # Three-qubit gate (e.g., CCX): 6 indices
            q0, q1, q2 = all_qubits
            in_q0 = qubit_indices[q0]
            in_q1 = qubit_indices[q1]
            in_q2 = qubit_indices[q2]
            out_q0 = opt_einsum.get_symbol(symbol_counter)
            out_q1 = opt_einsum.get_symbol(symbol_counter + 1)
            out_q2 = opt_einsum.get_symbol(symbol_counter + 2)
            gate_expr = f"{out_q0}{out_q1}{out_q2}{in_q0}{in_q1}{in_q2}"
            qubit_indices[q0] = out_q0
            qubit_indices[q1] = out_q1
            qubit_indices[q2] = out_q2
            symbol_counter += 3

        else:
            raise ValueError(f"Gates with {num_gate_qubits} qubits not supported")

        gate_expressions.append(gate_expr)

        # Step 3: Prepare operand tensor
        if gate.is_parametric:
            # Placeholder tensor for parameterized gates
            shape = (2,) * (2 * num_gate_qubits)
            if backend == 'torch':
                tensor = torch.zeros(shape, dtype=torch.complex128)
            else:
                tensor = np.zeros(shape, dtype=np.complex128)

            # Record parameter gate info for later update
            parameter_gates.append((gate_idx, gate, {
                'gate_type': gate.gate_type.value,
                'params': gate.params,
                'controls': gate.controls,
                'targets': gate.targets,
                'num_qubits': num_gate_qubits
            }))
        else:
            # Fixed gate matrix
            matrix = get_gate_matrix(gate)
            if backend == 'torch':
                tensor = torch.from_numpy(matrix).to(torch.complex128)
            else:
                tensor = matrix

            # Reshape to tensor format
            tensor = tensor.reshape((2,) * (2 * num_gate_qubits))

        operands.append(tensor)

        # Record gate info
        gate_info.append({
            'gate_type': gate.gate_type.value,
            'qubits': all_qubits,
            'position': gate.position
        })

    # Step 4: Assemble complete expression
    final_indices = ''.join(qubit_indices)
    expression = ','.join(gate_expressions) + '->' + final_indices

    # Step 5: Prepare metadata
    metadata = {
        'num_qubits': circuit.num_qubits,
        'parameter_gates': parameter_gates,
        'qubit_indices': list(qubit_indices),
        'gate_info': gate_info
    }

    return expression, operands, metadata


def update_parameters(
    operands: List,
    param_values: Union[List[float], np.ndarray, torch.Tensor],
    metadata: Dict,
    backend: str = 'numpy'
) -> List:
    """
    Update parameterized gate tensors with actual rotation matrix values.

    This function replaces placeholder tensors in the operands list with
    computed rotation matrices based on the provided parameter values.

    Args:
        operands: Original operand list from to_einsum() (with placeholders)
        param_values: Parameter values in the same order as metadata['parameter_gates']
        metadata: Metadata dictionary from to_einsum()
        backend: 'numpy' or 'torch'

    Returns:
        Updated operands list with computed rotation matrices

    Example:
        ```python
        # Get expression with placeholders
        expr, ops, meta = to_einsum(circuit)

        # Update with actual parameter values
        params = [0.5, 1.2, 0.8]  # theta values for RX, RY, etc.
        updated_ops = update_parameters(ops, params, meta)

        # Now contract with real values
        result = opt_einsum.contract(expr, *updated_ops, initial_state)
        ```
    """
    # Create a copy of operands to avoid modifying original
    updated_ops = list(operands)

    # Convert param_values to appropriate format
    if isinstance(param_values, torch.Tensor):
        param_values = param_values.cpu().numpy()
    elif isinstance(param_values, list):
        param_values = np.array(param_values)

    # Iterate through parameter gates and update operands
    param_idx = 0
    for operand_idx, gate, param_info in metadata['parameter_gates']:
        # Get parameter values for this gate
        gate_params = gate.params
        num_params = len(gate_params)

        if param_idx + num_params > len(param_values):
            raise ValueError(
                f"Not enough parameter values provided. "
                f"Expected at least {param_idx + num_params}, got {len(param_values)}"
            )

        # Get the parameter values
        theta_values = param_values[param_idx:param_idx + num_params]
        param_idx += num_params

        # Compute rotation matrix based on gate type
        gate_type = param_info['gate_type']
        num_qubits = param_info['num_qubits']

        # Get the rotation matrix
        if num_qubits == 1:
            # Single-qubit rotation gates
            theta = theta_values[0]
            if gate_type in ['rx', 'crx']:
                c = np.cos(theta / 2)
                s = np.sin(theta / 2)
                matrix = np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex128)
            elif gate_type in ['ry', 'cry']:
                c = np.cos(theta / 2)
                s = np.sin(theta / 2)
                matrix = np.array([[c, -s], [s, c]], dtype=np.complex128)
            elif gate_type in ['rz', 'crz']:
                matrix = np.array([[np.exp(-1j*theta/2), 0],
                                   [0, np.exp(1j*theta/2)]], dtype=np.complex128)
            elif gate_type in ['r1', 'cr1']:
                matrix = np.array([[1, 0], [0, np.exp(1j*theta)]], dtype=np.complex128)
            else:
                raise ValueError(f"Unknown parametric gate type: {gate_type}")

        elif num_qubits == 2:
            # Controlled rotation gates
            theta = theta_values[0]
            I = np.eye(2, dtype=np.complex128)

            if gate_type == 'crx':
                c = np.cos(theta / 2)
                s = np.sin(theta / 2)
                RX = np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex128)
                matrix = np.kron(np.array([[1, 0], [0, 0]]), I) + \
                         np.kron(np.array([[0, 0], [0, 1]]), RX)
            elif gate_type == 'cry':
                c = np.cos(theta / 2)
                s = np.sin(theta / 2)
                RY = np.array([[c, -s], [s, c]], dtype=np.complex128)
                matrix = np.kron(np.array([[1, 0], [0, 0]]), I) + \
                         np.kron(np.array([[0, 0], [0, 1]]), RY)
            elif gate_type == 'crz':
                RZ = np.array([[np.exp(-1j*theta/2), 0],
                               [0, np.exp(1j*theta/2)]], dtype=np.complex128)
                matrix = np.kron(np.array([[1, 0], [0, 0]]), I) + \
                         np.kron(np.array([[0, 0], [0, 1]]), RZ)
            elif gate_type == 'cr1':
                R1 = np.array([[1, 0], [0, np.exp(1j*theta)]], dtype=np.complex128)
                matrix = np.kron(np.array([[1, 0], [0, 0]]), I) + \
                         np.kron(np.array([[0, 0], [0, 1]]), R1)
            else:
                raise ValueError(f"Unknown controlled parametric gate type: {gate_type}")
        else:
            raise ValueError(f"Parametric gates with {num_qubits} qubits not supported")

        # Reshape matrix to tensor format
        tensor = matrix.reshape((2,) * (2 * num_qubits))

        # Convert to appropriate backend
        if backend == 'torch':
            tensor = torch.from_numpy(tensor).to(torch.complex128)

        # Update operand
        updated_ops[operand_idx] = tensor

    return updated_ops
