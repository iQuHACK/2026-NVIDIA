#!/usr/bin/env python3
"""
CUDA-Q MLIR IR Parser v2.0

Clean, structured parser for CUDA-Q MLIR (Quake dialect).
Replaces regex-based approach with a more maintainable structure.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class GateType(Enum):
    """Quantum gate types."""
    # Single-qubit gates
    H = "h"
    X = "x"
    Y = "y"
    Z = "z"
    S = "s"
    T = "t"
    SDG = "sdg"
    TDG = "tdg"

    # Rotation gates (parametric)
    RX = "rx"
    RY = "ry"
    RZ = "rz"
    R1 = "r1"

    # Two-qubit gates
    SWAP = "swap"

    # Controlled single-qubit gates
    CH = "ch"
    CS = "cs"
    CT = "ct"
    CSDG = "csdg"
    CTDG = "ctdg"
    CX = "cx"
    CY = "cy"
    CZ = "cz"

    # Controlled rotation gates
    CRX = "crx"
    CRY = "cry"
    CRZ = "crz"
    CR1 = "cr1"

    # Multi-controlled gates
    CCX = "ccx"  # Toffoli


@dataclass
class Gate:
    """
    Represents a quantum gate operation.

    Attributes:
        gate_type: Type of gate
        targets: Target qubit indices
        controls: Control qubit indices (empty for non-controlled gates)
        params: Gate parameters (e.g., rotation angles)
        position: Position in circuit (0-indexed)
    """
    gate_type: GateType
    targets: List[int]
    controls: List[int]
    params: List[float]
    position: int

    @property
    def name(self) -> str:
        """Get gate name."""
        return self.gate_type.value

    @property
    def is_controlled(self) -> bool:
        """Check if gate is controlled."""
        return len(self.controls) > 0

    @property
    def is_parametric(self) -> bool:
        """Check if gate has parameters."""
        return len(self.params) > 0

    @property
    def num_qubits(self) -> int:
        """Total number of qubits involved."""
        return len(self.targets) + len(self.controls)

    def __repr__(self) -> str:
        parts = [f"Gate({self.name}"]
        parts.append(f"targets={self.targets}")
        if self.controls:
            parts.append(f"controls={self.controls}")
        if self.params:
            parts.append(f"params={[f'{p:.4f}' for p in self.params]}")
        parts.append(f"pos={self.position})")
        return ", ".join(parts)


@dataclass
class Circuit:
    """
    Represents a quantum circuit.

    Attributes:
        num_qubits: Number of qubits
        gates: List of gates in order
    """
    num_qubits: int
    gates: List[Gate]

    def __repr__(self) -> str:
        return f"Circuit(num_qubits={self.num_qubits}, gates={len(self.gates)})"


class MLIRParser:
    """
    Parser for CUDA-Q MLIR intermediate representation.

    Parses Quake dialect operations from MLIR string representation.
    """

    # MLIR operation patterns
    PATTERNS = {
        # Qubit allocation: quake.alloca !quake.veq<N>
        'alloca': re.compile(r'%(\w+)\s*=\s*quake\.alloca\s+!quake\.veq<(\d+)>'),

        # Qubit extraction: %q1 = quake.extract_ref %q[0]
        'extract': re.compile(r'%(\w+)\s*=\s*quake\.extract_ref\s+%(\w+)\[(\d+)\]'),

        # Dynamic qubit extraction: %q1 = quake.extract_ref %q[%i]
        'extract_dynamic': re.compile(r'%(\w+)\s*=\s*quake\.extract_ref\s+%(\w+)\[%(\w+)\]'),

        # Single-qubit gates: quake.h %q0
        'single_qubit': re.compile(r'quake\.(h|x|y|z|s|t|sdg|tdg)\s+%(\w+)'),

        # Rotation gates: quake.rx (%angle) %q0
        'rotation': re.compile(r'quake\.(rx|ry|rz|r1)\s*\(([^)]+)\)\s+%(\w+)'),

        # Controlled single-qubit gates: quake.h [%ctrl] %target
        'controlled_single': re.compile(r'quake\.(h|s|t|sdg|tdg|x|y|z)\s+\[([^\]]+)\]\s+%(\w+)'),

        # Controlled rotation gates: quake.rx (%angle) [%ctrl] %target
        'controlled_rotation': re.compile(r'quake\.(rx|ry|rz|r1)\s*\(([^)]+)\)\s+\[([^\]]+)\]\s+%(\w+)'),

        # Swap: quake.swap %q0, %q1
        'swap': re.compile(r'quake\.swap\s+%(\w+)\s*,\s*%(\w+)'),

        # Constants
        'const_f64': re.compile(r'%(\w+)\s*=\s*arith\.constant\s+([0-9.eE+-]+)\s*:\s*f64'),
        'const_i64': re.compile(r'%(\w+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*i64'),
    }

    def __init__(self):
        """Initialize parser."""
        self._qubit_map: Dict[str, int] = {}
        self._constants: Dict[str, float] = {}
        self._num_qubits: int = 0
        self._loop_vars: Dict[str, int] = {}  # Track loop iteration variables

    def parse(self, kernel) -> Circuit:
        """
        Parse a CUDA-Q kernel into a Circuit.

        Args:
            kernel: CUDA-Q kernel function (decorated with @cudaq.kernel)

        Returns:
            Circuit object with topology information

        Raises:
            ValueError: If MLIR cannot be parsed
        """
        # Get MLIR string representation
        mlir_str = str(kernel)

        # Reset state
        self._qubit_map = {}
        self._constants = {}
        self._num_qubits = 0
        self._loop_vars = {}

        # Parse in stages
        self._extract_constants(mlir_str)
        self._extract_qubits(mlir_str)
        gates = self._extract_gates_with_loops(mlir_str)

        return Circuit(num_qubits=self._num_qubits, gates=gates)

    def _extract_constants(self, mlir_str: str) -> None:
        """Extract constant values from MLIR."""
        # Extract floating-point constants
        for match in self.PATTERNS['const_f64'].finditer(mlir_str):
            var_name, value_str = match.groups()
            self._constants[var_name] = float(value_str)

        # Extract integer constants (for loop bounds, etc.)
        for match in self.PATTERNS['const_i64'].finditer(mlir_str):
            var_name, value_str = match.groups()
            self._constants[var_name] = int(value_str)

    def _extract_qubits(self, mlir_str: str) -> None:
        """Extract qubit allocation and create mapping."""
        # Find qubit vector allocation
        alloca_match = self.PATTERNS['alloca'].search(mlir_str)
        if not alloca_match:
            raise ValueError("No qubit allocation found in MLIR")

        vec_name, num_qubits_str = alloca_match.groups()
        self._num_qubits = int(num_qubits_str)

        # Build mapping from SSA values to qubit indices
        # Handle static extractions: %q1 = quake.extract_ref %q[0]
        for match in self.PATTERNS['extract'].finditer(mlir_str):
            qubit_var, vec_var, index_str = match.groups()
            if vec_var == vec_name:
                self._qubit_map[qubit_var] = int(index_str)

        # Handle dynamic extractions from loops: %q1 = quake.extract_ref %q[%i]
        # For dynamic extractions, we need to infer the index from context
        for match in self.PATTERNS['extract_dynamic'].finditer(mlir_str):
            qubit_var, vec_var, index_var = match.groups()
            if vec_var == vec_name:
                # Try to resolve the index variable from constants
                if index_var in self._constants:
                    self._qubit_map[qubit_var] = int(self._constants[index_var])
                # Otherwise, we'll need to handle it in gate extraction

    def _extract_gates_with_loops(self, mlir_str: str) -> List[Gate]:
        """Extract gate operations from MLIR, handling loops."""
        gates = []
        position = 0

        # Get the qubit vector name for dynamic extractions
        alloca_match = self.PATTERNS['alloca'].search(mlir_str)
        vec_name = alloca_match.group(1) if alloca_match else None

        # Find all loops in the MLIR
        # Captures: iter_var, start_var, end_var, loop_body, step_body
        loop_pattern = re.compile(
            r'%\w+\s*=\s*cc\.loop\s+while\s+\(\(%(\w+)\s*=\s*%(\w+)\)\s*->\s*\(i64\)\)\s*\{'
            r'.*?arith\.cmpi\s+slt,\s+%\w+,\s*%(\w+)\s*:.*?'
            r'\}\s*do\s*\{(.*?)\}\s*step\s*\{(.*?)\}',
            re.DOTALL
        )

        loops = list(loop_pattern.finditer(mlir_str))

        if not loops:
            # No loops found - use regular extraction
            return self._extract_gates(mlir_str)

        # Process MLIR with loops
        current_pos = 0

        for loop_match in loops:
            # Process gates before this loop
            before_loop = mlir_str[current_pos:loop_match.start()]
            for line in before_loop.split('\n'):
                line = line.strip()
                gate = self._try_parse_gate(line, position)
                if gate:
                    gates.append(gate)
                    position += 1

            # Extract and unroll this loop
            iter_var, start_var, end_var, loop_body, step_body = loop_match.groups()
            start = int(self._constants.get(start_var, 0))
            end = int(self._constants.get(end_var, 0))

            # Extract step value from step block: arith.addi %arg, %step_const
            step = 1  # default
            step_match = re.search(r'arith\.addi\s+%\w+,\s*%(\w+)', step_body)
            if step_match:
                step_var = step_match.group(1)
                step = int(self._constants.get(step_var, 1))

            for i in range(start, end, step):
                self._loop_vars[iter_var] = i
                temp_vars = {}  # Track temporary computed values in this iteration

                # Parse loop body for this iteration
                body_lines = loop_body.split('\n')
                for line in body_lines:
                    line = line.strip()

                    # Track arithmetic operations: %result = arith.addi %a, %b
                    arith_add = re.search(r'%(\w+)\s*=\s*arith\.addi\s+%(\w+),\s*%(\w+)', line)
                    if arith_add:
                        result_var, op1_var, op2_var = arith_add.groups()
                        val1 = self._loop_vars.get(op1_var, temp_vars.get(op1_var, self._constants.get(op1_var, 0)))
                        val2 = self._loop_vars.get(op2_var, temp_vars.get(op2_var, self._constants.get(op2_var, 0)))
                        temp_vars[result_var] = int(val1) + int(val2)

                    # Track multiplication: %result = arith.muli %a, %b
                    arith_mul = re.search(r'%(\w+)\s*=\s*arith\.muli\s+%(\w+),\s*%(\w+)', line)
                    if arith_mul:
                        result_var, op1_var, op2_var = arith_mul.groups()
                        val1 = self._loop_vars.get(op1_var, temp_vars.get(op1_var, self._constants.get(op1_var, 0)))
                        val2 = self._loop_vars.get(op2_var, temp_vars.get(op2_var, self._constants.get(op2_var, 0)))
                        temp_vars[result_var] = int(val1) * int(val2)

                    # Track casting: %result = cc.cast signed %val : (i64) -> f64
                    cast_match = re.search(r'%(\w+)\s*=\s*cc\.cast\s+signed\s+%(\w+)\s*:\s*\(i64\)\s*->\s*f64', line)
                    if cast_match:
                        result_var, val_var = cast_match.groups()
                        val = self._loop_vars.get(val_var, temp_vars.get(val_var, 0))
                        temp_vars[result_var] = float(val)

                    # Track f64 multiplication: %result = arith.mulf %a, %b
                    arith_mulf = re.search(r'%(\w+)\s*=\s*arith\.mulf\s+%(\w+),\s*%(\w+)', line)
                    if arith_mulf:
                        result_var, op1_var, op2_var = arith_mulf.groups()
                        val1 = temp_vars.get(op1_var, self._constants.get(op1_var, 0.0))
                        val2 = temp_vars.get(op2_var, self._constants.get(op2_var, 0.0))
                        temp_vars[result_var] = float(val1) * float(val2)

                    # Track loads: %result = cc.load %ptr
                    load_match = re.search(r'%(\w+)\s*=\s*cc\.load\s+%(\w+)', line)
                    if load_match:
                        result_var, ptr_var = load_match.groups()
                        if ptr_var in temp_vars:
                            temp_vars[result_var] = temp_vars[ptr_var]

                    # Track stores: cc.store %val, %ptr
                    store_match = re.search(r'cc\.store\s+%(\w+),\s*%(\w+)', line)
                    if store_match:
                        val_var, ptr_var = store_match.groups()
                        if val_var in temp_vars:
                            temp_vars[ptr_var] = temp_vars[val_var]
                        elif val_var in self._constants:
                            temp_vars[ptr_var] = self._constants[val_var]

                    # Handle dynamic qubit extraction
                    if vec_name:
                        dyn_extract = re.search(
                            r'%(\w+)\s*=\s*quake\.extract_ref\s+%' + vec_name + r'\[%(\w+)\]',
                            line
                        )
                        if dyn_extract:
                            qubit_var, idx_var = dyn_extract.groups()
                            # Resolve index from loop vars or temp vars
                            if idx_var == iter_var:
                                idx = i
                            elif idx_var in self._loop_vars:
                                idx = self._loop_vars[idx_var]
                            elif idx_var in temp_vars:
                                idx = temp_vars[idx_var]
                            else:
                                idx = self._constants.get(idx_var, 0)
                            self._qubit_map[qubit_var] = int(idx)

                    # Try to parse gate (temporarily add temp_vars to constants)
                    old_constants = self._constants.copy()
                    self._constants.update(temp_vars)
                    gate = self._try_parse_gate(line, position)
                    self._constants = old_constants  # Restore constants

                    if gate:
                        gates.append(gate)
                        position += 1

            self._loop_vars.clear()
            current_pos = loop_match.end()

        # Process gates after all loops
        after_loops = mlir_str[current_pos:]
        for line in after_loops.split('\n'):
            line = line.strip()
            gate = self._try_parse_gate(line, position)
            if gate:
                gates.append(gate)
                position += 1

        return gates

    def _extract_gates(self, mlir_str: str) -> List[Gate]:
        """Extract gate operations from MLIR (non-loop version)."""
        gates = []
        position = 0

        lines = mlir_str.split('\n')
        for line in lines:
            line = line.strip()
            gate = self._try_parse_gate(line, position)
            if gate:
                gates.append(gate)
                position += 1

        return gates

    def _try_parse_gate(self, line: str, position: int) -> Optional[Gate]:
        """Try to parse a gate from a line."""
        # Try to match each gate type (order matters - try controlled before single-qubit)
        if gate := self._parse_controlled_rotation_gate(line, position):
            return gate
        elif gate := self._parse_controlled_single_gate(line, position):
            return gate
        elif gate := self._parse_rotation_gate(line, position):
            return gate
        elif gate := self._parse_single_qubit_gate(line, position):
            return gate
        elif gate := self._parse_swap_gate(line, position):
            return gate
        return None

    def _parse_single_qubit_gate(self, line: str, position: int) -> Optional[Gate]:
        """Parse single-qubit gate."""
        match = self.PATTERNS['single_qubit'].search(line)
        if not match:
            return None

        gate_name, qubit_var = match.groups()

        # Map SSA variable to qubit index
        if qubit_var not in self._qubit_map:
            return None

        qubit_idx = self._qubit_map[qubit_var]

        # Map gate name to GateType
        try:
            gate_type = GateType(gate_name)
        except ValueError:
            return None

        return Gate(
            gate_type=gate_type,
            targets=[qubit_idx],
            controls=[],
            params=[],
            position=position
        )

    def _parse_rotation_gate(self, line: str, position: int) -> Optional[Gate]:
        """Parse rotation gate with parameter."""
        match = self.PATTERNS['rotation'].search(line)
        if not match:
            return None

        gate_name, param_ref, qubit_var = match.groups()

        # Resolve parameter
        param_ref = param_ref.strip('%')
        if param_ref not in self._constants:
            return None

        param_value = float(self._constants[param_ref])

        # Resolve qubit
        if qubit_var not in self._qubit_map:
            return None

        qubit_idx = self._qubit_map[qubit_var]

        try:
            gate_type = GateType(gate_name)
        except ValueError:
            return None

        return Gate(
            gate_type=gate_type,
            targets=[qubit_idx],
            controls=[],
            params=[param_value],
            position=position
        )

    def _parse_controlled_single_gate(self, line: str, position: int) -> Optional[Gate]:
        """Parse controlled single-qubit gate (e.g., quake.h [%ctrl] %target)."""
        match = self.PATTERNS['controlled_single'].search(line)
        if not match:
            return None

        gate_name, control_str, target_var = match.groups()

        # Parse control qubits (can be multiple)
        control_vars = [c.strip().strip('%') for c in control_str.split(',')]
        control_indices = []
        for ctrl_var in control_vars:
            if ctrl_var in self._qubit_map:
                control_indices.append(self._qubit_map[ctrl_var])
            else:
                return None  # Invalid control qubit

        # Parse target qubit
        if target_var not in self._qubit_map:
            return None

        target_idx = self._qubit_map[target_var]

        # Determine gate type based on number of controls and gate name
        if len(control_indices) == 1:
            # Single-controlled gates
            gate_type_map = {
                'h': GateType.CH,
                's': GateType.CS,
                't': GateType.CT,
                'sdg': GateType.CSDG,
                'tdg': GateType.CTDG,
                'x': GateType.CX,
                'y': GateType.CY,
                'z': GateType.CZ,
            }
            gate_type = gate_type_map.get(gate_name)
            if not gate_type:
                return None
        elif len(control_indices) == 2 and gate_name == 'x':
            gate_type = GateType.CCX  # Toffoli
        else:
            return None  # Unsupported controlled gate

        return Gate(
            gate_type=gate_type,
            targets=[target_idx],
            controls=control_indices,
            params=[],
            position=position
        )

    def _parse_controlled_rotation_gate(self, line: str, position: int) -> Optional[Gate]:
        """Parse controlled rotation gate (e.g., quake.rx (%angle) [%ctrl] %target)."""
        match = self.PATTERNS['controlled_rotation'].search(line)
        if not match:
            return None

        gate_name, param_ref, control_str, target_var = match.groups()

        # Resolve parameter
        param_ref = param_ref.strip('%')
        if param_ref not in self._constants:
            return None

        param_value = float(self._constants[param_ref])

        # Parse control qubits (can be multiple, but we only support single control for now)
        control_vars = [c.strip().strip('%') for c in control_str.split(',')]
        control_indices = []
        for ctrl_var in control_vars:
            if ctrl_var in self._qubit_map:
                control_indices.append(self._qubit_map[ctrl_var])
            else:
                return None  # Invalid control qubit

        # Parse target qubit
        if target_var not in self._qubit_map:
            return None

        target_idx = self._qubit_map[target_var]

        # Only support single-controlled rotation gates
        if len(control_indices) != 1:
            return None

        # Map to controlled rotation gate type
        gate_type_map = {
            'rx': GateType.CRX,
            'ry': GateType.CRY,
            'rz': GateType.CRZ,
            'r1': GateType.CR1,
        }
        gate_type = gate_type_map.get(gate_name)
        if not gate_type:
            return None

        return Gate(
            gate_type=gate_type,
            targets=[target_idx],
            controls=control_indices,
            params=[param_value],
            position=position
        )

    def _parse_swap_gate(self, line: str, position: int) -> Optional[Gate]:
        """Parse SWAP gate."""
        match = self.PATTERNS['swap'].search(line)
        if not match:
            return None

        qubit1_var, qubit2_var = match.groups()

        # Resolve both qubits
        if qubit1_var not in self._qubit_map or qubit2_var not in self._qubit_map:
            return None

        qubit1_idx = self._qubit_map[qubit1_var]
        qubit2_idx = self._qubit_map[qubit2_var]

        return Gate(
            gate_type=GateType.SWAP,
            targets=[qubit1_idx, qubit2_idx],
            controls=[],
            params=[],
            position=position
        )
