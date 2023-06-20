from typing import List, Tuple

import cirq
import numpy as np
from braket.ir.jaqcd.program_v1 import Results
from cirq import (
    Circuit,
)

from braket.default_simulator.cirq.translation import (
    CIRQ_GATES,
    cirq_gate_to_instruction,
    get_cirq_qubits,
)
from braket.default_simulator.openqasm.program_context import AbstractProgramContext


class CirqProgramContext(AbstractProgramContext):
    def __init__(self):
        super().__init__(Circuit())

    def is_builtin_gate(self, name: str) -> bool:
        user_defined_gate = self.is_user_defined_gate(name)
        return name in CIRQ_GATES and not user_defined_gate

    def add_phase_instruction(self, target: Tuple[int], phase_value: int):
        raise NotImplementedError

    def _gate_accepts_parameters(self, gate_class):
        return hasattr(gate_class, "parameters") and len(gate_class.parameters) > 0

    def add_gate_instruction(
        self, gate_name: str, target: Tuple[int], params, ctrl_modifiers: List[int], power: int
    ):
        qubits = get_cirq_qubits(target)
        target_cirq_qubits = qubits[len(ctrl_modifiers) :]
        control_qubits = qubits[: len(ctrl_modifiers)]

        if params:
            gate = CIRQ_GATES[gate_name](*params).on(*target_cirq_qubits)
        else:
            gate = CIRQ_GATES[gate_name].on(*target_cirq_qubits)
        ctrl_modifiers = [bit ^ 1 for bit in ctrl_modifiers]
        gate = gate.controlled_by(*control_qubits, control_values=ctrl_modifiers)
        gate = gate**power

        self.circuit.append(gate)

    def add_custom_unitary(
        self,
        unitary: np.ndarray,
        target: Tuple[int],
    ) -> None:
        """Add a custom Unitary instruction to the circuit"""
        qubits = get_cirq_qubits(target)
        instruction = cirq.MatrixGate(unitary).on(*qubits)
        self.circuit.append(instruction)

    def add_noise_instruction(self, noise):
        """Add a noise instruction the circuit"""
        instruction = cirq_gate_to_instruction(noise)
        self.circuit.append(instruction)

    def add_result(self, result: Results) -> None:
        """Add a result type to the circuit"""
        raise NotImplementedError
