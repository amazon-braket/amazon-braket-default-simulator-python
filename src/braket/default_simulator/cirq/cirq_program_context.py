from typing import List, Tuple

import cirq
import numpy as np
from braket.ir.jaqcd.program_v1 import Results
from cirq import (
    Circuit,
    bit_flip,
    PhaseFlipChannel,
    DepolarizingChannel,
    AmplitudeDampingChannel,
    GeneralizedAmplitudeDampingChannel,
    PhaseDampingChannel,
)

from braket.default_simulator.openqasm.parser.braket_pragmas import (
    AbstractBraketPragmaNodeVisitor,
    parse_braket_pragma,
)
from braket.default_simulator.openqasm.parser.generated.BraketPragmasParser import (
    BraketPragmasParser,
)
from braket.default_simulator.openqasm.program_context import AbstractProgramContext

# cirq.XX, cirq.YY, and cirq.ZZ gates are not the same as Braket gates
CIRQ_GATES = {
    "i": cirq.I,
    "h": cirq.H,
    "x": cirq.X,
    "y": cirq.Y,
    "z": cirq.Z,
    "cnot": cirq.CNOT,
    "cz": cirq.CZ,
    "s": cirq.S,
    "t": cirq.T,
    "cphaseshift": cirq.cphase,
    "rx": cirq.rx,
    "ry": cirq.ry,
    "rz": cirq.rz,
    "swap": cirq.SWAP,
    "iswap": cirq.ISWAP,
    "ccnot": cirq.CCNOT,
    "cswap": cirq.CSWAP,
    "ccz": cirq.CCZ,
    "ccx": cirq.CCX,
    "measure": cirq.MeasurementGate,
}


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
        qubits = [cirq.LineQubit(int(qubit)) for qubit in target]
        target_qubits = qubits[len(ctrl_modifiers) :]
        control_qubits = qubits[: len(ctrl_modifiers)]

        if params:
            gate = CIRQ_GATES[gate_name](*params).on(*target_qubits)
        else:
            gate = CIRQ_GATES[gate_name].on(*target_qubits)
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
        raise NotImplementedError

    def add_noise_instruction(self, noise):
        """Add a noise instruction the circuit"""
        self.circuit.append(noise)

    def add_result(self, result: Results) -> None:
        """Add a result type to the circuit"""
        raise NotImplementedError

    def parse_pragma(self, pragma_body: str):
        """Parse pragma"""
        return parse_braket_pragma(pragma_body, CirqBraketPragmaNodeVisitor(self.qubit_mapping))


class CirqBraketPragmaNodeVisitor(AbstractBraketPragmaNodeVisitor):
    def visitNoise(self, ctx: BraketPragmasParser.NoiseContext):
        target = self.visit(ctx.target)
        qubits = [cirq.LineQubit(int(qubit)) for qubit in target]
        probabilities = self.visit(ctx.probabilities())
        noise_instruction = ctx.noiseInstructionName().getText()
        one_prob_noise_map = {
            "bit_flip": bit_flip,
            "phase_flip": PhaseFlipChannel,
            "depolarizing": DepolarizingChannel,
            "amplitude_damping": AmplitudeDampingChannel,
            "generalized_amplitude_damping": GeneralizedAmplitudeDampingChannel,
            "phase_damping": PhaseDampingChannel,
        }
        if noise_instruction in one_prob_noise_map:
            if noise_instruction == "generalized_amplitude_damping":
                return one_prob_noise_map[noise_instruction](*probabilities[::-1]).on(*qubits)
            return one_prob_noise_map[noise_instruction](*probabilities).on(*qubits)
        else:
            raise NotImplementedError

    def visitKraus(self, ctx: BraketPragmasParser.KrausContext):
        raise NotImplementedError
