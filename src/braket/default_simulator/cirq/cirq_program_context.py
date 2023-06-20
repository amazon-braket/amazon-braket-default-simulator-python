from functools import singledispatch
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

from braket.default_simulator.noise_operations import (
    BitFlip,
    PhaseFlip,
    GeneralizedAmplitudeDamping,
    PhaseDamping,
    AmplitudeDamping,
    Depolarizing,
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


@singledispatch
def cirq_gate_to_instruciton(noise):
    raise TypeError(f"Operation {type(noise).__name__} not supported")


@cirq_gate_to_instruciton.register(BitFlip)
def _(noise):
    qubits = CirqProgramContext.get_cirq_qubits(noise.targets)
    return bit_flip(noise.probability).on(*qubits)


@cirq_gate_to_instruciton.register(PhaseFlip)
def _(noise):
    qubits = CirqProgramContext.get_cirq_qubits(noise.targets)
    return PhaseFlipChannel(noise.probability).on(*qubits)


@cirq_gate_to_instruciton.register(Depolarizing)
def _(noise):
    qubits = CirqProgramContext.get_cirq_qubits(noise.targets)
    return DepolarizingChannel(noise.probability).on(*qubits)


@cirq_gate_to_instruciton.register(AmplitudeDamping)
def _(noise):
    qubits = CirqProgramContext.get_cirq_qubits(noise.targets)
    return AmplitudeDampingChannel(noise.gamma).on(*qubits)


@cirq_gate_to_instruciton.register(GeneralizedAmplitudeDamping)
def _(noise):
    qubits = CirqProgramContext.get_cirq_qubits(noise.targets)
    return GeneralizedAmplitudeDampingChannel(noise.probability, noise.gamma).on(*qubits)


@cirq_gate_to_instruciton.register(PhaseDamping)
def _(noise):
    qubits = CirqProgramContext.get_cirq_qubits(noise.targets)
    return PhaseDampingChannel(noise.gamma).on(*qubits)


class CirqProgramContext(AbstractProgramContext):
    def __init__(self):
        super().__init__(Circuit())

    @classmethod
    def get_cirq_qubits(cls, qubits: Tuple[int]):
        return [cirq.LineQubit(int(qubit)) for qubit in qubits]

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
        qubits = self.get_cirq_qubits(target)
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
        qubits = self.get_cirq_qubits(target)
        instruction = cirq.MatrixGate(unitary).on(*qubits)
        self.circuit.append(instruction)

    def add_noise_instruction(self, noise):
        """Add a noise instruction the circuit"""
        instruction = cirq_gate_to_instruciton(noise)
        self.circuit.append(instruction)

    def add_result(self, result: Results) -> None:
        """Add a result type to the circuit"""
        raise NotImplementedError
