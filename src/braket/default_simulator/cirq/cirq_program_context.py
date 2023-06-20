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
def cirq_gate_to_instruciton(gate, probabilities, gamma, qubits):
    raise TypeError(f"Operation {type(gate).__name__} not supported")


@cirq_gate_to_instruciton.register(BitFlip)
def _(gate, probabilities, gamma, qubits):
    return bit_flip(*probabilities, *gamma).on(*qubits)


@cirq_gate_to_instruciton.register(PhaseFlip)
def _(gate, probabilities, gamma, qubits):
    return PhaseFlipChannel(*probabilities, *gamma).on(*qubits)


@cirq_gate_to_instruciton.register(Depolarizing)
def _(gate, probabilities, gamma, qubits):
    return DepolarizingChannel(*probabilities, *gamma).on(*qubits)


@cirq_gate_to_instruciton.register(AmplitudeDamping)
def _(gate, probabilities, gamma, qubits):
    return AmplitudeDampingChannel(*probabilities, *gamma).on(*qubits)


@cirq_gate_to_instruciton.register(GeneralizedAmplitudeDamping)
def _(gate, probabilities, gamma, qubits):
    return GeneralizedAmplitudeDampingChannel(*probabilities, *gamma).on(*qubits)


@cirq_gate_to_instruciton.register(PhaseDamping)
def _(gate, probabilities, gamma, qubits):
    return PhaseDampingChannel(*probabilities, *gamma).on(*qubits)


class CirqProgramContext(AbstractProgramContext):
    def __init__(self):
        super().__init__(Circuit())

    def _get_qubits(self, qubits: Tuple[int]):
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
        qubits = self._get_qubits(target)
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
        qubits = self._get_qubits(target)
        instruction = cirq.MatrixGate(unitary).on(*qubits)
        self.circuit.append(instruction)

    def add_noise_instruction(self, noise):
        """Add a noise instruction the circuit"""
        qubits = self._get_qubits(noise.targets)
        probabilities = getattr(noise, "probabilities", [])
        gamma = getattr(noise, "gamma", [])
        instruction = cirq_gate_to_instruciton(noise, probabilities, gamma, qubits)
        self.circuit.append(instruction)

    def add_result(self, result: Results) -> None:
        """Add a result type to the circuit"""
        raise NotImplementedError
