from __future__ import annotations

from typing import List, Optional

from braket.ir.jaqcd.program_v1 import Results
from braket.ir.jaqcd.shared_models import Observable

from braket.default_simulator.operation import GateOperation, KrausOperation
from braket.default_simulator.operation_helpers import from_braket_instruction
from braket.default_simulator.result_types import _from_braket_observable


class Circuit:
    """
    This is a lightweight analog to braket.ir.jaqcd.program_v1.Program.
    The Interpreter compiles to an IR to hand off to the simulator,
    braket.default_simulator.state_vector_simulator.StateVectorSimulator, for example.
    Our simulator module takes in a circuit specification that satisfies the interface
    implemented by this class.
    """

    def __init__(
        self,
        instructions: Optional[List[GateOperation]] = None,
        results: Optional[List[Results]] = None,
    ):
        self.instructions = []
        self.results = []
        self.qubit_set = set()

        if instructions:
            for instruction in instructions:
                self.add_instruction(instruction)

        if results:
            for result in results:
                self.add_result(result)

    def add_instruction(self, instruction: [GateOperation, KrausOperation]) -> None:
        """
        Add instruction to the circuit.

        Args:
            instruction (GateOperation): Instruction to add.
        """
        self.instructions.append(instruction)
        self.qubit_set |= set(instruction.targets)

    def add_result(self, result: Results) -> None:
        """
        Add result type to the circuit.

        Args:
            result (Results): Result type to add.
        """
        self.results.append(result)

    @property
    def num_qubits(self) -> int:
        return len(self.qubit_set)

    @property
    def basis_rotation_instructions(self):
        """
        This function assumes all observables are commuting.
        """

        basis_rotation_instructions = []
        measured_qubits = set()

        for result in self.results:
            if isinstance(result, Observable):
                observables = result.observable
                targets = result.targets or range(self.num_qubits)

                if set(targets).issubset(measured_qubits):
                    continue
                elif set(targets).isdisjoint(measured_qubits):
                    braket_obs = _from_braket_observable(observables, targets)
                    diagonalizing_gates = braket_obs.diagonalizing_gates()
                    basis_rotation_instructions.extend(diagonalizing_gates)
                    measured_qubits |= set(targets)
                else:
                    raise NotImplementedError("Partially measured observable target")

        return basis_rotation_instructions

    def __eq__(self, other: Circuit):
        return (self.instructions, self.results) == (other.instructions, other.results)
