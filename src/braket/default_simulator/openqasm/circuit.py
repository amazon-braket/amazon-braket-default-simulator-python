from __future__ import annotations

from typing import List, Optional

from braket.ir.jaqcd.program_v1 import Results

from braket.default_simulator.operation import GateOperation


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

    def add_instruction(self, instruction: GateOperation) -> None:
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

    def __eq__(self, other: Circuit):
        return (self.instructions, self.results) == (other.instructions, other.results)
