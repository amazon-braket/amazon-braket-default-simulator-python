from braket.ir.jaqcd.program_v1 import Results

from braket.default_simulator import GateOperation


class Circuit:
    def __init__(self, instructions=None, results=None):
        self.instructions = []
        self.results = []
        self.qubit_set = set()

        if instructions:
            for instruction in instructions:
                self.add_instruction(instruction)

        if results:
            for result in results:
                self.add_result(result)

    def add_instruction(self, instruction: GateOperation):
        self.instructions.append(instruction)
        self.qubit_set |= set(instruction.targets)

    def add_result(self, result: Results):
        self.results.append(result)

    @property
    def num_qubits(self):
        return len(self.qubit_set)

    def __eq__(self, other):
        return (self.instructions, self.results) == (other.instructions, other.results)
