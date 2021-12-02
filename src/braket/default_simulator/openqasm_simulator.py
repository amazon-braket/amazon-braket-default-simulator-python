import attr
import numpy as np
from openqasm.ast import QubitDeclaration
from openqasm.parser.antlr.qasm_parser import parse


@attr.s
class QasmVariable:
    name = attr.ib()
    index = attr.ib()
    quantum = attr.ib(default=True)


class QasmSimulator:
    """
    For now, separate from BaseLocalSimulator. Will refactor to share what it can
    with the other existing classes. Currently mostly for prototyping.
    """

    def __init__(self):
        self.qubits = np.zeros(0, dtype=complex)
        self.variables = {}

    @property
    def num_qubits(self):
        return len(self.qubits)

    def get_variable_value(self, variable):
        var_index = self.variables[variable].index
        return self.qubits[var_index]

    def run_qasm(self, qasm: str):
        program = parse(qasm)
        for statement in program.statements:
            self.handle_statement(statement)

    def handle_statement(self, statement):
        if isinstance(statement, QubitDeclaration):
            self.handle_qubit_declaration(statement)

    def handle_qubit_declaration(self, statement):
        name = statement.qubit.name

        if statement.size:
            size = statement.size.value
            index = slice(self.num_qubits, self.num_qubits + size)
        else:
            size = 1
            index = self.num_qubits

        self.variables[name] = QasmVariable(name, index)
        self.qubits = np.append(self.qubits, np.zeros(size))
