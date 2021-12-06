import numpy as np
from openqasm.ast import (
    QubitDeclaration,
    ClassicalDeclaration,
    IntegerLiteral,
    RealLiteral,
    BooleanLiteral,
    StringLiteral,
    Constant,
    ConstantName,
    BitType, IntType, UnaryExpression, UnaryOperator, UintType, FloatType,
)
from openqasm.parser.antlr.qasm_parser import parse

from braket.default_simulator.openqasm_helpers import BitVariable, QubitPointer, IntVariable, UintVariable, \
    FloatVariable


class QasmSimulator:
    """
    For now, separate from BaseLocalSimulator. Will refactor to share what it can
    with the other existing classes. Currently mostly for prototyping.
    """

    def __init__(self):
        self.qubits = np.array([], dtype=complex)
        self.qasm_variables = {}

    @property
    def num_qubits(self):
        return len(self.qubits)

    def get_qubit_state(self, quantum_variable):
        return self.qubits[self.qasm_variables[quantum_variable].value]

    def run_qasm(self, qasm: str):
        program = parse(qasm)
        for statement in program.statements:
            self.handle_statement(statement)

    def handle_statement(self, statement):
        if isinstance(statement, QubitDeclaration):
            self.handle_qubit_declaration(statement)
        elif isinstance(statement, ClassicalDeclaration):
            self.handle_classical_declaration(statement)
        else:
            raise NotImplementedError(
                f"Handling statement not implemented for statement: {statement}"
            )

    def handle_qubit_declaration(self, statement):
        """
        Qubits and qubit registers can be declared, but only initialized using `reset`.
        New qubit declarations will automatically be mapped to the next contiguous
        block of qubits available.
        """
        name = statement.qubit.name
        size = self.evaluate_expression(statement.size)
        index = slice(self.num_qubits, self.num_qubits + size) if size else self.num_qubits
        self.qasm_variables[name] = QubitPointer(name, index, size)
        new_qubits = np.empty(size or 1)
        new_qubits[:] = np.nan
        self.qubits = np.append(self.qubits, new_qubits)

    def handle_classical_declaration(self, statement):
        type_map = {
            BitType: BitVariable,
            IntType: IntVariable,
            UintType: UintVariable,
            FloatType: FloatVariable,
        }
        variable_class = type_map[type(statement.type)]
        name = statement.identifier.name
        value = self.evaluate_expression(statement.init_expression)
        size = self.evaluate_expression(statement.type.size)
        self.qasm_variables[name] = variable_class(name, value, size)

    @staticmethod
    def evaluate_expression(expression):
        if expression is None:
            return None

        elif isinstance(expression, (
            IntegerLiteral, RealLiteral, BooleanLiteral, StringLiteral
        )):
            return expression.value

        elif isinstance(expression, Constant):
            constant_values = {
                ConstantName.pi: np.pi,
                ConstantName.tau: 2 * np.pi,
                ConstantName.euler: np.e,
            }
            return constant_values.get(expression.name)

        elif isinstance(expression, UnaryExpression):
            base_value = QasmSimulator.evaluate_expression(expression.expression)
            operator = expression.op
            if operator.name == "-":
                return -base_value
            elif operator.name == "~":
                return ~base_value
            elif operator.name == "!":
                return type(base_value)(not base_value)

        else:
            raise NotImplementedError
