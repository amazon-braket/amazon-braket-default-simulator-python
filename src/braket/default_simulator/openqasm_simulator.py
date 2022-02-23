from functools import reduce
from typing import List

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
    BitType, IntType, UnaryExpression, UnaryOperator, UintType, FloatType, AngleType, BinaryExpression, Identifier,
    BoolType, ClassicalAssignment, ComplexType, BranchingStatement, Statement, ArrayType, ArrayLiteral,
)
from openqasm.parser.antlr.qasm_parser import parse

from braket.default_simulator.openqasm_helpers import Bit, QubitPointer, Int, Uint, \
    Float, Angle, Bool, Complex, Array


class QasmSimulator:
    """
    For now, separate from BaseLocalSimulator. Will refactor to share what it can
    with the other existing classes. Currently mostly for prototyping.
    """

    def __init__(self):
        self.qubits = np.array([], dtype=complex)
        self._qasm_variables_stack = [{}]

    @property
    def qasm_variables(self):
        return reduce(
            lambda scope, new_scope: {**scope, **new_scope},
            self._qasm_variables_stack
        )

    def enter_scope(self):
        self._qasm_variables_stack.append({})

    def exit_scope(self):
        self._qasm_variables_stack.pop()

    def get_variable(self, name):
        for scope in reversed(self._qasm_variables_stack):
            if name in scope:
                return scope[name]

    def declare_variable(self, name, value):
        local_scope = self._qasm_variables_stack[-1]
        if name in local_scope:
            raise NameError(f"Variable '{name}' already declared in local scope.")
        local_scope[name] = value

    @property
    def num_qubits(self):
        return len(self.qubits)

    def get_qubit_state(self, quantum_variable):
        return self.qubits[self.get_variable(quantum_variable).value]

    def run_qasm(self, qasm: str):
        program = parse(qasm)
        self.run_program(program.statements)

    def run_program(self, statements: List[Statement]):
        for statement in statements:
            self.handle_statement(statement)

    def handle_statement(self, statement):
        if isinstance(statement, QubitDeclaration):
            self.handle_qubit_declaration(statement)
        elif isinstance(statement, ClassicalDeclaration):
            self.handle_classical_declaration(statement)
        elif isinstance(statement, ClassicalAssignment):
            self.handle_classical_assignment(statement)
        elif isinstance(statement, BranchingStatement):
            self.handle_branching_statement(statement)
        else:
            print(statement)
            raise NotImplementedError(
                f"Handling statement not implemented for statement: {statement}"
            )

    def handle_qubit_declaration(self, statement: QubitDeclaration):
        """
        Qubits and qubit registers can be declared, but only initialized using `reset`.
        New qubit declarations will automatically be mapped to the next contiguous
        block of qubits available.
        """
        name = statement.qubit.name
        size = self.evaluate_expression(statement.size)
        index = slice(self.num_qubits, self.num_qubits + size) if size else self.num_qubits
        # self.qasm_variables[name] = QubitPointer(name, index, size)
        self.declare_variable(name, QubitPointer(index, size))
        new_qubits = np.empty(size or 1)
        new_qubits[:] = np.nan
        self.qubits = np.append(self.qubits, new_qubits)

    def handle_classical_declaration(self, statement: ClassicalDeclaration):
        if isinstance(statement.type, ArrayType):
            self.handle_array_declaration(statement)
        else:
            type_map = {
                BitType: Bit,
                IntType: Int,
                UintType: Uint,
                FloatType: Float,
                AngleType: Angle,
                BoolType: Bool,
                ComplexType: Complex,
            }
            variable_class = type_map[type(statement.type)]
            name = statement.identifier.name
            value = self.evaluate_expression(statement.init_expression)
            size = (
               self.evaluate_expression(statement.type.size)
               if variable_class.supports_size
               else None
            )
            self.declare_variable(name, variable_class(value, size))

    def handle_array_declaration(self, statement: ClassicalDeclaration):
        name = statement.identifier.name
        base_type = statement.type.base_type
        base_type_variable_class = {
            BitType: Bit,
            IntType: Int,
            UintType: Uint,
            FloatType: Float,
            AngleType: Angle,
            BoolType: Bool,
            ComplexType: Complex,
        }[type(base_type)]
        base_type_size = (
           self.evaluate_expression(base_type.size)
           if base_type_variable_class.supports_size
           else None
        )
        dimensions = [dim.value for dim in statement.type.dimensions]

        def recast_values(values):
            if not isinstance(values[0], ArrayLiteral):
                return [
                    base_type_variable_class(self.evaluate_expression(v), base_type_size)
                    for v in values
                ]
            return [recast_values(v.values) for v in values]

        values = (
            recast_values(statement.init_expression.values)
            if statement.init_expression is not None
            else None
        )

        self.declare_variable(
            name,
            Array(values, dimensions, (base_type_variable_class, base_type_size)),
        )

    def handle_classical_assignment(self, statement: ClassicalAssignment):
        lvalue = statement.lvalue.name
        variable = self.qasm_variables.get(lvalue)
        if not variable:
            raise NameError(f"Variable '{lvalue}' not in scope.")
        rvalue = self.evaluate_expression(statement.rvalue)
        variable.assign_value(rvalue)

    def handle_branching_statement(self, statement: BranchingStatement):
        self.enter_scope()
        if self.evaluate_expression(statement.condition):
            self.run_program(statement.if_block)
        else:
            self.run_program(statement.else_block)
        self.exit_scope()

    def evaluate_expression(self, expression):
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
            base_value = self.evaluate_expression(expression.expression)
            operator = expression.op
            if operator.name == "-":
                return -base_value
            elif operator.name == "~":
                return ~base_value
            elif operator.name == "!":
                return type(base_value)(not base_value)

        elif isinstance(expression, BinaryExpression):
            lhs = self.evaluate_expression(expression.lhs)
            rhs = self.evaluate_expression(expression.rhs)
            operator = expression.op
            if operator.name == "*":
                return lhs * rhs
            elif operator.name == "/":
                return lhs / rhs
            elif operator.name == "+":
                return lhs + rhs
            elif operator.name == "-":
                return lhs - rhs

        elif isinstance(expression, Identifier):
            return self.get_variable(expression.name).value

        else:
            raise NotImplementedError
