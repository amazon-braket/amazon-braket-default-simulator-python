from functools import reduce
from typing import Iterable, List, Tuple

import numpy as np
from openqasm.ast import (
    AngleType,
    ArrayLiteral,
    ArrayType,
    BinaryExpression,
    BitType,
    BooleanLiteral,
    BoolType,
    BranchingStatement,
    ClassicalAssignment,
    ClassicalDeclaration,
    ComplexType,
    Constant,
    ConstantName,
    Expression,
    FloatType,
    Identifier,
    IntegerLiteral,
    IntType,
    QuantumGate,
    QuantumGateDefinition,
    QuantumMeasurementAssignment,
    QuantumReset,
    QubitDeclaration,
    RealLiteral,
    Statement,
    StringLiteral,
    UintType,
    UnaryExpression,
)
from openqasm.parser.antlr.qasm_parser import parse

from braket.default_simulator.openqasm_helpers import (
    Angle,
    Array,
    Bit,
    Bool,
    Complex,
    Float,
    Gate,
    GateCall,
    Int,
    Number,
    QubitPointer,
    Uint,
    sample_qubit,
)


class QasmSimulator:
    """
    For now, separate from BaseLocalSimulator. Will refactor to share what it can
    with the other existing classes. Currently mostly for prototyping.
    """

    def __init__(self):
        self.qubits = np.empty((0, 2), dtype=complex)
        self._qasm_variables_stack = [{}]

    @property
    def qasm_variables(self):
        return reduce(lambda scope, new_scope: {**scope, **new_scope}, self._qasm_variables_stack)

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
        return self.qubits.shape[0]

    def get_qubit_state(self, quantum_variable):
        return self.qubits[self.get_variable(quantum_variable).value]

    def reset_qubits(self, quantum_variable):
        self.qubits[self.get_variable(quantum_variable).value] = [1, 0]

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
        elif isinstance(statement, QuantumReset):
            self.handle_quantum_reset(statement)
        elif isinstance(statement, QuantumMeasurementAssignment):
            self.handle_quantum_measurement_assignment(statement)
        elif isinstance(statement, QuantumGateDefinition):
            self.handle_quantum_gate_definition(statement)
        elif isinstance(statement, QuantumGate):
            self.handle_quantum_gate(statement)
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
        self.declare_variable(name, QubitPointer(index, size))
        new_qubits = np.empty(((size or 1), 2))
        new_qubits[:] = [np.nan, np.nan]
        self.qubits = np.append(self.qubits, new_qubits, axis=0)

    def handle_quantum_reset(self, statement: QuantumReset):
        self.reset_qubits(statement.qubits.name)

    def handle_quantum_measurement_assignment(self, statement: QuantumMeasurementAssignment):
        assignment_target = statement.target.name
        measurement_target = statement.measure_instruction.qubit.name
        assignment_target_variable = self.get_variable(assignment_target)
        measurement_target_variable = self.get_variable(measurement_target)

        # assert types match
        if assignment_target_variable.size != measurement_target_variable.size:
            raise ValueError("Measurement target size must match assignment target size.")

        # sample and collapse quantum state
        if not measurement_target_variable.size:
            sampled = sample_qubit(self.get_qubit_state(measurement_target))
            self.qubits[measurement_target_variable.value] = np.array([0, 1]) == sampled
        else:
            sampled = "".join(
                str(sample_qubit(q)) for q in self.get_qubit_state(measurement_target)
            )
            sampled_array = np.array([[int(m)] for m in sampled])
            self.qubits[measurement_target_variable.value] = sampled_array == np.repeat(
                [[0, 1]], measurement_target_variable.size, axis=0
            )

        # assign result to assignment target
        assignment_target_variable.assign_value(sampled)

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

        base_type_template = base_type_variable_class(value=None, size=base_type_size)

        self.declare_variable(
            name,
            Array(values, dimensions, base_type_template),
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

    def handle_quantum_gate_definition(self, statement: QuantumGateDefinition):
        name = statement.name.name
        params = [arg.name for arg in statement.arguments]
        targets = [target.name for target in statement.qubits]
        body = []

        # need to parse body
        # decision: gates in body are stored by reference

        for directive in statement.body:
            if isinstance(directive, QuantumGate):
                body.append(self.parse_quantum_gate(directive))
            else:
                raise NotImplementedError("mods and stuff")

        self.declare_variable(name, Gate(params, targets, body))
        # print(self.get_variable(name))

    def parse_quantum_gate(self, statement: QuantumGate):
        name = statement.name.name
        params = statement.arguments
        targets = statement.qubits
        modifiers = statement.modifiers

        return GateCall(name, params, targets, modifiers)

    def handle_quantum_gate(self, statement: QuantumGate):
        gate_call = self.parse_quantum_gate(statement)
        self.execute_gate_call(gate_call)

    def execute_gate_call(self, gate_call: GateCall):
        params = [self.evaluate_expression(param, unwrap=True) for param in gate_call.params]
        targets = [self.evaluate_expression(target) for target in gate_call.targets]

        target_sizes = [t.size for t in targets if t.size is not None]
        if target_sizes:
            register_size = target_sizes[0]
            assert all(size == register_size for size in target_sizes), "unmatched target sizes"
        else:
            register_size = 1

        for i in range(register_size):
            unsliced_targets = [
                (
                    QubitPointer(t.value)
                    if not t.size
                    # need to generalize when implementing qreg aliasing
                    else QubitPointer(t.value.start + i)
                )
                for t in targets
            ]

            if gate_call.name == "U":
                assert len(unsliced_targets) == 1, "built-in unitary is a single qubit gate"
                self.execute_builtin_unitary(unsliced_targets[0], params)
            else:
                gate = self.get_variable(gate_call.name)

                self.enter_scope()

                assert len(gate.params) == len(params), "different number of params"
                for param, value in zip(gate.params, params):
                    self.declare_variable(param, Number(value))

                assert len(gate.targets) == len(targets), "different number of targets"
                for target, value in zip(gate.targets, unsliced_targets):
                    self.declare_variable(target, value)

                for directive in gate.body:
                    if isinstance(directive, GateCall):
                        self.execute_gate_call(directive)
                    else:
                        raise NotImplementedError("mods and stuff")

    def execute_builtin_unitary(self, qubit_pointer: QubitPointer, params: Iterable[complex]):
        θ, ϕ, λ = params
        unitary = np.array(
            [
                [np.cos(θ / 2), -np.exp(1j * λ) * np.sin(θ / 2)],
                [np.exp(1j * ϕ) * np.sin(θ / 2), np.exp(1j * (ϕ + λ)) * np.cos(θ / 2)],
            ]
        )
        self.qubits[qubit_pointer.value] = unitary @ self.qubits[qubit_pointer.value]

    def evaluate_expression(self, expression, unwrap=False):
        if expression is None:
            return None

        elif isinstance(expression, (IntegerLiteral, RealLiteral, BooleanLiteral, StringLiteral)):
            return expression.value

        elif isinstance(expression, Constant):
            constant_values = {
                ConstantName.pi: np.pi,
                ConstantName.tau: 2 * np.pi,
                ConstantName.euler: np.e,
            }
            return constant_values.get(expression.name)

        elif isinstance(expression, UnaryExpression):
            base_value = self.evaluate_expression(expression.expression, unwrap)
            operator = expression.op
            if operator.name == "-":
                return -base_value
            elif operator.name == "~":
                return ~base_value
            elif operator.name == "!":
                return type(base_value)(not base_value)

        elif isinstance(expression, BinaryExpression):
            lhs = self.evaluate_expression(expression.lhs, unwrap)
            rhs = self.evaluate_expression(expression.rhs, unwrap)
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
            variable = self.get_variable(expression.name)
            return variable.value if unwrap else variable

        else:
            raise NotImplementedError
