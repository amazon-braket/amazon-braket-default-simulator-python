from copy import deepcopy
from dataclasses import fields
from functools import singledispatchmethod
from typing import List, Optional

from openqasm3 import parse
from openqasm3.ast import (
    BinaryExpression,
    BooleanLiteral,
    BranchingStatement,
    ClassicalDeclaration,
    Constant,
    ConstantDeclaration,
    DiscreteSet,
    ForInLoop,
    GateModifierName,
    Identifier,
    Include,
    IndexedIdentifier,
    IndexExpression,
    IntegerLiteral,
    Program,
    QASMNode,
    QuantumGate,
    QuantumGateDefinition,
    QuantumGateModifier,
    QuantumMeasurement,
    QuantumMeasurementAssignment,
    QuantumPhase,
    QuantumReset,
    QuantumStatement,
    QubitDeclaration,
    RangeDefinition,
    RealLiteral,
    StringLiteral,
    UnaryExpression,
)

from braket.default_simulator.openqasm import data_manipulation
from braket.default_simulator.openqasm.data_manipulation import (
    convert_to_gate,
    get_ctrl_modifiers,
    invert,
    is_controlled,
    is_inverted,
    is_literal,
    modify_body,
)
from braket.default_simulator.openqasm.program_context import ProgramContext


class Interpreter:
    def __init__(self, context: Optional[ProgramContext] = None):
        # context keeps track of all state
        self.context = context or ProgramContext()

    def run(self, program: Program):
        self.visit(program)
        return self.context

    @singledispatchmethod
    def visit(self, node):
        print(f"Node: {node}")
        if node is None:
            return
        if not isinstance(node, QASMNode):
            return node
        for field in fields(node):
            value = getattr(node, field.name)
            setattr(node, field.name, self.visit(value))
        return node

    @visit.register
    def _(self, node_list: list):
        print(f"list: {node_list}")
        return [self.visit(node) for node in node_list]

    @visit.register
    def _(self, node: ClassicalDeclaration):
        print(f"Classical declaration: {node}")
        node_type = self.visit(node.type)
        if node.init_expression is not None:
            init_expression = self.visit(node.init_expression)
            init_value = data_manipulation.cast_to(node.type, init_expression)
        else:
            init_value = None
        self.context.declare_variable(node.identifier.name, node_type, init_value)

    @visit.register
    def _(self, node: ConstantDeclaration):
        print(f"Constant declaration: {node}")
        node_type = self.visit(node.type)
        if node.init_expression is not None:
            init_expression = self.visit(node.init_expression)
            init_value = data_manipulation.cast_to(node.type, init_expression)
        else:
            init_value = None
        self.context.declare_variable(node.identifier.name, node_type, init_value, const=True)

    @visit.register
    def _(self, node: BinaryExpression):
        print(f"Binary expression: {node}")
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        if is_literal(lhs) and is_literal(rhs):
            result_type = data_manipulation.resolve_result_type(type(lhs), type(rhs))
            lhs = data_manipulation.cast_to(result_type, lhs)
            rhs = data_manipulation.cast_to(result_type, rhs)
            return data_manipulation.evaluate_binary_expression(lhs, rhs, node.op)
        else:
            return BinaryExpression(node.op, lhs, rhs)

    @visit.register
    def _(self, node: UnaryExpression):
        print(f"Unary expression: {node}")
        expression = self.visit(node.expression)
        if is_literal(expression):
            return data_manipulation.evaluate_unary_expression(expression, node.op)
        else:
            return UnaryExpression(node.op, expression)

    @visit.register
    def _(self, node: Constant):
        print(f"Constant: {node}")
        return data_manipulation.evaluate_constant(node)

    @visit.register(BooleanLiteral)
    @visit.register(IntegerLiteral)
    @visit.register(RealLiteral)
    def _(self, node):
        print(f"Literal: {node}")
        return node

    @visit.register
    def _(self, node: Identifier):
        if self.context.get_value(node.name) is None:
            raise NameError(f"Identifier {node.name} is not initialized.")
        return self.context.get_value(node.name)

    @visit.register
    def _(self, node: QubitDeclaration):
        print(f"Qubit declaration: {node}")
        size = self.visit(node.size).value if node.size else 1
        self.context.add_qubits(node.qubit.name, size)

    @visit.register
    def _(self, node: QuantumReset):
        print(f"Quantum reset: {node}")
        qubits = self.visit(node.qubits)
        self.context.reset_qubits(qubits)

    @visit.register
    def _(self, node: IndexedIdentifier):
        print(f"Indexed identifier: {node}")
        name = node.name
        if name.name not in self.context.qubit_mapping:
            raise NotImplementedError("Indexed identifier only implemented for qubits")
        indices = []
        for index in node.indices:
            if isinstance(index, DiscreteSet):
                indices.append(index)
            else:
                for element in index:
                    # todo: set current dimension's length for [:] and [:-1] indexing
                    element = self.visit(element)
                    if isinstance(element, IntegerLiteral):
                        indices.append(DiscreteSet([element]))
                    elif isinstance(element, DiscreteSet):
                        indices.append(element)
                    else:
                        raise NotImplementedError(f"Index {index} not valid for qubit indexing")
        return IndexedIdentifier(name, indices)

    @visit.register
    def _(self, node: RangeDefinition):
        print(f"Range definition: {node}")
        start = self.visit(node.start).value if node.start else 0
        end = self.visit(node.end).value + 1 if node.end else NotImplementedError
        step = self.visit(node.step).value if node.step else 1
        return DiscreteSet([IntegerLiteral(i) for i in range(start, end, step)])

    @visit.register
    def _(self, node: IndexExpression):
        """cast index to list of integer literals"""
        print(f"Index expression: {node}")
        array = self.visit(node.collection)
        if isinstance(node.index, DiscreteSet):
            index = self.visit(node.index)
        else:
            index = [self.visit(i) for i in node.index]
        return data_manipulation.get_elements(array, index)

    @visit.register
    def _(self, node: QuantumGateDefinition):
        print(f"Quantum gate definition: {node}")
        with self.context.enter_scope():
            for qubit in node.qubits:
                self.context.declare_alias(qubit.name, qubit)

            for param in node.arguments:
                self.context.declare_alias(param.name, param)

            node.body = self.inline_gate_def_body(node.body)
        self.context.add_gate(node.name.name, node)

    def inline_gate_def_body(self, body: List[QuantumStatement]):
        inlined_body = []
        for statement in body:
            if isinstance(statement, QuantumPhase):
                statement.argument = self.visit(statement.argument)
                statement.modifiers = self.visit(statement.quantum_gate_modifiers)
                if is_inverted(statement):
                    statement = invert(statement)
                if is_controlled(statement):
                    statement = convert_to_gate(statement)
                # statement is a quantum phase instruction
                else:
                    inlined_body.append(statement)
            # this includes converted phase instructions
            if isinstance(statement, QuantumGate):
                gate_name = statement.name.name
                statement.arguments = self.visit(statement.arguments)
                statement.modifiers = self.visit(statement.modifiers)
                statement.qubits = self.visit(statement.qubits)
                if gate_name == "U":
                    if is_inverted(statement):
                        statement = invert(statement)
                    inlined_body.append(statement)
                else:
                    with self.context.enter_scope():
                        gate_def = self.context.get_gate_definition(gate_name)
                        ctrl_modifiers = get_ctrl_modifiers(statement.modifiers)
                        num_ctrl = sum(mod.argument.value for mod in ctrl_modifiers)
                        ctrl_qubits = statement.qubits[:num_ctrl]
                        gate_qubits = statement.qubits[num_ctrl:]

                        for qubit_called, qubit_defined in zip(gate_qubits, gate_def.qubits):
                            self.context.declare_alias(qubit_defined.name, qubit_called)

                        for param_called, param_defined in zip(
                            statement.arguments, gate_def.arguments
                        ):
                            self.context.declare_alias(param_defined.name, param_called)

                        body_copy = modify_body(
                            deepcopy(gate_def.body),
                            is_inverted(statement),
                            ctrl_modifiers,
                            ctrl_qubits,
                        )
                        inlined_body += self.inline_gate_def_body(body_copy)
        return inlined_body

    @visit.register
    def _(self, node: QuantumGate):
        print(f"Quantum gate: {node}")
        gate_name = node.name.name
        node.arguments = [self.visit(arg) for arg in node.arguments]
        qubits = [self.visit(qubit) for qubit in node.qubits]

        if gate_name == "U":
            self.context.execute_builtin_unitary(
                node.arguments,
                qubits,
                node.modifiers,
            )
        else:
            with self.context.enter_scope():
                gate_def = self.context.get_gate_definition(gate_name)

                ctrl_modifiers = get_ctrl_modifiers(node.modifiers)
                num_ctrl = sum(mod.argument.value for mod in ctrl_modifiers)
                gate_qubits = qubits[num_ctrl:]

                for qubit_called, qubit_defined in zip(gate_qubits, gate_def.qubits):
                    self.context.declare_alias(qubit_defined.name, qubit_called)

                for param_called, param_defined in zip(node.arguments, gate_def.arguments):
                    self.context.declare_alias(param_defined.name, param_called)

                for statement in deepcopy(gate_def.body):
                    if isinstance(statement, QuantumGate):
                        self.visit(statement)
                    elif isinstance(statement, QuantumPhase):
                        phase = statement.argument.value
                        self.context.apply_phase(phase, qubits)

    @visit.register
    def _(self, node: QuantumPhase):
        print(f"Quantum phase: {node}")
        node.argument = self.visit(node.argument)
        node.modifiers = self.visit(node.quantum_gate_modifiers)
        if is_inverted(node):
            node = invert(node)
        if is_controlled(node):
            node = convert_to_gate(node)
            self.visit(node)
        else:
            self.context.apply_phase(node.argument.value)

    @visit.register
    def _(self, node: QuantumGateModifier):
        print(f"Quantum gate modifier: {node}")
        if node.modifier == GateModifierName.inv:
            if node.argument is not None:
                raise ValueError("inv modifier does not take an argument")
        elif node.modifier in (GateModifierName.ctrl, GateModifierName.negctrl):
            if node.argument is None:
                node.argument = IntegerLiteral(1)
            else:
                node.argument = self.visit(node.argument)
        return node

    @visit.register
    def _(self, node: QuantumMeasurement):
        print(f"Quantum measurement: {node}")
        qubits = self.visit(node.qubit)
        return StringLiteral(self.context.measure_qubits(qubits))

    @visit.register
    def _(self, node: QuantumMeasurementAssignment):
        print(f"Quantum measurement assignment: {node}")
        measurement = self.visit(node.measure_instruction)
        if node.target is not None:
            self.context.update_value(node.target.name, measurement.value)

    @visit.register
    def _(self, node: BranchingStatement):
        print(f"Branching statement: {node}")
        block = node.if_block if self.visit(node.condition) else node.else_block
        for statement in block:
            self.visit(statement)

    @visit.register
    def _(self, node: ForInLoop):
        print(f"For in loop: {node}")
        index = self.visit(node.set_declaration)
        for i in index.values:
            with self.context.enter_scope():
                self.context.declare_variable(node.loop_variable.name, IntegerLiteral, i)
                self.visit(node.block)

    @visit.register
    def _(self, node: Include):
        print(f"Include: {node}")
        with open(node.filename, "r") as f:
            included = f.read()
            parsed = parse(included)
            self.visit(parsed)
