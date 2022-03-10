from copy import deepcopy
from functools import singledispatchmethod
from typing import List

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
    modify_body,
)
from braket.default_simulator.openqasm.program_context import ProgramContext
from braket.default_simulator.openqasm.visitor import QASMTransformer


class Interpreter(QASMTransformer):
    def run(self, program: QASMNode):
        program_context = ProgramContext()
        self.visit(program, program_context)
        return program_context

    @singledispatchmethod
    def visit(self, node, context=None):
        return super().visit(node, context)

    @visit.register
    def _(self, node: Program, context: ProgramContext):
        """Returns ProgramContext rather than the consumed Program node"""
        return super().visit(node, context)

    @visit.register
    def _(self, node_list: list, context: ProgramContext):
        return [self.visit(node, context) for node in node_list]

    @visit.register
    def _(self, node: ClassicalDeclaration, context: ProgramContext):
        print(f"Classical declaration: {node}")
        node_type = self.visit(node.type, context)
        if node.init_expression is not None:
            init_expression = self.visit(node.init_expression, context)
            init_value = data_manipulation.cast_to(node.type, init_expression)
        else:
            init_value = None
        context.declare_variable(node.identifier.name, node_type, init_value)

    @visit.register
    def _(self, node: ConstantDeclaration, context: ProgramContext):
        print(f"Constant declaration: {node}")
        node_type = self.visit(node.type, context)
        if node.init_expression is not None:
            init_expression = self.visit(node.init_expression, context)
            init_value = data_manipulation.cast_to(node.type, init_expression)
        else:
            init_value = None
        context.declare_variable(node.identifier.name, node_type, init_value, const=True)

    @visit.register
    def _(self, node: BinaryExpression, context: ProgramContext):
        print(f"Binary expression: {node}")
        lhs = self.visit(node.lhs, context)
        rhs = self.visit(node.rhs, context)
        op = self.visit(node.op, context)
        result_type = data_manipulation.resolve_result_type(type(lhs), type(rhs))
        lhs = data_manipulation.cast_to(result_type, lhs)
        rhs = data_manipulation.cast_to(result_type, rhs)
        return data_manipulation.evaluate_binary_expression(lhs, rhs, op)

    @visit.register
    def _(self, node: UnaryExpression, context: ProgramContext):
        print(f"Unary expression: {node}")
        expression = self.visit(node.expression, context)
        op = self.visit(node.op, context)
        return data_manipulation.evaluate_unary_expression(expression, op)

    @visit.register
    def _(self, node: Constant, context: ProgramContext):
        return data_manipulation.evaluate_constant(node)

    @visit.register(BooleanLiteral)
    @visit.register(IntegerLiteral)
    @visit.register(RealLiteral)
    def _(self, node, context: ProgramContext):
        return node

    @visit.register
    def _(self, node: Identifier, context: ProgramContext):
        if context.get_value(node.name) is None:
            raise NameError(f"Identifier {node.name} is not initialized.")
        value = context.get_value(node.name)
        return value if value == node else self.visit(value, context)

    @visit.register
    def _(self, node: QubitDeclaration, context: ProgramContext):
        print(f"Qubit declaration: {node}")
        size = self.visit(node.size, context).value if node.size else 1
        context.add_qubits(node.qubit.name, size)

    @visit.register
    def _(self, node: QuantumReset, context: ProgramContext):
        print(f"Quantum reset: {node}")
        qubits = self.visit(node.qubits, context)
        context.reset_qubits(qubits)

    @visit.register
    def _(self, node: IndexedIdentifier, context: ProgramContext):
        print(f"Indexed identifier: {node}")
        name = node.name
        if name.name not in context.qubit_mapping:
            raise NotImplementedError("Indexed identifier only implemented for qubits")
        indices = []
        for index in node.indices:
            if isinstance(index, DiscreteSet):
                indices.append(index)
            else:
                for element in index:
                    # todo: set current dimension's length for [:] and [:-1] indexing
                    element = self.visit(element, context)
                    if isinstance(element, IntegerLiteral):
                        indices.append(DiscreteSet([element]))
                    elif isinstance(element, DiscreteSet):
                        indices.append(element)
                    else:
                        raise NotImplementedError(f"Index {index} not valid for qubit indexing")
        return IndexedIdentifier(name, indices)

    @visit.register
    def _(self, node: RangeDefinition, context: ProgramContext):
        print(f"Range definition: {node}")
        start = self.visit(node.start, context).value if node.start else 0
        end = self.visit(node.end, context).value + 1 if node.end else NotImplementedError
        step = self.visit(node.step, context).value if node.step else 1
        return DiscreteSet([IntegerLiteral(i) for i in range(start, end, step)])

    @visit.register
    def _(self, node: IndexExpression, context: ProgramContext):
        """cast index to list of integer literals"""
        print(f"Index expression: {node}")
        array = self.visit(node.collection, context)
        if isinstance(node.index, DiscreteSet):
            index = self.visit(node.index, context)
        else:
            index = [self.visit(i, context) for i in node.index]
        return data_manipulation.get_elements(array, index)

    @visit.register
    def _(self, node: QuantumGateDefinition, context: ProgramContext):
        print(f"Quantum gate definition: {node}")
        with context.enter_scope():
            for qubit in node.qubits:
                context.declare_alias(qubit.name, qubit)

            for param in node.arguments:
                context.declare_alias(param.name, param)

            node.body = self.inline_gate_def_body(node.body, context)
        context.add_gate(node.name.name, node)

    def inline_gate_def_body(self, body: List[QuantumStatement], context: ProgramContext):
        inlined_body = []
        for statement in body:
            if isinstance(statement, QuantumPhase):
                statement.argument = self.visit(statement.argument, context)
                statement.modifiers = self.visit(statement.quantum_gate_modifiers, context)
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
                statement.arguments = self.visit(statement.arguments, context)
                statement.modifiers = self.visit(statement.modifiers, context)
                statement.qubits = self.visit(statement.qubits, context)
                if gate_name == "U":
                    if is_inverted(statement):
                        statement = invert(statement)
                    inlined_body.append(statement)
                else:
                    with context.enter_scope():
                        gate_def = context.get_gate_definition(gate_name)
                        ctrl_modifiers = get_ctrl_modifiers(statement.modifiers)
                        num_ctrl = sum(mod.argument.value for mod in ctrl_modifiers)
                        ctrl_qubits = statement.qubits[:num_ctrl]
                        gate_qubits = statement.qubits[num_ctrl:]

                        for qubit_called, qubit_defined in zip(gate_qubits, gate_def.qubits):
                            context.declare_alias(qubit_defined.name, qubit_called)

                        for param_called, param_defined in zip(
                            statement.arguments, gate_def.arguments
                        ):
                            context.declare_alias(param_defined.name, param_called)

                        body_copy = modify_body(
                            deepcopy(gate_def.body),
                            is_inverted(statement),
                            ctrl_modifiers,
                            ctrl_qubits,
                        )
                        inlined_body += self.inline_gate_def_body(body_copy, context)
        return inlined_body

    @visit.register
    def _(self, node: QuantumGate, context: ProgramContext):
        print(f"Quantum gate: {node}")
        gate_name = node.name.name
        node.arguments = [self.visit(arg, context) for arg in node.arguments]
        qubits = [self.visit(qubit, context) for qubit in node.qubits]

        if gate_name == "U":
            context.execute_builtin_unitary(
                node.arguments,
                qubits,
                node.modifiers,
            )
        else:
            with context.enter_scope():
                gate_def = context.get_gate_definition(gate_name)

                ctrl_modifiers = get_ctrl_modifiers(node.modifiers)
                num_ctrl = sum(mod.argument.value for mod in ctrl_modifiers)
                gate_qubits = qubits[num_ctrl:]

                for qubit_called, qubit_defined in zip(gate_qubits, gate_def.qubits):
                    context.declare_alias(qubit_defined.name, qubit_called)

                for param_called, param_defined in zip(node.arguments, gate_def.arguments):
                    context.declare_alias(param_defined.name, param_called)

                for statement in deepcopy(gate_def.body):
                    if isinstance(statement, QuantumGate):
                        self.visit(statement, context)
                    elif isinstance(statement, QuantumPhase):
                        phase = statement.argument.value
                        context.apply_phase(phase, qubits)

    @visit.register
    def _(self, node: QuantumPhase, context: ProgramContext):
        print(f"Quantum phase: {node}")
        node.argument = self.visit(node.argument, context)
        node.modifiers = self.visit(node.quantum_gate_modifiers, context)
        if is_inverted(node):
            node = invert(node)
        if is_controlled(node):
            node = convert_to_gate(node)
            self.visit(node, context)
        else:
            context.apply_phase(node.argument.value)

    @visit.register
    def _(self, node: QuantumGateModifier, context: ProgramContext):
        print(f"Quantum gate modifier: {node}")
        if node.modifier == GateModifierName.inv:
            if node.argument is not None:
                raise ValueError("inv modifier does not take an argument")
        elif node.modifier in (GateModifierName.ctrl, GateModifierName.negctrl):
            if node.argument is None:
                node.argument = IntegerLiteral(1)
            else:
                node.argument = self.visit(node.argument, context)
        return node

    @visit.register
    def _(self, node: QuantumMeasurement, context: ProgramContext):
        print(f"Quantum measurement: {node}")
        qubits = self.visit(node.qubit, context)
        return StringLiteral(context.measure_qubits(qubits))

    @visit.register
    def _(self, node: QuantumMeasurementAssignment, context: ProgramContext):
        print(f"Quantum measurement assignment: {node}")
        measurement = self.visit(node.measure_instruction, context)
        if node.target is not None:
            context.update_value(node.target.name, measurement.value)

    @visit.register
    def _(self, node: BranchingStatement, context: ProgramContext):
        print(f"Branching statement: {node}")
        block = node.if_block if self.visit(node.condition, context) else node.else_block
        for statement in block:
            self.visit(statement, context)

    @visit.register
    def _(self, node: ForInLoop, context: ProgramContext):
        print(f"For in loop: {node}")
        index = self.visit(node.set_declaration, context)
        for i in index.values:
            context.push_scope()
            context.declare_variable(node.loop_variable.name, IntegerLiteral, i)
            self.visit(node.block, context)
            context.pop_scope()
