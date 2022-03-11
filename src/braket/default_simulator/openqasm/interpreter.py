from copy import deepcopy
from dataclasses import fields
from functools import singledispatchmethod
from typing import List, Optional, Union

from openqasm3 import parse
from openqasm3.ast import (
    ArrayLiteral,
    ArrayType,
    BinaryExpression,
    BooleanLiteral,
    BranchingStatement,
    Cast,
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
    UnaryExpression, BitType,
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

    def run_file(self, filename: str):
        with open(filename, "r") as f:
            program = parse(f.read())
            return self.run(program)

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
    def _(self, node: Program):
        print(f"Program: {node}")
        self.visit(node.includes)
        self.visit(node.io_variables)
        self.visit(node.statements)

    @visit.register
    def _(self, node: ClassicalDeclaration):
        print(f"Classical declaration: {node}")
        node_type = self.visit(node.type)
        if node.init_expression is not None:
            init_expression = self.visit(node.init_expression)
            init_value = data_manipulation.cast_to(node.type, init_expression)
        elif isinstance(node_type, ArrayType) or (isinstance(node_type, BitType) and node_type.size):
            def build_empty_array(dims):
                if len(dims) == 1:
                    return ArrayLiteral([None] * dims[0].value)
                return ArrayLiteral([build_empty_array(dims[1:])] * dims[0].value)
            dims = node_type.dimensions if isinstance(node_type, ArrayType) else [node_type.size]
            init_value = build_empty_array(dims)
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
    def _(self, node: Cast):
        print(f"Cast: {node}")
        casted = [data_manipulation.cast_to(node.type, self.visit(arg)) for arg in node.arguments]
        return casted[0] if len(casted) == 1 else casted

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
        # name = self.visit(node.name)
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
                        indices.append([element])
                    elif isinstance(element, RangeDefinition):
                        indices.append([element])
                    else:
                        raise NotImplementedError(f"Index {index} not valid for qubit indexing")
        if self.context.get_value(name.name) is None:
            raise NameError(f"Identifier {name.name} is not initialized.")
        return IndexedIdentifier(name, indices)
        # return self.context.get_indexed_value(node)

    @visit.register
    def _(self, node: RangeDefinition):
        print(f"Range definition: {node}")
        start = self.visit(node.start).value if node.start else 0
        end = self.visit(node.end).value if node.end else NotImplementedError()
        step = self.visit(node.step).value if node.step else 1
        return RangeDefinition(IntegerLiteral(start), IntegerLiteral(end), IntegerLiteral(step))

    @visit.register
    def _(self, node: IndexExpression):
        print(f"Index expression: {node}")
        type_width = None
        if isinstance(node.collection, Identifier):
            if not isinstance(self.context.get_type(node.collection.name), ArrayType):
                type_width = self.context.get_type(node.collection.name).size.value
        # else:
        #     type_width = None
        collection = self.visit(node.collection)
        index = self.visit(node.index)
        # type_width = None
        # if not isinstance(collection, ArrayLiteral):
        #     type_width = self.context.get_type(node.collection.name).size.value
        return data_manipulation.get_elements(collection, index, type_width)

    @visit.register
    def _(self, node: QuantumGateDefinition):
        print(f"Quantum gate definition: {node}")
        with self.context.enter_scope():
            for qubit in node.qubits:
                self.context.declare_alias(qubit.name, qubit)

            for param in node.arguments:
                self.context.declare_alias(param.name, param)

            node.body = self.inline_gate_def_body(node.body)  # , node.qubits)
        self.context.add_gate(node.name.name, node)

    def inline_gate_def_body(self, body: List[QuantumStatement]):  # , qubit_map):
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

                        inlined_copy = self.inline_gate_def_body(deepcopy(gate_def.body))

                    inlined_body += modify_body(
                        inlined_copy,
                        is_inverted(statement),
                        ctrl_modifiers,
                        ctrl_qubits,
                    )
        return inlined_body

    @visit.register
    def _(self, node: QuantumGate):
        print(f"Quantum gate: {node}")
        gate_name = node.name.name
        if gate_name == "majority":
            print("in majority")
        node.arguments = [self.visit(arg) for arg in node.arguments]
        # qubits = [self.visit(qubit) for qubit in node.qubits]

        qubits = []
        for qubit in node.qubits:
            if isinstance(qubit, Identifier):
                qubits.append(self.visit(qubit))
            elif isinstance(qubit, IndexedIdentifier):
                dereffed_name = self.visit(qubit.name)
                simplified_indices = self.visit(qubit.indices)
                qubits.append(IndexedIdentifier(dereffed_name, simplified_indices))

        if gate_name == "U":
            # to simplify indices
            qubits = self.visit(qubits)
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
        if isinstance(node.target, IndexedIdentifier):
            node.target.indices = self.visit(node.target.indices)
        if node.target is not None:
            self.context.update_value(node.target, measurement.value)

    @visit.register
    def _(self, node: BranchingStatement):
        print(f"Branching statement: {node}")
        condition = data_manipulation.cast_to(BooleanLiteral, self.visit(node.condition))
        block = node.if_block if condition.value else node.else_block
        for statement in block:
            self.visit(statement)

    @visit.register
    def _(self, node: ForInLoop):
        print(f"For in loop: {node}")
        index = self.visit(node.set_declaration)
        if isinstance(index, RangeDefinition):
            index_values = [
                IntegerLiteral(x) for x in range(index.start.value, index.end.value + 1, index.step.value)
            ]
        # DiscreteSet
        else:
            index_values = index.values
        for i in index_values:
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
