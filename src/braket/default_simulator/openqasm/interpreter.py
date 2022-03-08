from copy import deepcopy
from functools import singledispatchmethod

from openqasm3.ast import (
    BinaryExpression,
    BooleanLiteral,
    BranchingStatement,
    ClassicalDeclaration,
    Constant,
    ConstantDeclaration,
    DiscreteSet,
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
    QuantumReset,
    QubitDeclaration,
    RangeDefinition,
    RealLiteral,
    UnaryExpression,
)

from braket.default_simulator.openqasm import data_manipulation
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
        if context.identifier_context == context.IdentifierContext.CLASSICAL:
            if context.get_value(node.name) is None:
                raise NameError(f"Identifier {node.name} is not initialized.")
            return context.get_value(node.name)

    @visit.register
    def _(self, node: QubitDeclaration, context: ProgramContext):
        print(f"Qubit declaration: {node}")
        size = self.visit(node.size, context).value if node.size else 1
        context.add_qubits(node.qubit.name, size)

    @visit.register
    def _(self, node: QuantumReset, context: ProgramContext):
        print(f"Quantum reset: {node}")
        target = node.qubits
        if isinstance(target, IndexedIdentifier):
            target = self.visit(target, context)
            range_def = target.indices[0][0]
            name = target.name.name
            index = slice(range_def.start, range_def.end, range_def.step)
        else:
            name = target.name
            index = None

        context.reset_qubits(name, index)

    @visit.register
    def _(self, node: IndexedIdentifier, context: ProgramContext):
        print(f"Indexed identifier: {node}")
        name = node.name
        if name.name not in context.qubit_mapping:
            raise NotImplementedError("Indexed identifier only implemented for qubits")
        if isinstance(node.indices[0], DiscreteSet):
            raise NotImplementedError("Indexed identifier with descrete set")
        indices = [
            [self.visit(index_element, context) for index_element in index]
            for index in node.indices
        ]
        if len(indices) > 1:
            raise NotImplementedError("Multiple dimensions of qubit indices")
        index = indices[0]
        if len(index) > 1:
            raise NotImplementedError("Multiple qubit indices")
        index = index[0]
        # define start, end, step of range
        if isinstance(index, IntegerLiteral):
            start, end, step = index.value, index.value + 1, 1
        elif isinstance(index, RangeDefinition):
            start = index.start.value if index.start else 0
            end = index.end.value if index.end else context.get_qubit_length(name.name)
            step = index.step.value if index.step else 1
        else:
            raise NotImplementedError(f"Index {index} not valid for qubit indexing")
        return IndexedIdentifier(name, [[RangeDefinition(start, end, step)]])

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
        context.push_scope()
        context.execution_context = ProgramContext.ExecutionContext.GATE_DEF

        for qubit in node.qubits:
            context.declare_alias(qubit.name, qubit)

        for param in node.arguments:
            context.declare_alias(param.name, param)

        node.body = sum((self.visit(statement, context) for statement in node.body), [])

        context.pop_scope()
        context.execution_context = ProgramContext.ExecutionContext.BASE
        context.add_gate(node.name.name, node)

    @visit.register
    def _(self, node: QuantumGate, context: ProgramContext):
        print(f"Quantum gate: {node}")
        gate_name = node.name.name
        node.arguments = [self.visit(arg, context) for arg in node.arguments]
        qubits = [self.visit(qubit, context) for qubit in node.qubits]
        print(qubits)

        if gate_name != "U":
            context.push_scope()
            gate_def = context.get_gate_definition(gate_name)

            num_ctrl = 0
            for mod in node.modifiers:
                if mod.modifier in (GateModifierName.ctrl, GateModifierName.negctrl):
                    num_ctrl += (
                        self.visit(mod.argument, context).value if mod.argument is not None else 1
                    )

            ctrl_qubits, gate_qubits = qubits[:num_ctrl], qubits[num_ctrl:]

            for qubit_called, qubit_defined in zip(gate_qubits, gate_def.qubits):
                context.declare_alias(qubit_defined.name, qubit_called)

            for param_called, param_defined in zip(node.arguments, gate_def.arguments):
                context.declare_alias(param_defined.name, param_called)

        if context.execution_context == ProgramContext.ExecutionContext.GATE_DEF:
            # flatten nested gate calls while replacing qubits from definition with
            # qubits from call

            if gate_name == "U":
                return [
                    QuantumGate(
                        node.modifiers,
                        node.name,
                        node.arguments,
                        qubits,
                    )
                ]

            gate_def_body = deepcopy(gate_def.body)
            inv_modifier = QuantumGateModifier(GateModifierName.inv, None)
            num_inv_modifiers = node.modifiers.count(inv_modifier)
            if num_inv_modifiers % 2:
                # reverse body and invert all gates
                gate_def_body = list(reversed(gate_def_body))
                for statement in gate_def_body:
                    if isinstance(statement, QuantumGate):
                        statement.modifiers.insert(0, inv_modifier)
            ctrl_modifiers = [
                mod
                for mod in node.modifiers
                if mod.modifier in (GateModifierName.ctrl, GateModifierName.negctrl)
            ]
            visited_body = []
            for statement in gate_def_body:
                if isinstance(statement, QuantumGate):
                    statement.modifiers = ctrl_modifiers + statement.modifiers
                    statement.qubits = ctrl_qubits + statement.qubits
                visited_body += self.visit(statement, context)

            context.pop_scope()
            return visited_body
        else:
            qubits = [self.visit(qubit, context) for qubit in node.qubits]

            if gate_name == "U":
                context.execute_builtin_unitary(
                    node.arguments,
                    qubits,
                    node.modifiers,
                )
            else:
                for statement in deepcopy(gate_def.body):
                    self.visit(statement, context)
                context.pop_scope()

    @visit.register
    def _(self, node: QuantumMeasurement, context: ProgramContext):
        print(f"Quantum measurement: {node}")
        target = node.qubit
        if isinstance(target, IndexedIdentifier):
            target = self.visit(target, context)
            range_def = target.indices[0][0]
            name = target.name.name
            index = slice(range_def.start, range_def.end, range_def.step)
        else:
            name = target.name
            index = None

        return context.measure_qubits(name, index)

    @visit.register
    def _(self, node: QuantumMeasurementAssignment, context: ProgramContext):
        print(f"Quantum measurement assignment: {node}")
        measurement = self.visit(node.measure_instruction, context)
        print("TARMEA: ", node.target, measurement)
        if node.target is not None:
            context.update_value(node.target.name, measurement)

    @visit.register
    def _(self, node: BranchingStatement, context: ProgramContext):
        print(f"Branching statement: {node}")
        block = node.if_block if self.visit(node.condition, context) else node.else_block
        for statement in block:
            self.visit(statement, context)
