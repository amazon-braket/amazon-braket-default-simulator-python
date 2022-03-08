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
    QuantumPhase,
    QuantumReset,
    QubitDeclaration,
    RangeDefinition,
    RealLiteral,
    StringLiteral,
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
        qubits = self.visit(node.qubits, context)
        context.reset_qubits(qubits)

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

            bound_gate_call = QuantumGate(node.modifiers, node.name, node.arguments, qubits)
            if gate_name == "U":
                return [bound_gate_call]

            gate_def_body = deepcopy(gate_def.body)
            inv_modifier = QuantumGateModifier(GateModifierName.inv, None)
            num_inv_modifiers = node.modifiers.count(inv_modifier)
            if num_inv_modifiers % 2:
                # reverse body and invert all gates
                gate_def_body = list(reversed(gate_def_body))
                for statement in gate_def_body:
                    statement.modifiers.insert(0, inv_modifier)
            ctrl_modifiers = [
                self.visit(mod, context)
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
                    if isinstance(statement, QuantumGate):
                        self.visit(statement, context)
                    elif isinstance(statement, QuantumPhase):
                        phase = statement.argument.value
                        context.apply_phase(phase, qubits)
                context.pop_scope()

    @visit.register
    def _(self, node: QuantumPhase, context: ProgramContext):
        print(f"Quantum phase: {node}")
        phase = self.visit(node.argument, context)

        inv_modifier = QuantumGateModifier(GateModifierName.inv, None)
        num_inv_modifiers = node.quantum_gate_modifiers.count(inv_modifier)
        if num_inv_modifiers % 2:
            phase.value *= -1

        ctrl_modifiers = [
            self.visit(mod, context)
            for mod in node.quantum_gate_modifiers
            if mod.modifier in (GateModifierName.ctrl, GateModifierName.negctrl)
        ]
        if ctrl_modifiers:
            first_ctrl_modifier = ctrl_modifiers[-1]
            if first_ctrl_modifier.modifier == GateModifierName.negctrl:
                raise ValueError("negctrl modifier undefined for gphase operation")
            if first_ctrl_modifier.argument.value == 1:
                ctrl_modifiers.pop()
            else:
                ctrl_modifiers[-1].argument.value -= 1
            return [
                QuantumGate(
                    ctrl_modifiers,
                    Identifier("U"),
                    [
                        IntegerLiteral(0),
                        IntegerLiteral(0),
                        phase,
                    ],
                    node.qubits,
                )
            ]
        else:
            if node.qubits:
                raise ValueError(
                    "Cannot specify qubits for a global phase instruction "
                    "unless using the controlled global phase gate."
                )
        return [QuantumPhase([], phase, [])]

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
