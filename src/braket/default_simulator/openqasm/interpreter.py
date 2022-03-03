from functools import singledispatchmethod

from openqasm3.ast import (
    BinaryExpression,
    BooleanLiteral,
    ClassicalDeclaration,
    Constant,
    ConstantDeclaration,
    DiscreteSet,
    Identifier,
    IndexedIdentifier,
    IndexExpression,
    IntegerLiteral,
    Program,
    QASMNode,
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
            name = target.name.name
            if len(target.indices) > 1:
                raise NotImplementedError("Multiple dimensions of qubit indices")
            index = target.indices[0]
            if len(index) > 1:
                raise NotImplementedError("Multiple qubit indices")
            index = index[0]
            # define start, end, step of range
            if isinstance(index, IntegerLiteral):
                start, end, step = index.value, index.value + 1, 1
            elif isinstance(index, RangeDefinition):
                start = index.start.value if index.start else 0
                end = index.end.value if index.end else context.get_qubit_length(name)
                step = index.step.value if index.step else 1
            else:
                raise NotImplementedError(f"Index {index} not valid for qubit indexing")
            index = slice(start, end, step)
        else:
            name = target.name
            index = None

        context.reset_qubits(name, index)

    @visit.register
    def _(self, node: IndexedIdentifier, context: ProgramContext):
        print(f"Indexed identifier: {node}")
        raise NotImplementedError

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

    # next up: implement indexed identifiers, for loops, gates
