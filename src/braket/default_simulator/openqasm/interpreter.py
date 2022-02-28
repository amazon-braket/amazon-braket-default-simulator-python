from functools import singledispatchmethod

from openqasm.ast import (
    BinaryExpression,
    BooleanLiteral,
    ClassicalDeclaration,
    Constant,
    ConstantDeclaration,
    IntegerLiteral,
    Program,
    RealLiteral,
    UnaryExpression,
)

from braket.default_simulator.openqasm.data_manipulation import DataManipulation
from braket.default_simulator.openqasm.program_context import ProgramContext
from braket.default_simulator.openqasm.visitor import QASMTransformer


def _register(self, cls, method=None):
    if hasattr(cls, "__func__"):
        setattr(cls, "__annotations__", cls.__func__.__annotations__)
    return self.dispatcher.register(cls, func=method)


singledispatchmethod.register = _register


class Interpreter(QASMTransformer):
    @singledispatchmethod
    def visit(self, node, context=None):
        return super().visit(node, context)

    @visit.register
    def _(self, node: Program) -> ProgramContext:
        """Returns ProgramContext rather than the consumed Program node"""
        context = ProgramContext()
        super().visit(node, context)
        return context

    @visit.register
    def _(self, node: ClassicalDeclaration, context: ProgramContext):
        print(f"Classical declaration: {node}")
        node_type = self.visit(node.type, context)
        context.symbol_table.add_symbol(node.identifier.name, node_type)
        if node.init_expression is not None:
            init_value = self.visit(node.init_expression, context)
            casted = DataManipulation.cast_to(node.type, init_value)
            context.variable_table.add_variable(node.identifier.name, casted)
        else:
            context.variable_table.add_variable(node.identifier.name, None)

    @visit.register
    def _(self, node: ConstantDeclaration, context: ProgramContext):
        print(f"Constant declaration: {node}")
        node_type = self.visit(node.type, context)
        context.symbol_table.add_symbol(node.identifier.name, node_type, True)
        if node.init_expression is not None:
            init_value = self.visit(node.init_expression, context)
            casted = DataManipulation.cast_to(node.type, init_value)
            context.variable_table.add_variable(node.identifier.name, casted)
        else:
            context.variable_table.add_variable(node.identifier.name, None)

    @visit.register
    def _(self, node: BinaryExpression, context: ProgramContext):
        print(f"Binary expression: {node}")
        lhs = self.visit(node.lhs, context)
        rhs = self.visit(node.rhs, context)
        op = self.visit(node.op, context)
        result_type = DataManipulation.resolve_result_type(type(lhs), type(rhs))
        lhs = DataManipulation.cast_to(result_type, lhs)
        rhs = DataManipulation.cast_to(result_type, rhs)
        return DataManipulation.evaluate_binary_expression(lhs, rhs, op)

    @visit.register
    def _(self, node: UnaryExpression, context: ProgramContext):
        print(f"Unary expression: {node}")
        expression = self.visit(node.expression, context)
        op = self.visit(node.op, context)
        return DataManipulation.evaluate_unary_expression(expression, op)

    @visit.register
    def _(self, node: Constant, context: ProgramContext):
        return DataManipulation.evaluate_constant(node)

    @visit.register(BooleanLiteral)
    @visit.register(IntegerLiteral)
    @visit.register(RealLiteral)
    def _(self, node, context: ProgramContext):
        return node
