from functools import singledispatchmethod

from openqasm.ast import (
    BinaryExpression,
    BooleanLiteral,
    ClassicalDeclaration,
    Constant,
    ConstantDeclaration,
    Identifier,
    IntegerLiteral,
    Program,
    QASMNode,
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
        context.symbol_table.add_symbol(node.identifier.name, node_type)
        if node.init_expression is not None:
            init_value = self.visit(node.init_expression, context)
            casted = data_manipulation.cast_to(node.type, init_value)
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
            casted = data_manipulation.cast_to(node.type, init_value)
            context.variable_table.add_variable(node.identifier.name, casted)
        else:
            context.variable_table.add_variable(node.identifier.name, None)

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
