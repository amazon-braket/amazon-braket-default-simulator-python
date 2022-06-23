import numpy as np
from antlr4 import CommonTokenStream, InputStream
from braket.ir.jaqcd import (
    Amplitude,
    DensityMatrix,
    Expectation,
    Probability,
    Sample,
    StateVector,
    Variance,
)
from openqasm3.parser import parse

from .dist.BraketPragmasLexer import BraketPragmasLexer
from .dist.BraketPragmasParser import BraketPragmasParser
from .dist.BraketPragmasVisitor import BraketPragmasVisitor


class BraketPragmaNodeVisitor(BraketPragmasVisitor):
    """
    This is a visitor for the BraketPragmas grammar. This class will be replaced
    when the parser is updated. Feel free to skim over in review.
    """

    def __init__(self, qubit_table):
        self.qubit_table = qubit_table

    def visitNoArgResultType(self, ctx: BraketPragmasParser.NoArgResultTypeContext):
        result_type = ctx.getChild(0).getText()
        no_arg_result_type_map = {
            "state_vector": StateVector,
        }
        return no_arg_result_type_map[result_type]()

    def visitOptionalMultiTargetResultType(
        self, ctx: BraketPragmasParser.OptionalMultiTargetResultTypeContext
    ):
        result_type = ctx.getChild(0).getText()
        optional_multitarget_result_type_map = {
            "probability": Probability,
            "density_matrix": DensityMatrix,
        }
        targets = self.visit(ctx.getChild(1)) if ctx.getChild(1) is not None else None
        return optional_multitarget_result_type_map[result_type](targets=targets)

    def visitMultiTarget(self, ctx: BraketPragmasParser.MultiTargetContext):
        parsable = f"target {''.join(x.getText() for x in ctx.getChildren())};"
        parsed_statement = parse(parsable)
        target_identifiers = parsed_statement.statements[0].qubits
        target = sum(
            (self.qubit_table.get_by_identifier(identifier) for identifier in target_identifiers),
            (),
        )
        return target

    def visitMultiStateResultType(self, ctx: BraketPragmasParser.MultiStateResultTypeContext):
        result_type = ctx.getChild(0).getText()
        multistate_result_type_map = {
            "amplitude": Amplitude,
        }
        states = self.visit(ctx.getChild(1))
        return multistate_result_type_map[result_type](states=states)

    def visitMultiState(self, ctx: BraketPragmasParser.MultiStateContext):
        # unquote and skip commas
        states = [x.getText()[1:-1] for x in list(ctx.getChildren())[::2]]
        return states

    def visitObservableResultType(self, ctx: BraketPragmasParser.ObservableResultTypeContext):
        result_type = ctx.getChild(0).getText()
        observable_result_type_map = {
            "expectation": Expectation,
            "sample": Sample,
            "variance": Variance,
        }
        observables, targets = self.visit(ctx.getChild(1))
        obs = observable_result_type_map[result_type](targets=targets, observable=observables)
        return obs

    def visitStandardObservable(self, ctx: BraketPragmasParser.StandardObservableContext):
        observable = ctx.getChild(0).getText()
        target_tuple = self.visit(ctx.getChild(2))
        if len(target_tuple) != 1:
            raise ValueError("Standard observable target must be exactly 1 qubit.")
        return (observable,), target_tuple

    def visitTensorProductObservable(self, ctx: BraketPragmasParser.TensorProductObservableContext):
        observables, targets = zip(
            *(self.visit(ctx.getChild(i)) for i in range(0, ctx.getChildCount(), 2))
        )
        observables = sum(observables, ())
        targets = sum(targets, ())
        return observables, targets

    def visitHermitianObservable(self, ctx: BraketPragmasParser.HermitianObservableContext):
        matrix = [
            [self.visit(ctx.getChild(4)), self.visit(ctx.getChild(6))],
            [self.visit(ctx.getChild(10)), self.visit(ctx.getChild(12))],
        ]
        target = self.visit(ctx.getChild(16))
        return (matrix,), target

    def visitIndexedIdentifier(self, ctx: BraketPragmasParser.IndexedIdentifierContext):
        parsable = f"target {''.join(x.getText() for x in ctx.getChildren())};"
        parsed_statement = parse(parsable)
        identifier = parsed_statement.statements[0].qubits[0]
        target = self.qubit_table.get_by_identifier(identifier)
        return target

    def visitComplexOneValue(self, ctx: BraketPragmasParser.ComplexOneValueContext):
        sign = -1 if ctx.neg else 1
        value = ctx.value.text
        imag = False
        if value.endswith("im"):
            value = value[:-2]
            imag = True
        complex_array = [0, 0]
        complex_array[imag] = sign * float(value)
        return complex_array

    def visitComplexTwoValues(self, ctx: BraketPragmasParser.ComplexTwoValuesContext):
        real = float(ctx.real.text)
        imag = float(ctx.imag.text[:-2])  # exclude "im"
        return [real, imag]

    def visitBraketUnitaryPragma(self, ctx: BraketPragmasParser.BraketUnitaryPragmaContext):
        target = self.visit(ctx.multiTarget())
        matrix = self.visit(ctx.twoDimMatrix())
        return matrix, target

    def visitRow(self, ctx: BraketPragmasParser.RowContext):
        numbers = ctx.children[1::2]
        return [x[0] + x[1] * 1j for x in [self.visit(number) for number in numbers]]

    def visitTwoDimMatrix(self, ctx: BraketPragmasParser.TwoDimMatrixContext):
        rows = [self.visit(row) for row in ctx.children[1::2]]
        if not all(len(r) == len(rows) for r in rows):
            raise TypeError("Not a valid square matrix")
        matrix = np.array(rows)
        return matrix


def parse_braket_pragma(pragma_body: str, qubit_table):
    data = InputStream(pragma_body)
    lexer = BraketPragmasLexer(data)
    stream = CommonTokenStream(lexer)
    parser = BraketPragmasParser(stream)
    tree = parser.braketPragma()
    visited = BraketPragmaNodeVisitor(qubit_table).visit(tree)
    return visited
