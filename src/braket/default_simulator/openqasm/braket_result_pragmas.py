from antlr4 import CommonTokenStream, InputStream
from braket.ir.jaqcd import DensityMatrix, Probability, StateVector, Amplitude
from openqasm3.antlr.qasm3Lexer import qasm3Lexer
from openqasm3.antlr.qasm3Parser import qasm3Parser
from openqasm3.parser import QASMNodeVisitor, parse

from braket.default_simulator.openqasm.dist.BraketPragmasVisitor import BraketPragmasVisitor
from src.braket.default_simulator.openqasm.dist.BraketPragmasLexer import BraketPragmasLexer
from src.braket.default_simulator.openqasm.dist.BraketPragmasParser import BraketPragmasParser

# def parse_identifier(identifier: str):
#     lexer = qasm3Lexer(InputStream(identifier))
#     stream = CommonTokenStream(lexer)
#     parser = qasm3Parser(stream)
#
#     tree = parser.expression()
#
#     return QASMNodeVisitor().visitExpression(tree)


class BraketPragmaNodeVisitor(BraketPragmasVisitor):
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

    def visitMultiStateResultType(self, ctx:BraketPragmasParser.MultiStateResultTypeContext):
        result_type = ctx.getChild(0).getText()
        multistate_result_type_map = {
            "amplitude": Amplitude,
        }
        states = self.visit(ctx.getChild(1))
        return multistate_result_type_map[result_type](states=states)

    def visitMultiState(self, ctx:BraketPragmasParser.MultiStateContext):
        # unquote and skip commas
        states = [x.getText()[1:-1] for x in list(ctx.getChildren())[::2]]
        return states


def parse_braket_pragma(pragma_body: str, qubit_table):
    data = InputStream(pragma_body)
    lexer = BraketPragmasLexer(data)
    stream = CommonTokenStream(lexer)
    parser = BraketPragmasParser(stream)
    tree = parser.braketPragma()
    return BraketPragmaNodeVisitor(qubit_table).visit(tree)


# qubit_table = QubitTable()
# qubit_table["q"] = (0, 1, 2)
#
# # print(parse_braket_pragma("braket result state_vector"))
# # print(parse_braket_pragma("braket result probability"))
# print(parse_braket_pragma("braket result probability q[0], q[2]", qubit_table))
