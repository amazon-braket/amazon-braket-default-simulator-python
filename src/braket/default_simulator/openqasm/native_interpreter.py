from logging import Logger
from pathlib import Path
from typing import Optional, Union, List

from ._helpers.utils import singledispatchmethod
from .interpreter import Interpreter
from .parser.openqasm_ast import Include, QASMNode, QubitDeclaration, QuantumMeasurement, BooleanLiteral, ArrayLiteral, \
    IndexedIdentifier
from .parser.openqasm_parser import parse
from .program_context import ProgramContext
from ..simulation import Simulation


class NativeInterpreter(Interpreter):
    def __init__(
        self,
        simulation: Simulation,
        context: Optional[ProgramContext] = None,
        logger: Optional[Logger] = None,
    ):
        self.simulation = simulation
        super().__init__(context, logger)

    def simulate(self, source, inputs=None, is_file=False):
        if inputs:
            self.context.load_inputs(inputs)

        if is_file:
            with open(source, "r") as f:
                source = f.read()

        program = parse(source)
        self.visit(Include(Path(Path(__file__).parent, "braket_gates.inc")))
        self.visit(program)
        return self.context

    @singledispatchmethod
    def visit(self, node: Union[QASMNode, List[QASMNode]]) -> Optional[QASMNode]:
        """Generic visit function for an AST node"""
        return super().visit(node)

    @visit.register
    def _(self, node: QubitDeclaration) -> None:
        self.logger.debug(f"Qubit declaration: {node}")
        size = self.visit(node.size).value if node.size else 1
        self.context.add_qubits(node.qubit.name, size)
        self.simulation.add_qubits(size)

    @visit.register
    def _(self, node: QuantumMeasurement) -> Union[BooleanLiteral, ArrayLiteral]:
        """Doesn't do anything, but may add more functionality in the future"""
        self.logger.debug(f"Quantum measurement: {node}")
        self.simulation.evolve(self.context.pop_instructions())
        targets = self.context.get_qubits(node.qubit)
        outcome = self.simulation.measure(targets)
        if len(targets) > 1 or isinstance(node.qubit, IndexedIdentifier):
            return ArrayLiteral([BooleanLiteral(x) for x in outcome])
        return BooleanLiteral(outcome[0])


