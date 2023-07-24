from copy import deepcopy
from functools import singledispatchmethod
from logging import Logger
from typing import List, Optional, Union

from openqasm3.ast import IntegerLiteral

from ..simulation import Simulation
from ._helpers.casting import cast_to, get_identifier_name, wrap_value_into_literal
from .interpreter import Interpreter
from .parser.openqasm_ast import (
    ArrayLiteral,
    BitType,
    BooleanLiteral,
    ClassicalDeclaration,
    IndexedIdentifier,
    IODeclaration,
    IOKeyword,
    QASMNode,
    QuantumMeasurement,
    QuantumMeasurementStatement,
    QuantumReset,
    QubitDeclaration,
)
from .parser.openqasm_parser import parse
from .program_context import ProgramContext


class NativeInterpreter(Interpreter):
    def __init__(
        self,
        simulation: Simulation,
        context: Optional[ProgramContext] = None,
        logger: Optional[Logger] = None,
    ):
        self.simulation = simulation
        super().__init__(context, logger)

    def simulate(self, source, inputs=None, is_file=False, shots=1):
        if inputs:
            self.context.load_inputs(inputs)

        if is_file:
            with open(source, encoding="utf-8", mode="r") as f:
                source = f.read()

        program = parse(source)
        for _ in range(shots):
            program_copy = deepcopy(program)
            self.visit(program_copy)
            self.context.save_output_values()
            self.context.num_qubits = 0
            self.simulation.reset()
        return self.context.outputs

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
        self.logger.debug(f"Quantum measurement: {node}")
        self.simulation.evolve(self.context.pop_instructions())
        targets = self.context.get_qubits(node.qubit)
        outcome = self.simulation.measure(targets)
        if len(targets) > 1 or (
            isinstance(node.qubit, IndexedIdentifier)
            and not len(node.qubit.indices[0]) == 1
            and isinstance(node.qubit.indices[0], IntegerLiteral)
        ):
            return ArrayLiteral([BooleanLiteral(x) for x in outcome])
        return BooleanLiteral(outcome[0])

    @visit.register
    def _(self, node: QuantumMeasurementStatement) -> Union[BooleanLiteral, ArrayLiteral]:
        self.logger.debug(f"Quantum measurement statement: {node}")
        outcome = self.visit(node.measure)
        current_value = self.context.get_value_by_identifier(node.target)
        result_type = (
            BooleanLiteral
            if isinstance(current_value, BooleanLiteral) or current_value is None
            else BitType(size=IntegerLiteral(len(current_value.values)))
        )
        value = cast_to(result_type, outcome)
        self.context.update_value(node.target, value)

    @visit.register
    def _(self, node: QuantumReset) -> None:
        self.logger.debug(f"Quantum reset: {node}")
        self.simulation.evolve(self.context.pop_instructions())
        targets = self.context.get_qubits(node.qubits)
        outcome = self.simulation.measure(targets)
        for qubit, result in zip(targets, outcome):
            if result:
                self.simulation.flip(qubit)

    @visit.register
    def _(self, node: IODeclaration) -> None:
        self.logger.debug(f"IO Declaration: {node}")
        if node.io_identifier == IOKeyword.output:
            if node.identifier.name not in self.context.outputs:
                self.context.add_output(node.identifier.name)
            self.context.declare_variable(
                node.identifier.name,
                node.type,
            )
        else:  # IOKeyword.input:
            if node.identifier.name not in self.context.inputs:
                raise NameError(f"Missing input variable '{node.identifier.name}'.")
            init_value = wrap_value_into_literal(self.context.inputs[node.identifier.name])
            declaration = ClassicalDeclaration(node.type, node.identifier, init_value)
            self.visit(declaration)