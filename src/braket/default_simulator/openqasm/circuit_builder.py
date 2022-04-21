from logging import Logger, getLogger
from typing import Optional

from openqasm3 import parse
from openqasm3.ast import (
    QuantumMeasurement,
    QuantumMeasurementAssignment,
    QuantumReset,
    StringLiteral,
)

from braket.default_simulator.openqasm.circuit_builder_context import CircuitBuilderContext
from braket.default_simulator.openqasm.data_manipulation import singledispatchmethod

from braket.default_simulator.openqasm.interpreter import Interpreter


class CircuitBuilder(Interpreter):

    def __init__(
        self, context: Optional[CircuitBuilderContext] = None, logger: Optional[Logger] = None
    ):
        # context keeps track of all state
        context = context or CircuitBuilderContext()
        logger = logger or getLogger(__name__)
        super().__init__(context, logger)

    def build_circuit(self, source, inputs, is_file):
        if inputs:
            self.context.load_inputs(inputs)

        if is_file:
            with open(source, "r") as f:
                source = f.read()

        program = parse(source)
        self.visit(program)
        return self.context.circuit

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: QuantumReset):
        raise NotImplementedError("Quantum reset not implemented")

    def handle_builtin_unitary(self, arguments, qubits, modifiers):
        self.context.add_builtin_unitary(
            arguments,
            qubits,
            modifiers,
        )

    def handle_phase(self, phase, qubits=None):
        self.context.add_phase(phase, qubits)

    @visit.register
    def _(self, node: QuantumMeasurement):
        self.logger.debug(f"Quantum measurement: {node}")
        qubits = self.visit(node.qubit)
        return StringLiteral(self.context.measure_qubits(qubits))

    @visit.register
    def _(self, node: QuantumMeasurementAssignment):
        raise NotImplementedError("Measurements not implemented")
